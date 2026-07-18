"""Async signed REST client for BingX USDT-M perpetual swap.

Signing follows BingX's official reference implementation exactly:
sorted parameters, `timestamp` appended last, HMAC-SHA256 hex of the
UNENCODED parameter string, `&signature=` appended. The URL must carry the
parameters in the same order they were signed (values percent-encoded).
"""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
from hashlib import sha256
from typing import Any
from urllib.parse import quote

import aiohttp

from ..util import RateLimiter, now_ms, safe_float
from .errors import BingXAPIError, BingXError
from .models import BookTop, Candle, ContractSpec, DepthSnapshot

log = logging.getLogger("bingx.rest")


class BingXRest:
    def __init__(
        self,
        base_url: str = "https://open-api.bingx.com",
        api_key: str = "",
        api_secret: str = "",
        recv_window_ms: int = 5000,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret
        self.recv_window_ms = recv_window_ms
        self._session: aiohttp.ClientSession | None = None
        self._time_offset_ms = 0
        # Documented caps: quote endpoints ~1/s per IP, trade ~3/s per IP.
        self._quote_limiter = RateLimiter(rate=0.95, burst=2)
        self._trade_limiter = RateLimiter(rate=2.5, burst=3)

    async def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15, connect=6),
                headers={"User-Agent": "bingxbot/1.0"},
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------ core

    def _sign(self, params: dict[str, Any]) -> tuple[str, str]:
        """Return (unencoded param string incl. timestamp, signature hex)."""
        items = [(k, str(v)) for k, v in sorted(params.items())]
        items.append(("timestamp", str(now_ms() + self._time_offset_ms)))
        raw = "&".join(f"{k}={v}" for k, v in items)
        sig = hmac.new(self.api_secret.encode(), raw.encode(), sha256).hexdigest()
        encoded = "&".join(f"{k}={quote(v, safe='')}" for k, v in items)
        return encoded, sig

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        signed: bool = False,
        retries: int = 3,
    ) -> Any:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        limiter = self._trade_limiter if "/trade/" in path or "/user/" in path else self._quote_limiter
        headers = {}
        if signed:
            if not (self.api_key and self.api_secret):
                raise BingXError("API keys required for this endpoint but not configured")
            params.setdefault("recvWindow", self.recv_window_ms)
            query, sig = self._sign(params)
            url = f"{self.base_url}{path}?{query}&signature={sig}"
            headers["X-BX-APIKEY"] = self.api_key
        else:
            query = "&".join(f"{k}={quote(str(v), safe='')}" for k, v in params.items())
            url = f"{self.base_url}{path}" + (f"?{query}" if query else "")

        # Never blindly retry order placement: a timeout may still have filled.
        can_retry = method == "GET"
        attempt = 0
        while True:
            attempt += 1
            await limiter.acquire()
            try:
                sess = await self.session()
                async with sess.request(method, url, headers=headers) as resp:
                    text = await resp.text()
                    if resp.status >= 500:
                        raise BingXError(f"HTTP {resp.status}: {text[:200]}")
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError as e:
                        raise BingXError(f"non-JSON response ({resp.status}): {text[:200]}") from e
                    code = payload.get("code", 0)
                    if code not in (0, 200):
                        raise BingXAPIError(code, payload.get("msg", ""), path)
                    return payload.get("data", payload)
            except (aiohttp.ClientError, asyncio.TimeoutError, BingXError) as e:
                if isinstance(e, BingXAPIError) or not can_retry or attempt > retries:
                    raise
                delay = min(2 ** attempt, 10)
                log.warning("GET %s failed (%s), retry %d in %ss", path, e, attempt, delay)
                await asyncio.sleep(delay)

    async def sync_time(self) -> int:
        data = await self._request("GET", "/openApi/swap/v2/server/time")
        server = int(data.get("serverTime", now_ms())) if isinstance(data, dict) else int(data)
        self._time_offset_ms = server - now_ms()
        log.info("server time offset: %d ms", self._time_offset_ms)
        return self._time_offset_ms

    # ---------------------------------------------------------------- market

    async def contracts(self) -> dict[str, ContractSpec]:
        data = await self._request("GET", "/openApi/swap/v2/quote/contracts")
        specs: dict[str, ContractSpec] = {}
        for row in data or []:
            sym = row.get("symbol", "")
            if not sym:
                continue
            specs[sym] = ContractSpec(
                symbol=sym,
                qty_precision=int(row.get("quantityPrecision", 4)),
                price_precision=int(row.get("pricePrecision", 2)),
                min_qty=safe_float(row.get("tradeMinQuantity"), 0.0001),
                min_notional_usdt=safe_float(row.get("tradeMinUSDT"), 2.0),
                max_long_leverage=int(safe_float(row.get("maxLongLeverage"), 100)),
                max_short_leverage=int(safe_float(row.get("maxShortLeverage"), 100)),
                maker_fee=safe_float(row.get("makerFeeRate"), 0.0002),
                taker_fee=safe_float(row.get("takerFeeRate"), 0.0005),
            )
        return specs

    @staticmethod
    def _parse_kline_row(row: Any) -> Candle | None:
        if isinstance(row, dict):
            ts = int(safe_float(row.get("time") or row.get("t") or row.get("T")))
            return Candle(
                ts=ts,
                open=safe_float(row.get("open") or row.get("o")),
                high=safe_float(row.get("high") or row.get("h")),
                low=safe_float(row.get("low") or row.get("l")),
                close=safe_float(row.get("close") or row.get("c")),
                volume=safe_float(row.get("volume") or row.get("v")),
            )
        if isinstance(row, (list, tuple)) and len(row) >= 6:
            return Candle(
                ts=int(safe_float(row[0])),
                open=safe_float(row[1]),
                high=safe_float(row[2]),
                low=safe_float(row[3]),
                close=safe_float(row[4]),
                volume=safe_float(row[5]),
            )
        return None

    async def klines(
        self,
        symbol: str,
        interval: str = "1m",
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = 1440,
    ) -> list[Candle]:
        data = await self._request(
            "GET",
            "/openApi/swap/v3/quote/klines",
            {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": min(limit, 1440),
            },
        )
        out = [c for row in (data or []) if (c := self._parse_kline_row(row))]
        out.sort(key=lambda c: c.ts)
        return out

    async def depth(self, symbol: str, limit: int = 20) -> DepthSnapshot:
        data = await self._request("GET", "/openApi/swap/v2/quote/depth", {"symbol": symbol, "limit": limit})
        bids = [(safe_float(p), safe_float(q)) for p, q in data.get("bids", [])]
        asks = [(safe_float(p), safe_float(q)) for p, q in data.get("asks", [])]
        bids.sort(key=lambda x: -x[0])
        asks.sort(key=lambda x: x[0])
        return DepthSnapshot(ts=int(data.get("T", now_ms())), bids=bids, asks=asks)

    async def book_ticker(self, symbol: str) -> BookTop:
        data = await self._request("GET", "/openApi/swap/v2/quote/bookTicker", {"symbol": symbol})
        row = data.get("book_ticker", data) if isinstance(data, dict) else data
        return BookTop(
            ts=int(safe_float(row.get("time"), now_ms())),
            bid=safe_float(row.get("bidPrice") or row.get("bid_price") or row.get("b")),
            bid_qty=safe_float(row.get("bidQty") or row.get("B")),
            ask=safe_float(row.get("askPrice") or row.get("ask_price") or row.get("a")),
            ask_qty=safe_float(row.get("askQty") or row.get("A")),
        )

    async def latest_price(self, symbol: str) -> float:
        data = await self._request("GET", "/openApi/swap/v1/ticker/price", {"symbol": symbol})
        row = data[0] if isinstance(data, list) and data else data
        return safe_float(row.get("price"))

    async def premium_index(self, symbol: str) -> dict:
        """Mark price, index price and current funding rate."""
        data = await self._request("GET", "/openApi/swap/v2/quote/premiumIndex", {"symbol": symbol})
        row = data[0] if isinstance(data, list) and data else data
        return {
            "mark": safe_float(row.get("markPrice")),
            "index": safe_float(row.get("indexPrice")),
            "funding_rate": safe_float(row.get("lastFundingRate")),
            "next_funding_time": int(safe_float(row.get("nextFundingTime"))),
        }

    async def open_interest(self, symbol: str) -> float:
        data = await self._request("GET", "/openApi/swap/v2/quote/openInterest", {"symbol": symbol})
        row = data[0] if isinstance(data, list) and data else data
        return safe_float(row.get("openInterest"))

    async def premium_index_all(self) -> list[dict]:
        """Funding rate + mark price for EVERY perp in one call — the market
        radar's raw feed. Omitting `symbol` returns the whole universe."""
        data = await self._request("GET", "/openApi/swap/v2/quote/premiumIndex")
        rows = data if isinstance(data, list) else [data] if isinstance(data, dict) else []
        out = []
        for row in rows:
            sym = row.get("symbol", "")
            if not sym:
                continue
            out.append({
                "symbol": sym,
                "mark": safe_float(row.get("markPrice")),
                "funding_rate": safe_float(row.get("lastFundingRate")),
                "next_funding_time": int(safe_float(row.get("nextFundingTime"))),
            })
        return out

    async def funding_rate_history(self, symbol: str, limit: int = 1000) -> list[dict]:
        """Historical funding prints (rate + settlement time), oldest first —
        the raw material for measuring the carry edge instead of assuming it."""
        data = await self._request("GET", "/openApi/swap/v2/quote/fundingRate",
                                   {"symbol": symbol, "limit": min(limit, 1000)})
        rows = data if isinstance(data, list) else []
        out = [{"ts": int(safe_float(r.get("fundingTime"))),
                "rate": safe_float(r.get("fundingRate"))}
               for r in rows if r.get("fundingTime") is not None]
        out.sort(key=lambda r: r["ts"])
        return out

    async def tickers_24h(self) -> list[dict]:
        """24h stats for every perp (volume + change) — feeds universe ranking."""
        data = await self._request("GET", "/openApi/swap/v2/quote/ticker")
        rows = data if isinstance(data, list) else [data] if isinstance(data, dict) else []
        out = []
        for row in rows:
            sym = row.get("symbol", "")
            if not sym:
                continue
            out.append({
                "symbol": sym,
                "last": safe_float(row.get("lastPrice")),
                "quote_volume": safe_float(row.get("quoteVolume")),
                "change_pct": safe_float(row.get("priceChangePercent")),
            })
        return out

    # --------------------------------------------------------------- account

    async def balance(self) -> dict:
        """Normalized USDT-margin balance: {balance, equity, available, unrealized}."""
        data = await self._request("GET", "/openApi/swap/v3/user/balance", signed=True)
        row: dict = {}
        if isinstance(data, list):
            row = next((r for r in data if r.get("asset") in ("USDT", "VST")), data[0] if data else {})
        elif isinstance(data, dict):
            row = data.get("balance", data)
        return {
            "asset": row.get("asset", "USDT"),
            "balance": safe_float(row.get("balance")),
            "equity": safe_float(row.get("equity")),
            "available": safe_float(row.get("availableMargin")),
            "unrealized": safe_float(row.get("unrealizedProfit")),
        }

    async def positions(self, symbol: str | None = None) -> list[dict]:
        data = await self._request("GET", "/openApi/swap/v2/user/positions", {"symbol": symbol}, signed=True)
        return data or []

    async def commission_rate(self) -> dict:
        data = await self._request("GET", "/openApi/swap/v2/user/commissionRate", signed=True)
        row = data.get("commission", data) if isinstance(data, dict) else {}
        return {
            "taker": safe_float(row.get("takerCommissionRate"), 0.0005),
            "maker": safe_float(row.get("makerCommissionRate"), 0.0002),
        }

    # ----------------------------------------------------------------- trade

    async def set_position_mode(self, dual_side: bool = True) -> Any:
        return await self._request(
            "POST", "/openApi/swap/v1/positionSide/dual",
            {"dualSidePosition": "true" if dual_side else "false"}, signed=True,
        )

    async def set_leverage(self, symbol: str, side: str, leverage: int) -> Any:
        return await self._request(
            "POST", "/openApi/swap/v2/trade/leverage",
            {"symbol": symbol, "side": side, "leverage": leverage}, signed=True,
        )

    async def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED") -> Any:
        return await self._request(
            "POST", "/openApi/swap/v2/trade/marginType",
            {"symbol": symbol, "marginType": margin_type}, signed=True,
        )

    async def place_order(
        self,
        symbol: str,
        side: str,                 # BUY | SELL
        position_side: str,        # LONG | SHORT
        order_type: str = "MARKET",
        quantity: float | None = None,
        price: float | None = None,
        stop_loss: dict | None = None,
        take_profit: dict | None = None,
        client_order_id: str | None = None,
        reduce_only: bool | None = None,
        time_in_force: str | None = None,
    ) -> dict:
        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "positionSide": position_side,
            "type": order_type,
            "quantity": quantity,
            "price": price,
            "clientOrderId": client_order_id,
            "timeInForce": time_in_force,
        }
        if reduce_only is not None:
            params["reduceOnly"] = "true" if reduce_only else "false"
        if stop_loss:
            params["stopLoss"] = json.dumps(stop_loss, separators=(",", ":"))
        if take_profit:
            params["takeProfit"] = json.dumps(take_profit, separators=(",", ":"))
        data = await self._request("POST", "/openApi/swap/v2/trade/order", params, signed=True)
        return data.get("order", data) if isinstance(data, dict) else data

    async def get_order(self, symbol: str, order_id: str) -> dict:
        data = await self._request(
            "GET", "/openApi/swap/v2/trade/order",
            {"symbol": symbol, "orderId": order_id}, signed=True,
        )
        return data.get("order", data) if isinstance(data, dict) else data

    async def cancel_order(self, symbol: str, order_id: str) -> Any:
        return await self._request(
            "DELETE", "/openApi/swap/v2/trade/order",
            {"symbol": symbol, "orderId": order_id}, signed=True,
        )

    async def cancel_all_orders(self, symbol: str) -> Any:
        return await self._request(
            "DELETE", "/openApi/swap/v2/trade/allOpenOrders", {"symbol": symbol}, signed=True,
        )

    async def open_orders(self, symbol: str | None = None) -> list[dict]:
        data = await self._request("GET", "/openApi/swap/v2/trade/openOrders", {"symbol": symbol}, signed=True)
        if isinstance(data, dict):
            return data.get("orders", []) or []
        return data or []

    async def close_all_positions(self, symbol: str | None = None) -> Any:
        return await self._request(
            "POST", "/openApi/swap/v2/trade/closeAllPositions", {"symbol": symbol}, signed=True,
        )

    # ------------------------------------------------------------ user stream

    async def create_listen_key(self) -> str:
        data = await self._request("POST", "/openApi/user/auth/userDataStream", signed=True)
        key = data.get("listenKey", "") if isinstance(data, dict) else ""
        if not key:
            raise BingXError(f"no listenKey in response: {data}")
        return key

    async def keepalive_listen_key(self, listen_key: str) -> None:
        await self._request("PUT", "/openApi/user/auth/userDataStream", {"listenKey": listen_key}, signed=True)
