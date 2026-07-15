"""BingX swap WebSocket streams (market + user data).

Protocol: every frame is GZIP-compressed. After decompression the server may
send the literal text `Ping`, which must be answered with `Pong` or the
connection is dropped. Market subscriptions are JSON:
    {"id": "<uuid>", "reqType": "sub", "dataType": "BTC-USDT@trade"}
"""
from __future__ import annotations

import asyncio
import gzip
import json
import logging
import time
import uuid
from typing import Awaitable, Callable

import aiohttp

from ..util import now_ms, safe_float
from .models import BookTop, Candle, DepthSnapshot, Tick
from .rest import BingXRest

log = logging.getLogger("bingx.ws")

STALE_AFTER_S = 40          # no frames for this long -> force reconnect
RECONNECT_MAX_DELAY = 30


def _decode(msg: aiohttp.WSMessage) -> str | None:
    if msg.type == aiohttp.WSMsgType.BINARY:
        try:
            return gzip.decompress(msg.data).decode("utf-8")
        except (OSError, UnicodeDecodeError):
            return None
    if msg.type == aiohttp.WSMsgType.TEXT:
        return msg.data
    return None


class _BaseWS:
    """Shared connect/reconnect/heartbeat loop."""

    name = "ws"

    def __init__(self, url: str):
        self.url = url
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._task: asyncio.Task | None = None
        self._stopped = False
        self.last_msg_ts = 0.0
        self.connected = False

    async def start(self) -> None:
        self._stopped = False
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name=f"{self.name}-loop")

    async def stop(self) -> None:
        self._stopped = True
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        if self._session and not self._session.closed:
            await self._session.close()
        self.connected = False

    async def _connect_url(self) -> str:
        return self.url

    async def _on_connected(self) -> None:  # subscribe etc.
        pass

    async def _on_payload(self, payload: str) -> None:
        raise NotImplementedError

    async def _run(self) -> None:
        delay = 1
        while not self._stopped:
            try:
                if self._session is None or self._session.closed:
                    self._session = aiohttp.ClientSession()
                url = await self._connect_url()
                async with self._session.ws_connect(url, heartbeat=20, max_msg_size=8 * 2**20) as ws:
                    self._ws = ws
                    self.connected = True
                    self.last_msg_ts = time.monotonic()
                    delay = 1
                    log.info("%s connected", self.name)
                    await self._on_connected()
                    watchdog = asyncio.create_task(self._watchdog())
                    try:
                        async for msg in ws:
                            self.last_msg_ts = time.monotonic()
                            if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break
                            payload = _decode(msg)
                            if payload is None:
                                continue
                            if payload == "Ping":
                                await ws.send_str("Pong")
                                continue
                            await self._on_payload(payload)
                    finally:
                        watchdog.cancel()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - reconnect on anything
                if not self._stopped:
                    log.warning("%s error: %s", self.name, e)
            self.connected = False
            if self._stopped:
                break
            log.info("%s reconnecting in %ss", self.name, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, RECONNECT_MAX_DELAY)

    async def _watchdog(self) -> None:
        while True:
            await asyncio.sleep(5)
            if time.monotonic() - self.last_msg_ts > STALE_AFTER_S:
                log.warning("%s stale (no frames %ss) -> reconnect", self.name, STALE_AFTER_S)
                if self._ws and not self._ws.closed:
                    await self._ws.close()
                return


class BingXMarketWS(_BaseWS):
    """Public market stream: klines, trades, book ticker, partial depth."""

    name = "market-ws"

    def __init__(
        self,
        url: str,
        on_kline: Callable[[str, Candle], Awaitable[None]] | None = None,
        on_tick: Callable[[str, Tick], Awaitable[None]] | None = None,
        on_book: Callable[[str, BookTop], Awaitable[None]] | None = None,
        on_depth: Callable[[str, DepthSnapshot], Awaitable[None]] | None = None,
    ):
        super().__init__(url)
        self.on_kline = on_kline
        self.on_tick = on_tick
        self.on_book = on_book
        self.on_depth = on_depth
        self._channels: set[str] = set()
        # kline rollover detection: last open-time per (symbol, interval)
        self._kline_open: dict[str, int] = {}
        self._kline_last: dict[str, Candle] = {}

    def subscribe_symbol(self, symbol: str, interval: str = "1m", depth_level: int = 20) -> None:
        self._channels.update(
            {
                f"{symbol}@kline_{interval}",
                f"{symbol}@trade",
                f"{symbol}@bookTicker",
                f"{symbol}@depth{depth_level}@500ms",
            }
        )

    async def _on_connected(self) -> None:
        for ch in sorted(self._channels):
            await self._ws.send_str(
                json.dumps({"id": str(uuid.uuid4()), "reqType": "sub", "dataType": ch})
            )
            await asyncio.sleep(0.05)
        log.info("subscribed %d channels", len(self._channels))

    async def _on_payload(self, payload: str) -> None:
        try:
            msg = json.loads(payload)
        except json.JSONDecodeError:
            return
        data_type: str = msg.get("dataType", "")
        data = msg.get("data")
        if not data_type or data is None:
            return
        symbol = data_type.split("@", 1)[0]
        try:
            if "@kline_" in data_type:
                await self._handle_kline(symbol, data)
            elif data_type.endswith("@trade"):
                await self._handle_trades(symbol, data)
            elif data_type.endswith("@bookTicker"):
                await self._handle_book(symbol, data)
            elif "@depth" in data_type:
                await self._handle_depth(symbol, data)
        except Exception:  # noqa: BLE001
            log.exception("handler failed for %s", data_type)

    async def _handle_kline(self, symbol: str, data) -> None:
        rows = data if isinstance(data, list) else [data]
        for row in rows:
            k = row.get("K", row) if isinstance(row, dict) else None
            if not isinstance(k, dict):
                continue
            ts = int(safe_float(k.get("T") or k.get("t") or row.get("T")))
            candle = Candle(
                ts=ts,
                open=safe_float(k.get("o")),
                high=safe_float(k.get("h")),
                low=safe_float(k.get("l")),
                close=safe_float(k.get("c")),
                volume=safe_float(k.get("v")),
                closed=False,
            )
            prev_ts = self._kline_open.get(symbol)
            if prev_ts is not None and ts > prev_ts:
                # New bar started -> the previous one just closed.
                prev = self._kline_last.get(symbol)
                if prev is not None and self.on_kline:
                    prev.closed = True
                    await self.on_kline(symbol, prev)
            self._kline_open[symbol] = ts
            self._kline_last[symbol] = candle
            if self.on_kline:
                await self.on_kline(symbol, candle)

    async def _handle_trades(self, symbol: str, data) -> None:
        if not self.on_tick:
            return
        rows = data if isinstance(data, list) else [data]
        for row in rows:
            if not isinstance(row, dict):
                continue
            await self.on_tick(
                symbol,
                Tick(
                    ts=int(safe_float(row.get("T") or row.get("t"), now_ms())),
                    price=safe_float(row.get("p")),
                    qty=safe_float(row.get("q")),
                    is_buyer_maker=bool(row.get("m", False)),
                ),
            )

    async def _handle_book(self, symbol: str, data) -> None:
        if not self.on_book:
            return
        row = data[0] if isinstance(data, list) and data else data
        if not isinstance(row, dict):
            return
        bid = safe_float(row.get("b") or row.get("bidPrice"))
        ask = safe_float(row.get("a") or row.get("askPrice"))
        if bid <= 0 or ask <= 0:
            return
        await self.on_book(
            symbol,
            BookTop(
                ts=int(safe_float(row.get("T") or row.get("time"), now_ms())),
                bid=bid,
                bid_qty=safe_float(row.get("B") or row.get("bidQty")),
                ask=ask,
                ask_qty=safe_float(row.get("A") or row.get("askQty")),
            ),
        )

    async def _handle_depth(self, symbol: str, data) -> None:
        if not self.on_depth:
            return
        row = data if isinstance(data, dict) else (data[0] if data else None)
        if not isinstance(row, dict):
            return
        bids = [(safe_float(p), safe_float(q)) for p, q in row.get("bids", [])]
        asks = [(safe_float(p), safe_float(q)) for p, q in row.get("asks", [])]
        if not bids or not asks:
            return
        bids.sort(key=lambda x: -x[0])
        asks.sort(key=lambda x: x[0])
        await self.on_depth(symbol, DepthSnapshot(ts=now_ms(), bids=bids, asks=asks))


class BingXUserWS(_BaseWS):
    """Private account stream via listenKey (orders, balance, positions)."""

    name = "user-ws"

    def __init__(
        self,
        rest: BingXRest,
        ws_base: str,
        on_order_update: Callable[[dict], Awaitable[None]] | None = None,
        on_account_update: Callable[[dict], Awaitable[None]] | None = None,
    ):
        super().__init__(ws_base)
        self.rest = rest
        self.on_order_update = on_order_update
        self.on_account_update = on_account_update
        self._listen_key = ""
        self._keepalive_task: asyncio.Task | None = None

    async def start(self) -> None:
        await super().start()
        if self._keepalive_task is None or self._keepalive_task.done():
            self._keepalive_task = asyncio.create_task(self._keepalive_loop(), name="listenkey-keepalive")

    async def stop(self) -> None:
        if self._keepalive_task:
            self._keepalive_task.cancel()
            self._keepalive_task = None
        await super().stop()

    async def _connect_url(self) -> str:
        self._listen_key = await self.rest.create_listen_key()
        sep = "&" if "?" in self.url else "?"
        return f"{self.url}{sep}listenKey={self._listen_key}"

    async def _keepalive_loop(self) -> None:
        while True:
            await asyncio.sleep(25 * 60)
            if self._listen_key:
                try:
                    await self.rest.keepalive_listen_key(self._listen_key)
                    log.info("listenKey extended")
                except Exception as e:  # noqa: BLE001
                    log.warning("listenKey keepalive failed: %s", e)

    async def _on_payload(self, payload: str) -> None:
        try:
            msg = json.loads(payload)
        except json.JSONDecodeError:
            return
        event = msg.get("e", "")
        if event == "ORDER_TRADE_UPDATE" and self.on_order_update:
            await self.on_order_update(msg.get("o", {}))
        elif event == "ACCOUNT_UPDATE" and self.on_account_update:
            await self.on_account_update(msg.get("a", {}))
        elif event == "listenKeyExpired":
            log.warning("listenKey expired -> reconnecting")
            if self._ws and not self._ws.closed:
                await self._ws.close()
