"""Brokers: identical interface for simulated and real execution.

PaperBroker fills against the live order book (best bid/ask + slippage) with
fake money — the "realtime simulation on the real market" mode. LiveBroker
sends real BingX orders and attaches exchange-side stop-loss / take-profit to
every entry so protective exits survive bot or connectivity failure.
"""
from __future__ import annotations

import asyncio
import logging
import uuid

from ..config import BotConfig
from ..exchange.errors import BingXAPIError, BingXError
from ..exchange.models import BUY, LONG, SELL, SHORT, ContractSpec, OrderResult, Position
from ..exchange.rest import BingXRest
from ..risk.manager import SizedOrder
from ..util import now_ms, round_step, safe_float
from .portfolio import Portfolio

log = logging.getLogger("broker")


class Broker:
    async def open_position(self, symbol: str, side: str, sized: SizedOrder,
                            reason: str, bar_ts: int) -> OrderResult:
        raise NotImplementedError

    async def close_position(self, symbol: str, reason: str) -> OrderResult:
        raise NotImplementedError

    async def flatten_all(self, reason: str) -> None:
        raise NotImplementedError


class PaperBroker(Broker):
    def __init__(self, portfolio: Portfolio, feed_states: dict, specs: dict[str, ContractSpec],
                 taker_fee: float, slippage_bps: float,
                 maker_fee: float = 0.0002, entry_mode: str = "maker"):
        self.portfolio = portfolio
        self.states = feed_states
        self.specs = specs
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        self.entry_mode = entry_mode
        self.slip = slippage_bps / 10_000.0

    def _fill_price(self, symbol: str, is_buy: bool, maker: bool = False) -> float:
        st = self.states.get(symbol)
        if st is None:
            return 0.0
        if st.book is not None:
            # taker crosses the spread; a resting maker fills at the near touch
            if maker:
                px = st.book.bid if is_buy else st.book.ask
            else:
                px = st.book.ask if is_buy else st.book.bid
        else:
            px = st.last_price or st.candles.last_close
        if maker:
            return px
        return px * (1 + self.slip) if is_buy else px * (1 - self.slip)

    async def open_position(self, symbol: str, side: str, sized: SizedOrder,
                            reason: str, bar_ts: int) -> OrderResult:
        maker = self.entry_mode == "maker"
        px = self._fill_price(symbol, is_buy=(side == LONG), maker=maker)
        if px <= 0:
            return OrderResult(ok=False, error="no market price")
        fee = sized.qty * px * (self.maker_fee if maker else self.taker_fee)
        pos = Position(
            symbol=symbol, side=side, qty=sized.qty, entry_price=px,
            opened_ts=now_ms(), leverage=sized.leverage,
            stop_price=sized.stop_price, take_profit=sized.take_profit,
            entry_fee=fee, entry_reason=reason, entry_bar_ts=bar_ts,
        )
        self.portfolio.open_position(pos, fee)
        log.info("[paper] OPEN %s %s qty=%.6g @ %.6g sl=%.6g tp=%.6g (%s)",
                 side, symbol, sized.qty, px, sized.stop_price, sized.take_profit, reason)
        return OrderResult(ok=True, order_id=f"paper-{uuid.uuid4().hex[:10]}",
                           filled_price=px, filled_qty=sized.qty, fee=fee)

    async def close_position(self, symbol: str, reason: str) -> OrderResult:
        pos = self.portfolio.positions.get(symbol)
        if pos is None:
            return OrderResult(ok=False, error="no position")
        px = self._fill_price(symbol, is_buy=(pos.side == SHORT))
        if px <= 0:
            return OrderResult(ok=False, error="no market price")
        fee = pos.qty * px * self.taker_fee
        planned_risk = abs(pos.entry_price - pos.stop_price) * pos.qty if pos.stop_price > 0 else 0.0
        tr = self.portfolio.close_position(symbol, px, now_ms(), fee, reason, planned_risk)
        if tr:
            log.info("[paper] CLOSE %s %s @ %.6g pnl=%.4f (%s)", pos.side, symbol, px, tr.pnl, reason)
        return OrderResult(ok=True, filled_price=px, filled_qty=pos.qty, fee=fee)

    async def flatten_all(self, reason: str) -> None:
        for symbol in list(self.portfolio.positions):
            await self.close_position(symbol, reason)


class LiveBroker(Broker):
    def __init__(self, rest: BingXRest, portfolio: Portfolio, specs: dict[str, ContractSpec],
                 cfg: BotConfig):
        self.rest = rest
        self.portfolio = portfolio
        self.specs = specs
        self.cfg = cfg
        self._prepared: set[str] = set()

    async def prepare_symbol(self, symbol: str) -> None:
        """Margin mode + leverage, once per symbol. Never fatal."""
        if symbol in self._prepared:
            return
        lev = self.cfg.risk.max_leverage
        for call in (
            self.rest.set_margin_type(symbol, self.cfg.risk.margin_mode),
            self.rest.set_leverage(symbol, LONG, lev),
            self.rest.set_leverage(symbol, SHORT, lev),
        ):
            try:
                await call
            except BingXAPIError as e:
                log.info("prepare %s: %s (usually already set)", symbol, e.msg)
            except BingXError as e:
                log.warning("prepare %s failed: %s", symbol, e)
        self._prepared.add(symbol)

    async def _await_fill(self, symbol: str, order_id: str, fallback: float) -> tuple[float, float]:
        """Poll a market order briefly for its average fill price."""
        for _ in range(4):
            try:
                o = await self.rest.get_order(symbol, order_id)
                status = str(o.get("status", "")).upper()
                if status == "FILLED":
                    ap = safe_float(o.get("avgPrice") or o.get("averagePrice"))
                    qty = safe_float(o.get("executedQty") or o.get("origQty"))
                    fee = abs(safe_float(o.get("commission") or o.get("fee")))
                    return (ap if ap > 0 else fallback), fee if fee > 0 else 0.0
                if status in ("CANCELED", "EXPIRED", "REJECTED"):
                    break
            except BingXError as e:
                log.warning("fill poll %s: %s", order_id, e)
            await asyncio.sleep(0.35)
        return fallback, 0.0

    async def open_position(self, symbol: str, side: str, sized: SizedOrder,
                            reason: str, bar_ts: int) -> OrderResult:
        if not self.cfg.allow_live:
            return OrderResult(ok=False, error="allow_live is false")
        await self.prepare_symbol(symbol)
        spec = self.specs.get(symbol, ContractSpec(symbol))
        wt = "MARK_PRICE"
        try:
            resp = await self.rest.place_order(
                symbol=symbol,
                side=BUY if side == LONG else SELL,
                position_side=side,
                order_type="MARKET",
                quantity=sized.qty,
                client_order_id=f"bxb{uuid.uuid4().hex[:12]}",
                stop_loss={"type": "STOP_MARKET", "stopPrice": sized.stop_price, "workingType": wt},
                take_profit={"type": "TAKE_PROFIT_MARKET", "stopPrice": sized.take_profit, "workingType": wt},
            )
        except (BingXAPIError, BingXError) as e:
            log.error("live OPEN %s %s failed: %s", side, symbol, e)
            return OrderResult(ok=False, error=str(e))
        order_id = str(resp.get("orderId", ""))
        fill_px, fee = await self._await_fill(symbol, order_id, fallback=sized.notional / max(sized.qty, 1e-12))
        if fee <= 0:
            fee = sized.qty * fill_px * spec.taker_fee
        pos = Position(
            symbol=symbol, side=side, qty=sized.qty, entry_price=fill_px,
            opened_ts=now_ms(), leverage=sized.leverage,
            stop_price=sized.stop_price, take_profit=sized.take_profit,
            entry_fee=fee, entry_reason=reason, entry_bar_ts=bar_ts,
        )
        self.portfolio.open_position(pos, fee)
        log.info("[LIVE] OPEN %s %s qty=%.6g @ %.6g sl=%.6g tp=%.6g id=%s (%s)",
                 side, symbol, sized.qty, fill_px, sized.stop_price, sized.take_profit, order_id, reason)
        return OrderResult(ok=True, order_id=order_id, filled_price=fill_px,
                           filled_qty=sized.qty, fee=fee, raw=resp if isinstance(resp, dict) else {})

    async def close_position(self, symbol: str, reason: str) -> OrderResult:
        pos = self.portfolio.positions.get(symbol)
        if pos is None:
            return OrderResult(ok=False, error="no position")
        spec = self.specs.get(symbol, ContractSpec(symbol))
        try:
            resp = await self.rest.place_order(
                symbol=symbol,
                side=SELL if pos.side == LONG else BUY,
                position_side=pos.side,
                order_type="MARKET",
                quantity=round_step(pos.qty, spec.qty_precision),
            )
        except BingXAPIError as e:
            if "position" in e.msg.lower() or e.code in (80012, 101205, 101400):
                # Position already gone (exchange-side SL/TP fired). Reconcile.
                log.info("close %s: position already flat on exchange (%s)", symbol, e.msg)
                self._record_external_close(symbol, reason="exchange SL/TP")
                return OrderResult(ok=True, error="already flat")
            log.error("live CLOSE %s failed: %s", symbol, e)
            return OrderResult(ok=False, error=str(e))
        except BingXError as e:
            return OrderResult(ok=False, error=str(e))
        order_id = str(resp.get("orderId", ""))
        fill_px, fee = await self._await_fill(symbol, order_id, fallback=pos.entry_price)
        if fee <= 0:
            fee = pos.qty * fill_px * spec.taker_fee
        planned_risk = abs(pos.entry_price - pos.stop_price) * pos.qty if pos.stop_price > 0 else 0.0
        tr = self.portfolio.close_position(symbol, fill_px, now_ms(), fee, reason, planned_risk)
        try:
            await self.rest.cancel_all_orders(symbol)  # clear leftover SL/TP legs
        except BingXError:
            pass
        if tr:
            log.info("[LIVE] CLOSE %s %s @ %.6g pnl=%.4f (%s)", pos.side, symbol, fill_px, tr.pnl, reason)
        return OrderResult(ok=True, order_id=order_id, filled_price=fill_px, filled_qty=pos.qty, fee=fee)

    def _record_external_close(self, symbol: str, reason: str, price: float = 0.0) -> None:
        pos = self.portfolio.positions.get(symbol)
        if pos is None:
            return
        px = price
        if px <= 0:
            # best effort: assume the protective level that was armed
            px = pos.stop_price if pos.stop_price > 0 else pos.entry_price
        spec = self.specs.get(symbol, ContractSpec(symbol))
        fee = pos.qty * px * spec.taker_fee
        planned_risk = abs(pos.entry_price - pos.stop_price) * pos.qty if pos.stop_price > 0 else 0.0
        self.portfolio.close_position(symbol, px, now_ms(), fee, reason, planned_risk)

    async def reconcile(self, symbols: list[str]) -> None:
        """Compare exchange truth with local state; adopt or record differences."""
        try:
            rows = await self.rest.positions()
            bal = await self.rest.balance()
        except BingXError as e:
            log.warning("reconcile failed: %s", e)
            return
        self.portfolio.live_equity = bal["equity"] or bal["balance"] or self.portfolio.live_equity
        on_exchange: dict[str, dict] = {}
        for r in rows:
            amt = safe_float(r.get("positionAmt") or r.get("availableAmt"))
            if abs(amt) > 1e-12:
                on_exchange[r.get("symbol", "")] = r
        for symbol in list(self.portfolio.positions):
            if symbol not in on_exchange:
                log.warning("reconcile: %s closed on exchange (SL/TP or manual)", symbol)
                self._record_external_close(symbol, reason="exchange close (reconcile)")
        for symbol, r in on_exchange.items():
            if symbol in self.portfolio.positions or symbol not in symbols:
                continue
            side = LONG if str(r.get("positionSide", "")).upper() == LONG or safe_float(r.get("positionAmt")) > 0 else SHORT
            qty = abs(safe_float(r.get("positionAmt") or r.get("availableAmt")))
            entry = safe_float(r.get("avgPrice"))
            if qty <= 0 or entry <= 0:
                continue
            log.warning("reconcile: adopting unknown %s %s position qty=%.6g", side, symbol, qty)
            self.portfolio.open_position(Position(
                symbol=symbol, side=side, qty=qty, entry_price=entry, opened_ts=now_ms(),
                leverage=safe_float(r.get("leverage"), 1.0), entry_reason="adopted",
                entry_bar_ts=0,
            ), entry_fee=0.0)

    async def flatten_all(self, reason: str) -> None:
        for symbol in list(self.portfolio.positions):
            await self.close_position(symbol, reason)
        try:
            await self.rest.close_all_positions()
        except BingXError:
            pass
