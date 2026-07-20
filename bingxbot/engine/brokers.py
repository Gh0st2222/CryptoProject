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
                 maker_fee: float = 0.0002, entry_mode: str = "maker",
                 maker_adverse_bps: float = 0.4):
        self.portfolio = portfolio
        self.states = feed_states
        self.specs = specs
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        self.entry_mode = entry_mode
        self.slip = slippage_bps / 10_000.0
        self.maker_adverse = maker_adverse_bps / 10_000.0

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

    def _fill_open(self, symbol: str, side: str, sized: SizedOrder, reason: str,
                   bar_ts: int, px: float, maker: bool) -> OrderResult:
        if maker:
            # honest adverse-selection penalty on the passive fill — a resting
            # limit gets filled when price moves through it, same as the backtest.
            d = 1 if side == LONG else -1
            px *= 1 + d * self.maker_adverse
        fee = sized.qty * px * (self.maker_fee if maker else self.taker_fee)
        pos = Position(
            symbol=symbol, side=side, qty=sized.qty, entry_price=px,
            opened_ts=now_ms(), leverage=sized.leverage,
            stop_price=sized.stop_price, take_profit=sized.take_profit,
            entry_fee=fee, entry_reason=reason, entry_bar_ts=bar_ts,
        )
        if not self.portfolio.open_position(pos, fee):
            return OrderResult(ok=False, error=f"position already open on {symbol}")
        log.info("[paper] OPEN %s %s qty=%.6g @ %.6g sl=%.6g tp=%.6g (%s)",
                 side, symbol, sized.qty, px, sized.stop_price, sized.take_profit, reason)
        return OrderResult(ok=True, order_id=f"paper-{uuid.uuid4().hex[:10]}",
                           filled_price=px, filled_qty=sized.qty, fee=fee)

    async def open_position(self, symbol: str, side: str, sized: SizedOrder,
                            reason: str, bar_ts: int) -> OrderResult:
        if symbol in self.portfolio.positions:
            return OrderResult(ok=False, error=f"position already open on {symbol}")
        if sized.entry_limit > 0:
            return await self._open_resting(symbol, side, sized, reason, bar_ts)
        maker = self.entry_mode == "maker"
        px = self._fill_price(symbol, is_buy=(side == LONG), maker=maker)
        if px <= 0:
            return OrderResult(ok=False, error="no market price")
        return self._fill_open(symbol, side, sized, reason, bar_ts, px, maker)

    async def _open_resting(self, symbol: str, side: str, sized: SizedOrder,
                            reason: str, bar_ts: int) -> OrderResult:
        """Pullback entry: the limit RESTS until the live tape trades through it
        or the window expires — mirroring exactly what the backtest models and
        what the live broker does on the exchange. No touch, no trade."""
        lim = sized.entry_limit
        deadline = asyncio.get_running_loop().time() + max(sized.entry_wait_s, 5.0)
        d = 1 if side == LONG else -1
        while asyncio.get_running_loop().time() < deadline:
            st = self.states.get(symbol)
            px = st.last_price if st is not None else 0.0
            if px > 0 and (px - lim) * d <= 0:      # tape touched/through the limit
                return self._fill_open(symbol, side, sized, reason, bar_ts, lim, maker=True)
            await asyncio.sleep(0.5)
        log.info("[paper] pullback limit %s %s unfilled @ %.6g — entry abandoned",
                 side, symbol, lim)
        return OrderResult(ok=False, error="pullback limit unfilled")

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
        self._lev_set: dict[tuple[str, str], int] = {}

    async def prepare_symbol(self, symbol: str) -> None:
        """Set isolated/cross margin mode once per symbol. Never fatal."""
        if symbol in self._prepared:
            return
        try:
            await self.rest.set_margin_type(symbol, self.cfg.risk.margin_mode)
        except BingXAPIError as e:
            log.info("prepare %s: %s (usually already set)", symbol, e.msg)
        except BingXError as e:
            log.warning("prepare %s failed: %s", symbol, e)
        self._prepared.add(symbol)

    async def _ensure_leverage(self, symbol: str, side: str, lev: int) -> None:
        """Set the per-trade leverage the sizer chose, only when it changed."""
        lev = max(1, int(lev))
        if self._lev_set.get((symbol, side)) == lev:
            return
        try:
            await self.rest.set_leverage(symbol, side, lev)
            self._lev_set[(symbol, side)] = lev
        except BingXError as e:
            log.info("set leverage %s %s %dx: %s", symbol, side, lev, e)

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
        if symbol in self.portfolio.positions:
            return OrderResult(ok=False, error=f"position already open on {symbol}")
        await self.prepare_symbol(symbol)
        await self._ensure_leverage(symbol, side, sized.leverage)
        spec = self.specs.get(symbol, ContractSpec(symbol))
        if sized.entry_limit > 0:
            # resting entry: unfilled window = abandoned. A post-only PLACEMENT
            # rejection (price moved into the limit) may fall through to taker
            # only for touch-style limits — a deep pullback limit never chases.
            r = await self._open_maker(symbol, side, sized, reason, bar_ts, spec)
            if r is not None:
                return r
            if sized.allow_taker_fallback:
                return await self._open_taker(symbol, side, sized, reason, bar_ts, spec)
            return OrderResult(ok=False, error="pullback limit rejected")
        if self.cfg.strategy.entry_mode == "maker":
            # rest a post-only limit to pay the maker fee (~0.02%) instead of taker
            # (~0.05%) — on a fast strategy that halved round-trip cost is often the
            # difference between a live edge and a loss. If it doesn't fill in the
            # window we abort and let the next scan re-decide (no taker chasing).
            r = await self._open_maker(symbol, side, sized, reason, bar_ts, spec)
            if r is not None:
                return r
        return await self._open_taker(symbol, side, sized, reason, bar_ts, spec)

    def _sl_tp(self, sized: SizedOrder) -> tuple[dict, dict | None]:
        wt = "MARK_PRICE"
        sl = {"type": "STOP_MARKET", "stopPrice": sized.stop_price, "workingType": wt}
        tp = ({"type": "TAKE_PROFIT_MARKET", "stopPrice": sized.take_profit, "workingType": wt}
              if sized.take_profit > 0 else None)
        return sl, tp

    async def _open_taker(self, symbol: str, side: str, sized: SizedOrder,
                          reason: str, bar_ts: int, spec: ContractSpec) -> OrderResult:
        sl, tp = self._sl_tp(sized)
        try:
            resp = await self.rest.place_order(
                symbol=symbol, side=BUY if side == LONG else SELL, position_side=side,
                order_type="MARKET", quantity=sized.qty,
                client_order_id=f"bxb{uuid.uuid4().hex[:12]}", stop_loss=sl, take_profit=tp,
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
        log.info("[LIVE] TAKER OPEN %s %s qty=%.6g @ %.6g sl=%.6g tp=%.6g id=%s (%s)",
                 side, symbol, sized.qty, fill_px, sized.stop_price, sized.take_profit, order_id, reason)
        return OrderResult(ok=True, order_id=order_id, filled_price=fill_px,
                           filled_qty=sized.qty, fee=fee, raw=resp if isinstance(resp, dict) else {})

    async def _open_maker(self, symbol: str, side: str, sized: SizedOrder,
                          reason: str, bar_ts: int, spec: ContractSpec) -> OrderResult | None:
        """Place a post-only limit inside the touch and wait for a fill. Returns
        an OrderResult on fill/hard-error, or None to signal 'unfilled — caller
        may fall through' (we abort rather than chase)."""
        ref = sized.notional / max(sized.qty, 1e-12)
        off = self.cfg.strategy.maker_offset_bps / 10_000.0
        d = 1 if side == LONG else -1
        # pullback entries carry their own (deeper) limit; else rest just inside the touch
        raw_limit = sized.entry_limit if sized.entry_limit > 0 else ref * (1 - d * off)
        limit = round_step(raw_limit, spec.price_precision)
        qty = round_step(sized.qty, spec.qty_precision)
        if limit <= 0 or qty <= 0:
            return None
        sl, tp = self._sl_tp(sized)
        try:
            resp = await self.rest.place_order(
                symbol=symbol, side=BUY if side == LONG else SELL, position_side=side,
                order_type="LIMIT", quantity=qty, price=limit, time_in_force="PostOnly",
                client_order_id=f"bxm{uuid.uuid4().hex[:11]}", stop_loss=sl, take_profit=tp,
            )
        except (BingXAPIError, BingXError) as e:
            log.warning("live MAKER place %s %s failed (%s) — will try taker", side, symbol, e)
            return None
        order_id = str(resp.get("orderId", ""))
        fill_px, filled_qty, fee = await self._await_limit_fill(symbol, order_id,
                                                                window_s=sized.entry_wait_s)
        if filled_qty <= 0:
            try:
                await self.rest.cancel_order(symbol, order_id)
            except BingXError:
                pass
            log.info("[LIVE] maker %s %s unfilled @ %.6g — aborting entry", side, symbol, limit)
            return OrderResult(ok=False, error="maker unfilled")
        if fee <= 0:
            fee = filled_qty * fill_px * spec.maker_fee
        pos = Position(
            symbol=symbol, side=side, qty=filled_qty, entry_price=fill_px,
            opened_ts=now_ms(), leverage=sized.leverage,
            stop_price=sized.stop_price, take_profit=sized.take_profit,
            entry_fee=fee, entry_reason=reason, entry_bar_ts=bar_ts,
        )
        self.portfolio.open_position(pos, fee)
        log.info("[LIVE] MAKER OPEN %s %s qty=%.6g @ %.6g (limit %.6g) id=%s (%s)",
                 side, symbol, filled_qty, fill_px, limit, order_id, reason)
        return OrderResult(ok=True, order_id=order_id, filled_price=fill_px,
                           filled_qty=filled_qty, fee=fee, raw=resp if isinstance(resp, dict) else {})

    async def _await_limit_fill(self, symbol: str, order_id: str,
                                window_s: float = 0.0) -> tuple[float, float, float]:
        """Poll a resting maker order for a fill within the wait window. The
        window matches what the backtest models: the limit rests for
        `maker_wait_bars` SIGNAL bars (e.g. 2 x 15m), not a fixed few seconds —
        the old 12s window abandoned nearly every maker entry the simulation
        assumed would fill. The order's own entry_wait_s wins when provided so
        the engine and broker can never disagree about the window. Poll cadence
        stretches with the window so the number of REST calls stays bounded.
        Returns (avg_price, filled_qty, fee); filled_qty 0 => never filled."""
        from ..util import interval_ms
        if window_s <= 0:
            window_s = max(1, self.cfg.strategy.maker_wait_bars) * interval_ms(self.cfg.strategy.interval) / 1000.0
        poll_gap = min(max(1.5, window_s / 40.0), 20.0)
        polls = max(2, int(window_s / poll_gap))
        for _ in range(polls):
            await asyncio.sleep(poll_gap)
            try:
                o = await self.rest.get_order(symbol, order_id)
            except BingXError as e:
                log.warning("maker fill poll %s: %s", order_id, e)
                continue
            status = str(o.get("status", "")).upper()
            exec_qty = safe_float(o.get("executedQty") or o.get("cumQty"))
            if status == "FILLED":
                ap = safe_float(o.get("avgPrice") or o.get("averagePrice"))
                return ap, exec_qty, abs(safe_float(o.get("commission") or o.get("fee")))
            if status in ("CANCELED", "EXPIRED", "REJECTED"):
                return 0.0, 0.0, 0.0
        # window elapsed — take any partial fill, cancel the remainder
        try:
            o = await self.rest.get_order(symbol, order_id)
            exec_qty = safe_float(o.get("executedQty") or o.get("cumQty"))
            if exec_qty > 0:
                await self.rest.cancel_order(symbol, order_id)
                return safe_float(o.get("avgPrice")), exec_qty, abs(safe_float(o.get("commission")))
        except BingXError:
            pass
        return 0.0, 0.0, 0.0

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
