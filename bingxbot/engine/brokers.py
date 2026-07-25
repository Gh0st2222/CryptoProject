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
from .backtest import FILL_THROUGH_BPS
from .portfolio import Portfolio

log = logging.getLogger("broker")


class Broker:
    async def open_position(self, symbol: str, side: str, sized: SizedOrder,
                            reason: str, bar_ts: int) -> OrderResult:
        raise NotImplementedError

    async def close_position(self, symbol: str, reason: str, frac: float | None = None,
                             maker_price: float | None = None) -> OrderResult:
        raise NotImplementedError

    async def arm_maker_exit(self, symbol: str, side: str, qty: float, price: float) -> bool:
        return False

    async def cancel_maker_exit(self, symbol: str) -> None:
        return None

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
        self._exit_orders: dict[str, tuple[str, float]] = {}   # symbol -> (side, resting price)

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
                   bar_ts: int, px: float, maker: bool, adverse: bool = True) -> OrderResult:
        if maker and adverse:
            # adverse-selection penalty on an INSTANT passive fill (no resting
            # wait modeled). A genuinely rested limit pays its honesty through
            # the trade-through requirement instead and fills AT its price — a
            # limit can never fill worse than the price it rested at.
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
        """Resting entry (pullback-depth or maker-at-touch): the limit RESTS
        until the live tape trades THROUGH it or the window expires — the same
        trade-through margin the backtest demands (`FILL_THROUGH_BPS`) before
        assuming a queued maker fill. A mere touch does not fill: on the real
        book, price kissing the level fills the queue ahead of us and leaves;
        counting those flattered paper against both the backtest and live."""
        lim = sized.entry_limit
        deadline = asyncio.get_running_loop().time() + max(sized.entry_wait_s, 5.0)
        d = 1 if side == LONG else -1
        need = lim * (1 - d * FILL_THROUGH_BPS / 10_000.0)
        while asyncio.get_running_loop().time() < deadline:
            st = self.states.get(symbol)
            px = st.last_price if st is not None else 0.0
            if px > 0 and (px - need) * d <= 0:     # tape traded through the limit
                return self._fill_open(symbol, side, sized, reason, bar_ts, lim,
                                       maker=True, adverse=False)
            await asyncio.sleep(0.5)
        log.info("[paper] pullback limit %s %s unfilled @ %.6g — entry abandoned",
                 side, symbol, lim)
        return OrderResult(ok=False, error="pullback limit unfilled")

    async def arm_maker_exit(self, symbol: str, side: str, qty: float, price: float) -> bool:
        """Paper must simulate the resting exit, not skip it. The base class
        returns False, and inheriting that silently would make paper the ONLY
        engine still taking its targets: the simulator and the compiled kernel
        both price a maker exit when the setting is on, so a paper account that
        market-closes on touch would pay taker where the champion was measured
        paying maker — the champion is then judged on an execution model the
        account never uses. Same failure the whole parity doctrine exists to
        prevent, just on the cheaper side of the fee.

        Post-only is modelled honestly: an order that would already cross the
        current mark is REJECTED, exactly as the exchange rejects it, and the
        caller falls back to market-close-on-touch."""
        if price <= 0 or qty <= 0:
            return False
        spec = self.specs.get(symbol, ContractSpec(symbol))
        if round_step(qty, spec.qty_precision) < spec.min_qty:
            return False
        px = round_step(price, spec.price_precision)
        # the touch this order would cross INTO: closing a LONG is a SELL, which
        # crosses against the bid; closing a SHORT is a BUY, against the ask.
        # (`maker=True` asks _fill_price for the raw touch with no slippage.)
        touch = self._fill_price(symbol, is_buy=(side == LONG), maker=True)
        d = 1 if side == LONG else -1
        if touch > 0 and (touch - px) * d >= 0:
            # the target is already at/through the mark: a post-only order there
            # would be an immediate taker, so the exchange refuses it.
            log.debug("[paper] maker exit %s @ %.6g would cross touch %.6g — rejected",
                      symbol, px, touch)
            return False
        self._exit_orders[symbol] = (side, px)
        log.info("[paper] maker exit armed %s %s qty=%.6g @ %.6g", side, symbol, qty, px)
        return True

    async def cancel_maker_exit(self, symbol: str) -> None:
        self._exit_orders.pop(symbol, None)

    def maker_exit_price(self, symbol: str) -> float:
        rec = self._exit_orders.get(symbol)
        return rec[1] if rec else 0.0

    async def close_position(self, symbol: str, reason: str, frac: float | None = None,
                             maker_price: float | None = None) -> OrderResult:
        """`maker_price` closes the position as a RESTING order that the tape
        traded through: it fills at exactly that price and pays the maker fee,
        because we were the passive side. Everything else crosses the spread
        and pays taker, like a market order."""
        pos = self.portfolio.positions.get(symbol)
        if pos is None:
            return OrderResult(ok=False, error="no position")
        maker = maker_price is not None and maker_price > 0
        px = float(maker_price) if maker else self._fill_price(symbol, is_buy=(pos.side == SHORT))
        if px <= 0:
            return OrderResult(ok=False, error="no market price")
        spec = self.specs.get(symbol, ContractSpec(symbol))
        if maker:
            fee = pos.qty * px * self.maker_fee
            planned_risk = abs(pos.entry_price - pos.stop_price) * pos.qty if pos.stop_price > 0 else 0.0
            tr = self.portfolio.close_position(symbol, px, now_ms(), fee, reason, planned_risk)
            self._exit_orders.pop(symbol, None)   # the order that just filled
            if tr:
                log.info("[paper] MAKER EXIT %s %s @ %.6g pnl=%.4f (%s)",
                         pos.side, symbol, px, tr.pnl, reason)
            return OrderResult(ok=True, filled_price=px, filled_qty=pos.qty, fee=fee)
        if frac is not None and 0.0 < frac < 1.0:
            qty_out = pos.qty * frac
            # dust guard: if either leg would violate exchange minimums, the
            # partial degenerates to a full close rather than stranding dust.
            if qty_out >= spec.min_qty and (pos.qty - qty_out) >= spec.min_qty:
                fee = qty_out * px * self.taker_fee
                tr = self.portfolio.scale_out(symbol, frac, px, now_ms(), fee, reason)
                if tr:
                    log.info("[paper] SCALE-OUT %s %s %.0f%% @ %.6g pnl=%.4f (%s)",
                             pos.side, symbol, frac * 100, px, tr.pnl, reason)
                    return OrderResult(ok=True, filled_price=px, filled_qty=qty_out, fee=fee)
        fee = pos.qty * px * self.taker_fee
        planned_risk = abs(pos.entry_price - pos.stop_price) * pos.qty if pos.stop_price > 0 else 0.0
        tr = self.portfolio.close_position(symbol, px, now_ms(), fee, reason, planned_risk)
        self._exit_orders.pop(symbol, None)   # position gone: the target is moot
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
        self._exit_orders: dict[str, tuple[str, float]] = {}   # symbol -> (order id, price)

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
                    fee = abs(safe_float(o.get("commission") or o.get("fee")))
                    # executedQty is deliberately NOT used to size the position.
                    # This poll gives up after ~1.4s, so a fill still in progress
                    # would be read as SMALLER than it ends up — and booking that
                    # would make reconcile see the exchange as larger and "adopt"
                    # size upward on the next poll. Booking the requested qty and
                    # letting reconcile correct any real shortfall is the stable
                    # direction to be wrong in.
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
        # The protective STOP is ALWAYS a market order and always attached to
        # the entry — a stop that fails to fill is the one loss you cannot
        # iterate on. Only the profit target is ever allowed to rest passively.
        if self.cfg.strategy.maker_exits:
            return sl, None
        tp = ({"type": "TAKE_PROFIT_MARKET", "stopPrice": sized.take_profit, "workingType": wt}
              if sized.take_profit > 0 else None)
        return sl, tp

    async def arm_maker_exit(self, symbol: str, side: str, qty: float, price: float) -> bool:
        """Rest the profit target on the book as a post-only, REDUCE-ONLY limit
        so the exit earns the maker fee instead of paying taker + slippage.

        Two exchange-enforced properties make this safe to leave sitting there:
        `reduceOnly` means the order can only ever SHRINK an existing position —
        it can never open a reverse one if the position is already gone — and
        `PostOnly` means it is rejected rather than filled if it would cross,
        so it can never turn into the taker order we are trying to avoid.

        Returns False on any failure, and the caller then keeps the engine's
        existing market-close-on-touch behaviour: a position is never left
        without a way out."""
        spec = self.specs.get(symbol, ContractSpec(symbol))
        px = round_step(price, spec.price_precision)
        q = round_step(qty, spec.qty_precision)
        if px <= 0 or q < spec.min_qty:
            return False
        await self.cancel_maker_exit(symbol)     # never stack two resting exits
        try:
            resp = await self.rest.place_order(
                symbol=symbol,
                side=SELL if side == LONG else BUY,   # the closing side
                position_side=side,                   # ...of THIS position (hedge mode)
                order_type="LIMIT", quantity=q, price=px,
                time_in_force="PostOnly", reduce_only=True,
                client_order_id=f"bxx{uuid.uuid4().hex[:11]}",
            )
        except (BingXAPIError, BingXError) as e:
            log.warning("maker exit %s @ %.6g rejected (%s) — falling back to market close",
                        symbol, px, e)
            return False
        oid = str(resp.get("orderId", "")) if isinstance(resp, dict) else ""
        if not oid:
            return False
        self._exit_orders[symbol] = (oid, px)
        log.info("[LIVE] maker exit armed %s %s qty=%.6g @ %.6g id=%s", side, symbol, q, px, oid)
        return True

    async def cancel_maker_exit(self, symbol: str) -> None:
        """Take the resting exit off the book. Must run whenever the position
        changes size or goes away, or the order outlives what it was sized for."""
        rec = self._exit_orders.pop(symbol, None)
        if not rec:
            return
        try:
            await self.rest.cancel_order(symbol, rec[0])
        except BingXError as e:
            log.debug("cancel maker exit %s: %s", symbol, e)

    def maker_exit_price(self, symbol: str) -> float:
        """The price of the resting exit, if one is armed — so a position that
        vanished from the exchange between polls is booked at the price it
        actually filled at, not at a guess."""
        rec = self._exit_orders.get(symbol)
        return rec[1] if rec else 0.0

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

    async def close_position(self, symbol: str, reason: str, frac: float | None = None,
                             maker_price: float | None = None) -> OrderResult:
        # `maker_price` is a paper-simulation concept: live maker exits are a
        # real resting order on the book, so a live close here is always the
        # deliberate market exit (stop, edge flip, time stop, flatten).
        pos = self.portfolio.positions.get(symbol)
        if pos is None:
            return OrderResult(ok=False, error="no position")
        await self.cancel_maker_exit(symbol)   # the target is moot now
        spec = self.specs.get(symbol, ContractSpec(symbol))
        partial = frac is not None and 0.0 < frac < 1.0
        close_qty = round_step(pos.qty * frac, spec.qty_precision) if partial else \
            round_step(pos.qty, spec.qty_precision)
        # dust guard: degrade a partial to a full close if either leg would
        # violate exchange minimums.
        if partial and (close_qty < spec.min_qty or (pos.qty - close_qty) < spec.min_qty):
            partial, close_qty = False, round_step(pos.qty, spec.qty_precision)
        try:
            resp = await self.rest.place_order(
                symbol=symbol,
                side=SELL if pos.side == LONG else BUY,
                position_side=pos.side,
                order_type="MARKET",
                quantity=close_qty,
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
            fee = close_qty * fill_px * spec.taker_fee
        if partial:
            tr = self.portfolio.scale_out(symbol, close_qty / pos.qty, fill_px, now_ms(), fee, reason)
            if tr:
                log.info("[LIVE] SCALE-OUT %s %s qty=%.6g @ %.6g pnl=%.4f (%s)",
                         pos.side, symbol, close_qty, fill_px, tr.pnl, reason)
            return OrderResult(ok=True, order_id=order_id, filled_price=fill_px,
                               filled_qty=close_qty, fee=fee)
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
        maker = False
        if px <= 0:
            # A resting maker exit is the MOST likely reason a position
            # disappeared between polls, and it fills at a known price — book
            # it there (and at the maker fee it actually paid) instead of
            # guessing the stop level and recording a loss that never happened.
            mx = self.maker_exit_price(symbol)
            if mx > 0:
                px, maker, reason = mx, True, "target (maker exit)"
            else:
                px = pos.stop_price if pos.stop_price > 0 else pos.entry_price
        spec = self.specs.get(symbol, ContractSpec(symbol))
        fee = pos.qty * px * (spec.maker_fee if maker else spec.taker_fee)
        planned_risk = abs(pos.entry_price - pos.stop_price) * pos.qty if pos.stop_price > 0 else 0.0
        self.portfolio.close_position(symbol, px, now_ms(), fee, reason, planned_risk)
        self._exit_orders.pop(symbol, None)   # it filled (or died with the position)

    async def _protective_levels(self, symbol: str, side: str) -> tuple[float, float]:
        """Read this position's stop and target back off the book.

        The levels are OUR numbers — we placed them at entry — but after a
        restart the only copy left is the exchange's. A reduce-only resting
        limit on the closing side is the maker exit; a stop-market is the
        protective stop. Anything we cannot identify is ignored rather than
        guessed at: inventing a stop level would invent an `init_risk`, and
        every R statistic downstream would then be quietly fictional."""
        try:
            orders = await self.rest.open_orders(symbol)
        except Exception as e:  # noqa: BLE001 — see below
            # Deliberately broad. This is best-effort ENRICHMENT of an adoption
            # that must happen either way: failing to recover the levels costs
            # us the trail, but letting the exception escape aborts reconcile
            # and loses track of the POSITION itself, which is far worse.
            log.warning("adopt %s: could not read open orders (%s)", symbol, e)
            return 0.0, 0.0
        stop = tp = 0.0
        closing = SELL if side == LONG else BUY
        for o in orders or []:
            if str(o.get("side", "")).upper() != closing:
                continue
            ps = str(o.get("positionSide", "")).upper()
            if ps and ps not in ("BOTH", side):
                continue
            otype = str(o.get("type", "")).upper()
            trigger = safe_float(o.get("stopPrice") or o.get("triggerPrice"))
            if "STOP" in otype and "PROFIT" not in otype and trigger > 0:
                stop = trigger
            elif "TAKE_PROFIT" in otype and trigger > 0:
                tp = trigger
            elif otype == "LIMIT" and safe_float(o.get("price")) > 0:
                # The resting post-only target from maker_exits — re-adopt it so
                # the engine knows a target is already working on the book.
                # REDUCE-ONLY is what identifies it as ours: a plain limit on
                # this side could be the user's own manual order, and adopting
                # that would mean cancel_maker_exit() later pulls an order we
                # were never entitled to touch.
                if str(o.get("reduceOnly", "")).lower() not in ("true", "1"):
                    continue
                px = safe_float(o.get("price"))
                tp = px
                oid = str(o.get("orderId", ""))
                if oid:
                    self._exit_orders[symbol] = (oid, px)
        return stop, tp

    def _sync_qty(self, symbol: str, row: dict) -> None:
        """Reconcile the SIZE of a position that exists on both sides.

        Presence was already checked; size never was, so any drift between our
        book and the exchange's was permanent. That matters more than it sounds:
        a TradeRecord books `qty * (exit - entry)`, so carrying a stale qty does
        not just misstate exposure — it manufactures P&L that never happened,
        and that number feeds the journal, the live-evidence demotion rule, the
        champion's probation stats and the report. The system LEARNS from it.

        A partial fill is the ordinary way this happens, not an exotic one: a
        resting reduce-only exit is a limit order, and price trading through it
        briefly fills part of the size and leaves. A partially filled entry does
        the same. Either way the symbol is still on both sides, so the presence
        check above sees nothing wrong.

        Shrinkage is booked as a real partial close (at the resting exit price
        when one is armed — the most likely cause and a price we actually know),
        so the P&L is recorded instead of silently vanishing. Growth can only
        mean size we did not open, so the exchange wins and we say so loudly."""
        pos = self.portfolio.positions.get(symbol)
        if pos is None or pos.qty <= 0:
            return
        actual = abs(safe_float(row.get("positionAmt") or row.get("availableAmt")))
        if actual <= 0:
            return
        spec = self.specs.get(symbol, ContractSpec(symbol))
        # ignore dust: rounding at the symbol's own step, or sub-1% noise
        tol = max(spec.min_qty, pos.qty * 0.01)
        if abs(actual - pos.qty) <= tol:
            return
        if actual > pos.qty:
            log.warning("reconcile: %s exchange qty %.8g > local %.8g — adopting exchange truth",
                        symbol, actual, pos.qty)
            pos.qty = actual
            return
        frac = 1.0 - actual / pos.qty
        px = self.maker_exit_price(symbol)
        reason = "partial fill (maker exit)" if px > 0 else "partial close (reconcile)"
        if px <= 0:
            px = safe_float(row.get("markPrice")) or pos.entry_price
        fee = pos.qty * frac * px * (spec.maker_fee if "maker" in reason else spec.taker_fee)
        log.warning("reconcile: %s shrank %.8g -> %.8g, booking %.1f%% as %s @ %.6g",
                    symbol, pos.qty, actual, frac * 100, reason, px)
        was_scaled = pos.scaled_out
        if self.portfolio.scale_out(symbol, frac, px, now_ms(), fee, reason) is None:
            pos.qty = actual        # too small to book as a trade; still sync the size
            return
        # scale_out() marks the position as having taken its scale-out. This was
        # an execution artifact, not the strategy's decision, so leave that flag
        # exactly as it was — otherwise a partial fill silently cancels a
        # scale-out the strategy still intends to take.
        pos.scaled_out = was_scaled

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
            else:
                self._sync_qty(symbol, on_exchange[symbol])
        for symbol, r in on_exchange.items():
            if symbol in self.portfolio.positions or symbol not in symbols:
                continue
            side = LONG if str(r.get("positionSide", "")).upper() == LONG or safe_float(r.get("positionAmt")) > 0 else SHORT
            qty = abs(safe_float(r.get("positionAmt") or r.get("availableAmt")))
            entry = safe_float(r.get("avgPrice"))
            if qty <= 0 or entry <= 0:
                continue
            # Recover the protective levels from the exchange's own resting
            # orders. Live deliberately never restores local state ("the
            # exchange is the truth"), so every restart with an open position
            # came through here — and adopting with stop_price=0 quietly
            # DISABLED the whole adaptive exit stack for that position:
            # `exits.manage` falls back to `risk = |entry - stop|`, which with
            # stop=0 is the entry price itself, so rr is ~0.0001 forever. The
            # breakeven move, the chandelier trail, the give-back lock and the
            # scale-out are all rr-gated and none of them could ever fire again.
            # The position was left to the exchange stop and the time stop, and
            # its eventual trade booked r_multiple=0 into the live stats the
            # champion demotion rule reads.
            stop, tp = await self._protective_levels(symbol, side)
            log.warning("reconcile: adopting unknown %s %s position qty=%.6g sl=%.6g tp=%.6g",
                        side, symbol, qty, stop, tp)
            if stop <= 0:
                log.error("reconcile: %s has NO protective stop on the exchange — the "
                          "adaptive trail cannot arm; exits fall back to the time stop", symbol)
            self.portfolio.open_position(Position(
                symbol=symbol, side=side, qty=qty, entry_price=entry, opened_ts=now_ms(),
                leverage=safe_float(r.get("leverage"), 1.0), entry_reason="adopted",
                entry_bar_ts=0, stop_price=stop, take_profit=tp,
                init_risk=abs(entry - stop) if stop > 0 else 0.0,
            ), entry_fee=0.0)

    async def flatten_all(self, reason: str) -> None:
        for symbol in list(self.portfolio.positions):
            await self.close_position(symbol, reason)
        try:
            await self.rest.close_all_positions()
        except BingXError:
            pass
