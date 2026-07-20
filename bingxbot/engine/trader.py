"""TraderEngine: the realtime decision loop shared by paper and live modes.

Bar close  -> full brain evaluation, entry decision (Kelly-sized), exits.
Trade tick -> protective exit checks (stop / take-profit / trailing cross).

The identical TradingBrain + RiskManager pipeline runs in the backtester, so
what you simulate is what trades.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict as dc_asdict

from ..config import BotConfig, MODE_LIVE
from ..data.feed import BaseFeed, MarketState
from ..exchange.models import LONG, SHORT, ContractSpec
from ..risk.manager import RiskManager
from ..strategy.brain import TradingBrain
from ..strategy.exits import AdaptiveExitManager
from ..strategy.features import FeatureFrame, mtf_from_row
from ..util import now_ms
from .backtest import (_entry_signal_ok, gate_funding, gate_mtf_veto,
                       gate_regime)
from .brokers import Broker, LiveBroker
from .portfolio import Portfolio

log = logging.getLogger("trader")

FEATURE_TAIL = 1400  # bars fed to FeatureFrame each close
REACT_MS = 850       # min gap between reactive intra-bar scans, per symbol
ADOPTED_FULL_GRADED = 30  # adopted symbols trade at reduced size until this many graded calls

# Execution-cycle stages surfaced to the UI pipeline.
STAGES = ("SCAN", "DETECT", "VALIDATE", "SIZE", "FILL", "MANAGE", "SETTLE")


class SymbolCtx:
    def __init__(self, symbol: str, brain: TradingBrain):
        self.symbol = symbol
        self.brain = brain
        self.last_row: dict = {}
        self.last_eval: dict = {}
        self.mtf: dict = {}           # per-timeframe view (1m/5m/15m/1h)
        self.bars_held = 0
        self.last_entry_block = ""
        self.stage = "SCAN"
        self.stage_ts = 0
        self.eval_ms = 0.0
        self.react_ts = 0             # last reactive intra-bar scan (ms)
        self.react_dir = 0            # sign of the last reactive signal (confirmation)
        self.react_confirm = 0        # consecutive scans agreeing on direction
        self.entry_ctx: dict = {}     # decision context captured at the open (for the journal)
        self.gates: list[dict] = []   # live entry-gate X-ray [{n, ok, d}] for the UI
        self.busy = False  # guards against overlapping broker calls
        self.pending_task: asyncio.Task | None = None  # resting pullback-limit entry

    def set_stage(self, stage: str) -> None:
        self.stage = stage
        self.stage_ts = now_ms()


class TraderEngine:
    def __init__(self, cfg: BotConfig, feed: BaseFeed, broker: Broker, portfolio: Portfolio,
                 risk: RiskManager, specs: dict[str, ContractSpec], on_update=None, journal=None,
                 record=None):
        self.cfg = cfg
        self.feed = feed
        self.broker = broker
        self.portfolio = portfolio
        self.risk = risk
        self.specs = specs
        self.exits = AdaptiveExitManager(cfg.risk)
        self.on_update = on_update  # async callback(kind: str) -> None
        self.journal = journal      # TradeJournal (records closed trades + context)
        self.record = record        # TrackRecord (daily performance snapshots)
        self.active_champion_id: str | None = None  # vault champion currently driving trades (journal tag)
        self.overlays: dict[str, dict] = {}   # per-symbol brain-scalar overlays (tuner-owned)
        self.ctx: dict[str, SymbolCtx] = {sym: self._make_ctx(sym) for sym in cfg.symbols}
        self.adopted: set[str] = set()   # radar-adopted symbols (beyond the user's list)
        # trades already fed to the risk manager (daily loss / streak / health).
        # Restored trades were accounted in the restored day-state already.
        self._risk_settled = len(portfolio.trades)
        self._task: asyncio.Task | None = None
        self._housekeeper: asyncio.Task | None = None
        self._fast_pusher: asyncio.Task | None = None
        self.running = False
        self.started_ts = 0
        self.tape: list[dict] = []      # recent fills/exits for the UI ticker

    def _interval_ms(self) -> int:
        from ..util import interval_ms
        return interval_ms(self.cfg.strategy.interval)

    # ------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        await self.feed.start()
        if isinstance(self.broker, LiveBroker):
            await self.broker.reconcile(self.cfg.symbols)
        self.running = True
        self.started_ts = now_ms()
        self._task = asyncio.create_task(self._loop(), name="trader-loop")
        self._housekeeper = asyncio.create_task(self._housekeeping(), name="trader-housekeeping")
        self._fast_pusher = asyncio.create_task(self._fast_push_loop(), name="trader-fastpush")
        log.info("trader started: %s %s %s", self.portfolio.mode, self.cfg.symbols, self.cfg.strategy.interval)

    async def stop(self, flatten: bool = False) -> None:
        self.running = False
        pending = [c.pending_task for c in self.ctx.values()
                   if c.pending_task is not None and not c.pending_task.done()]
        for t in (self._task, self._housekeeper, self._fast_pusher, *pending):
            if t:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        self._task = self._housekeeper = self._fast_pusher = None
        if flatten:
            await self.broker.flatten_all("engine stop")
        if self.portfolio.mode == "paper":
            from .persist import save_paper_state
            save_paper_state(self.portfolio, self.risk.state)   # the session survives
        await self.feed.stop()
        log.info("trader stopped")

    # ------------------------------------------------------------ event loop

    async def _loop(self) -> None:
        while self.running:
            try:
                kind, symbol = await asyncio.wait_for(self.feed.events.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            try:
                if kind == "bar":
                    await self._on_bar(symbol)
                elif kind == "tick":
                    await self._on_tick(symbol)
            except Exception:  # noqa: BLE001 - the loop must survive anything
                log.exception("event %s/%s failed", kind, symbol)

    def settle_risk(self) -> None:
        """Feed every not-yet-accounted closed trade to the risk manager exactly
        once — the single choke-point for daily-loss / kill-switch / streak /
        health accounting. Closes that don't pass through the engine's own exit
        path (carry desk, live reconcile after an exchange-side SL/TP, manual
        flatten) used to bypass risk accounting entirely, so a stopped-out carry
        trade or an exchange-side stop never counted toward the daily loss cap."""
        trades = self.portfolio.trades
        if self._risk_settled > len(trades):     # paper reset cleared the list
            self._risk_settled = len(trades)
            return
        if self._risk_settled == len(trades):
            return
        marks = {s: st.mark_price() for s, st in self.feed.states.items()}
        equity = self.portfolio.equity(marks)
        for t in trades[self._risk_settled:]:
            self.risk.on_trade_closed(t, equity)
        self._risk_settled = len(trades)

    async def _housekeeping(self) -> None:
        beat = 0
        while self.running:
            await asyncio.sleep(5)
            beat += 1
            self.settle_risk()   # catch closes that bypassed the engine exit path
            marks = {s: st.mark_price() for s, st in self.feed.states.items()}
            self.portfolio.record_equity(now_ms(), marks)
            self.risk.health.mark_equity(self.portfolio.equity(marks))
            if isinstance(self.broker, LiveBroker) and beat % 6 == 0:
                await self.broker.reconcile(self.cfg.symbols)
            if self.risk.state.killed and self.portfolio.positions:
                await self.broker.flatten_all("kill switch")
            if self.record is not None:
                try:
                    self.record.maybe_roll(self.portfolio, self.portfolio.mode)
                except Exception as e:  # noqa: BLE001
                    log.warning("track record roll failed: %s", e)
            if self.portfolio.mode == "paper" and beat % 6 == 0:   # every 30s
                from .persist import save_paper_state
                save_paper_state(self.portfolio, self.risk.state)
            if self.on_update:
                await self.on_update("state")

    async def _fast_push_loop(self) -> None:
        """High-cadence 'hot' pushes so live prices / uPnL / stage refresh in
        near-real-time between bar closes — a small payload, separate from the
        heavy full-state push, so the two never contend for the same cycle."""
        while self.running:
            await asyncio.sleep(0.4)
            if self.on_update:
                try:
                    await self.on_update("hot")
                except Exception:  # noqa: BLE001
                    pass

    def _make_ctx(self, sym: str) -> SymbolCtx:
        s = self.cfg.strategy
        bars_per_hour = 3_600_000 / max(1, self._interval_ms())
        return SymbolCtx(sym, TradingBrain(
            eta=s.hedge_eta, weight_floor=s.weight_floor, horizon_bars=s.horizon_bars,
            base_threshold=s.base_threshold, threshold_adapt=s.threshold_adapt,
            target_trades_per_hour=s.target_trades_per_hour,
            bars_per_hour=bars_per_hour, cost_multiple=s.cost_multiple,
            min_p_win=s.min_p_win, kelly_fraction=s.kelly_fraction,
        ))

    async def adopt_symbol(self, sym: str) -> bool:
        """Radar adoption: attach the feed at runtime and give the symbol its own
        brain — it trades through the exact same gate chain as everyone else."""
        if sym in self.ctx:
            return True
        add = getattr(self.feed, "add_symbol", None)
        if add is None or not await add(sym):
            return False
        self.ctx[sym] = self._make_ctx(sym)
        if sym in self.overlays:   # a stored overlay follows the symbol back in
            self._apply_brain_params(self.ctx[sym].brain, self.cfg.strategy, self.overlays[sym])
        self.adopted.add(sym)
        log.info("ADOPTED %s (radar trend pick)", sym)
        return True

    async def drop_symbol(self, sym: str) -> bool:
        """Release an adopted symbol (never with an open position)."""
        if sym not in self.adopted or self.portfolio.positions.get(sym) is not None:
            return False
        c = self.ctx.get(sym)
        if c is not None and c.pending_task is not None and not c.pending_task.done():
            c.pending_task.cancel()   # a resting entry dies with the symbol
        self.ctx.pop(sym, None)
        self.adopted.discard(sym)
        rm = getattr(self.feed, "remove_symbol", None)
        if rm is not None:
            await rm(sym)
        log.info("DROPPED %s (trend gone)", sym)
        return True

    def focus_symbol(self) -> str:
        """The symbol the machine is 'looking at' right now: an open position
        wins; otherwise whichever is closest to firing (edge vs threshold)."""
        for sym in self.ctx:
            if self.portfolio.positions.get(sym) is not None:
                return sym
        best, best_v = "", -1.0
        for sym, c in self.ctx.items():
            thr = max(c.last_eval.get("threshold", 0.3), 1e-6) if c.last_eval else 0.3
            v = abs(c.last_eval.get("edge", 0.0)) / thr if c.last_eval else 0.0
            if v > best_v:
                best, best_v = sym, v
        return best or (next(iter(self.ctx), ""))

    def _push_tape(self, symbol: str, kind: str, side: str, price: float, extra: dict | None = None) -> None:
        row = {"ts": now_ms(), "symbol": symbol, "kind": kind, "side": side, "price": price}
        if extra:
            row.update(extra)
        self.tape.append(row)
        self.tape = self.tape[-60:]

    # ------------------------------------------------------------ bar logic

    async def _on_bar(self, symbol: str) -> None:
        ctx = self.ctx.get(symbol)
        st = self.feed.states.get(symbol)
        if ctx is None or st is None or ctx.busy:
            return
        n = len(st.candles)
        if n < self.cfg.strategy.warmup_bars:
            ctx.last_entry_block = f"warmup {n}/{self.cfg.strategy.warmup_bars}"
            ctx.set_stage("SCAN")
            return
        t0 = time.perf_counter()
        ff = FeatureFrame(st.candles.arrays(min(n, FEATURE_TAIL)), interval=self.cfg.strategy.interval)
        row = ff.row(-1)
        micro = st.micro_snapshot()
        data_ctx = st.context_snapshot()
        row["funding_rate"] = data_ctx.get("funding_rate") or 0.0
        ev = ctx.brain.evaluate(row, micro, data_ctx)
        ctx.eval_ms = (time.perf_counter() - t0) * 1000.0
        ctx.last_row = row
        ctx.last_eval = ev
        ctx.mtf = mtf_from_row(row, ff.ladder)
        ctx.react_ts = now_ms()   # the bar-close pass counts as a fresh scan

        pos = self.portfolio.positions.get(symbol)
        if pos is None:
            self._build_gates(ctx, st, row, ev)
        ctx.busy = True
        try:
            if pos is not None:
                ctx.bars_held += 1
                ctx.set_stage("MANAGE")
                await self._manage_position_on_bar(ctx, st, row, ev)
            else:
                ctx.bars_held = 0
                await self._try_enter(ctx, st, row, ev)
        finally:
            ctx.busy = False
        if self.on_update:
            await self.on_update("bar")

    async def _manage_position_on_bar(self, ctx: SymbolCtx, st: MarketState, row: dict, ev: dict) -> None:
        """Adaptive exit management on bar close — the brain decides whether the
        move continues (hold / advance the chandelier trail) or is done (exit)."""
        symbol = ctx.symbol
        pos = self.portfolio.positions.get(symbol)
        if pos is None:
            return
        moved, reason = self.exits.manage(
            pos, row["close"], row["high"], row["low"], row.get("atr", 0.0),
            row, ev["edge"], ev["threshold"], ev["regime"], ctx.bars_held)
        if reason:
            await self._close(symbol, reason)
        elif moved:
            log.info("%s stop -> %.6g (adaptive trail)", symbol, pos.stop_price)

    async def _try_enter(self, ctx: SymbolCtx, st: MarketState, row: dict, ev: dict) -> None:
        symbol = ctx.symbol
        edge, p_win = ev["edge"], ev["p_win"]
        if ctx.pending_task is not None and not ctx.pending_task.done():
            ctx.last_entry_block = "resting pullback limit"
            return
        ctx.set_stage("SCAN")
        marks = {s: s_st.mark_price() for s, s_st in self.feed.states.items()}
        equity = self.portfolio.equity(marks)
        spread = st.spread_bps.get(1.0)

        ok, why = self.risk.can_enter(equity, len(self.portfolio.positions), spread)
        if not ok:
            ctx.last_entry_block = why
            return
        spec = self.specs.get(symbol, ContractSpec(symbol))
        fees_rt, slip = self._entry_costs(symbol)
        if not _entry_signal_ok(ctx.brain, self.cfg.strategy, edge, p_win, row, ev, fees_rt, slip):
            ctx.last_entry_block = self._block_reason(ctx.brain, edge, p_win, row, ev, fees_rt, spread, slip)
            return
        if self.cfg.strategy.micro_confirm and not ctx.brain.micro_confirms(edge, st.micro_snapshot()):
            ctx.last_entry_block = "order-flow veto"
            return
        ctx.set_stage("VALIDATE")

        side = LONG if edge > 0 else SHORT
        style = "scalp" if ev["regime"] == "RANGE" else "trend"
        price = st.mark_price() or row["close"]
        atr = row.get("atr", 0.0)
        # pullback entry: plan the trade AT the resting limit (bracket, sizing,
        # exchange-side SL/TP all measured from where we'd actually fill), so
        # the geometry matches the backtest's fill-price bracket exactly.
        pull = self.cfg.strategy.entry_pullback_atr
        entry_ref = price
        if pull > 0 and style == "trend" and atr > 0:
            entry_ref = price - (1 if side == LONG else -1) * pull * atr
        bracket = self.exits.initial_bracket(entry_ref, side, atr, row, ev["regime"], style)
        if bracket is None:
            ctx.last_entry_block = "no volatility for bracket"
            return
        kelly = ctx.brain.kelly_size_mult(p_win, self.risk.payoff_ratio(style)) if self.cfg.strategy.use_kelly else 1.0
        size_mult = kelly * self.risk.health.scalar
        # a freshly ADOPTED symbol trades small until its brain has real graded
        # evidence — adoption gives it a seat, not full conviction on day one.
        if symbol in self.adopted and getattr(ctx.brain, "graded", 0) < ADOPTED_FULL_GRADED:
            size_mult *= 0.6
        # correlation haircut: shrink a same-direction add across symbols (BTC/ETH
        # move together — don't quietly double the same directional bet).
        side_d = 1 if side == LONG else -1
        others = [p for s, p in self.portfolio.positions.items() if s != symbol]
        if any(p.direction() == side_d for p in others):
            size_mult *= self.cfg.risk.correlation_haircut
        ctx.set_stage("SIZE")
        sized = self.risk.size_entry(equity, entry_ref, bracket.init_risk, side, spec, size_mult)
        if sized is None:
            ctx.last_entry_block = "size below exchange minimum"
            return
        # net directional exposure cap across the whole account
        same_dir_notional = sum(p.qty * (marks.get(p.symbol, p.entry_price) or p.entry_price)
                                for p in others if p.direction() == side_d)
        if equity > 0 and (same_dir_notional + sized.qty * entry_ref) > equity * self.cfg.risk.max_net_exposure:
            ctx.last_entry_block = "net exposure cap"
            return
        sized.stop_price = bracket.stop
        sized.take_profit = bracket.take_profit
        reason = f"edge {edge:+.2f} P{p_win:.0%} k{kelly:.2f} {ev['regime']}"
        # re-check right before the fill: another desk (carry) may have opened
        # this token while we were sizing — one position per token, always.
        if self.portfolio.positions.get(symbol) is not None:
            ctx.last_entry_block = "position already open on this token"
            return
        ctx.set_stage("FILL")
        if entry_ref != price:
            # rest the pullback limit in the BACKGROUND: the brain keeps
            # evaluating bars (and the UI stays live) while the limit waits for
            # the retrace; unfilled in the window = entry abandoned, not chased.
            sized.entry_limit = entry_ref
            sized.entry_wait_s = max(1, self.cfg.strategy.maker_wait_bars) * self._interval_ms() / 1000.0
            ctx.last_entry_block = f"resting pullback limit @ {entry_ref:.6g}"
            ctx.pending_task = asyncio.create_task(
                self._rest_entry(ctx, symbol, side, sized, reason, row, ev, style, bracket.init_risk, atr),
                name=f"pullback-{symbol}")
            return
        res = await self.broker.open_position(symbol, side, sized, reason, bar_ts=int(row["ts"]))
        ctx.last_entry_block = "" if res.ok else f"broker: {res.error}"
        if res.ok:
            await self._finalize_open(ctx, symbol, side, res, ev, row, style, bracket.init_risk, atr)

    async def _rest_entry(self, ctx: SymbolCtx, symbol: str, side: str, sized, reason: str,
                          row: dict, ev: dict, style: str, init_risk: float, atr: float) -> None:
        """Background pullback entry: the broker rests the limit (paper polls the
        live tape; live rests post-only on the exchange) until touch or expiry."""
        try:
            res = await self.broker.open_position(symbol, side, sized, reason, bar_ts=int(row["ts"]))
            if res.ok:
                ctx.last_entry_block = ""
                await self._finalize_open(ctx, symbol, side, res, ev, row, style, init_risk, atr)
            else:
                ctx.last_entry_block = f"pullback: {res.error}"
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — a failed entry task must never crash the engine
            log.exception("pullback entry task failed for %s", symbol)
            ctx.last_entry_block = "pullback entry error"

    async def _finalize_open(self, ctx: SymbolCtx, symbol: str, side: str, res,
                             ev: dict, row: dict, style: str, init_risk: float, atr: float) -> None:
        pos = self.portfolio.positions.get(symbol)
        if pos is not None:
            pos.style = style
            self.exits.attach(pos, atr, init_risk)
        ctx.entry_ctx = self._entry_context(ctx, ev, row)
        self._push_tape(symbol, "OPEN", side, res.filled_price,
                        {"p_win": round(ev.get("p_win", 0.0), 3), "edge": round(ev.get("edge", 0.0), 3)})
        if self.on_update:
            await self.on_update("trade")

    @staticmethod
    def _entry_context(ctx: SymbolCtx, ev: dict, row: dict) -> dict:
        """Snapshot the decision context at the open, for the trade journal."""
        alloc = ev.get("alloc", {})
        desk_sig = ev.get("desk_sig", {})
        # dominant desk = biggest signed contribution to the fused edge
        desk = max(alloc, key=lambda d: abs(alloc.get(d, 0.0) * desk_sig.get(d, 0.0)), default="")
        return {
            "regime": ev.get("regime", ""),
            "edge": round(ev.get("edge", 0.0), 4),
            "p_win": round(ev.get("p_win", 0.0), 4),
            "mtf_align": round(row.get("mtf_align", 0.0), 4),
            "mtf_bias": round(row.get("mtf_bias", 0.0), 4),
            "mtf": dict(ctx.mtf),
            "desk": desk,
            "funding_rate": round(row.get("funding_rate", 0.0), 6),
        }

    def _block_reason(self, brain, edge, p_win, row, ev, fees_rt, spread, slip) -> str:
        """The FIRST failing gate, with its live numbers — never a vague label.
        'trend ER 0.14 < 0.27' tells you exactly what the machine is waiting for;
        'awaiting aligned trend' told you nothing."""
        ok, why = brain.entry_ok(edge, p_win, row, fees_rt, spread, slip)
        if not ok:
            return why
        strat = self.cfg.strategy
        for fn in (gate_mtf_veto, gate_funding):
            ok, d = fn(strat, edge, row)
            if not ok:
                return d
        ok, d = gate_regime(strat, edge, row, ev["regime"])
        return d if not ok else ""

    def _entry_costs(self, symbol: str) -> tuple[float, float]:
        spec = self.specs.get(symbol, ContractSpec(symbol))
        fees_rt = spec.taker_fee + (spec.maker_fee if self.cfg.strategy.entry_mode == "maker" else spec.taker_fee)
        slip = self.cfg.paper.slippage_bps if self.portfolio.mode != "live" else 1.0
        return fees_rt, slip

    def _build_gates(self, ctx: SymbolCtx, st: MarketState, row: dict, ev: dict) -> None:
        """Refresh the symbol's entry-gate X-ray: EVERY rung of the entry chain
        with its live numbers, pass or fail, no early exit — so the UI can show
        exactly which gate is holding fire and by how much."""
        strat = self.cfg.strategy
        edge, p_win = ev["edge"], ev["p_win"]
        fees_rt, slip = self._entry_costs(ctx.symbol)
        spread = st.spread_bps.get(1.0)
        marks = {s: s_st.mark_price() for s, s_st in self.feed.states.items()}
        risk_ok, risk_why = self.risk.can_enter(
            self.portfolio.equity(marks), len(self.portfolio.positions), spread)
        gates = [{"n": "risk", "ok": risk_ok, "d": risk_why or "clear"}]
        gates += ctx.brain.entry_report(edge, p_win, row, fees_rt, spread or 1.0, slip)
        for name, (g_ok, d) in (("mtf veto", gate_mtf_veto(strat, edge, row)),
                                ("funding", gate_funding(strat, edge, row))):
            gates.append({"n": name, "ok": g_ok, "d": d})
        r_ok, r_d = gate_regime(strat, edge, row, ev["regime"])
        gates.append({"n": "regime", "ok": r_ok, "d": f"{ev['regime']} · {r_d}"})
        if strat.micro_confirm:
            micro = st.micro_snapshot()
            lean = 0.6 * micro.get("flow", 0.0) + 0.4 * micro.get("obi", 0.0)
            gates.append({"n": "order-flow", "ok": ctx.brain.micro_confirms(edge, micro),
                          "d": f"lean {lean:+.2f} vs edge {edge:+.2f}"})
        need = max(1, strat.entry_confirm_scans)
        gates.append({"n": "confirm", "ok": ctx.react_confirm >= need,
                      "d": f"{min(ctx.react_confirm, need)}/{need} scans agree"})
        ctx.gates = gates

    # ------------------------------------------------------------ tick logic

    async def _on_tick(self, symbol: str) -> None:
        ctx = self.ctx.get(symbol)
        st = self.feed.states.get(symbol)
        if ctx is None or st is None or ctx.busy:
            return
        pos = self.portfolio.positions.get(symbol)
        if pos is None:
            # flat: hunt for an entry on the live-forming bar + order book + flow,
            # throttled — this is what stops us waiting for a bar to close.
            await self._maybe_react(symbol, ctx, st)
            return
        price = st.last_price
        if price <= 0:
            return
        d = pos.direction()
        if pos.stop_price > 0 and (price - pos.stop_price) * d <= 0:
            ctx.busy = True
            try:
                await self._close(symbol, "stop loss" if not pos.breakeven_moved else "trailing stop")
            finally:
                ctx.busy = False
        elif pos.take_profit > 0 and (price - pos.take_profit) * d >= 0:
            ctx.busy = True
            try:
                await self._close(symbol, "take profit")
            finally:
                ctx.busy = False

    async def _maybe_react(self, symbol: str, ctx: SymbolCtx, st: MarketState) -> None:
        """Reactive intra-bar entry scan: re-score the brain on the still-forming
        bar plus live microstructure and, if it clears every gate, enter now
        instead of at the next close. Throttled per symbol; grading stays strictly
        one-per-closed-bar (score() only, never observe()) so the online weights
        are untouched."""
        now = now_ms()
        # adaptive throttle: the reactive scanner's CPU grows with the book —
        # widen the per-symbol gap as adoption adds symbols (3 symbols = 0.85s,
        # 6 symbols = 1.7s) so the event loop stays cool.
        gap = REACT_MS * max(1.0, len(self.ctx) / 3.0)
        if now - ctx.react_ts < gap:
            return
        ctx.react_ts = now
        n = len(st.candles)
        if n < self.cfg.strategy.warmup_bars:
            return
        t0 = time.perf_counter()
        ff = FeatureFrame(st.candles.arrays_live(min(n + 1, FEATURE_TAIL)), interval=self.cfg.strategy.interval)
        row = ff.row(-1)
        data_ctx = st.context_snapshot()
        row["funding_rate"] = data_ctx.get("funding_rate") or 0.0
        micro = st.micro_snapshot()
        ev = ctx.brain.score(row, micro, data_ctx)   # score only, no learning
        ctx.eval_ms = (time.perf_counter() - t0) * 1000.0
        ctx.last_row, ctx.last_eval = row, ev
        ctx.mtf = mtf_from_row(row, ff.ladder)
        self._build_gates(ctx, st, row, ev)

        # confirmation buffer: an intra-bar signal must persist in the same
        # direction across a few scans before we act, so a transient tick that
        # crosses threshold and immediately reverses can't trigger a whipsaw entry.
        edge, thr = ev["edge"], ev.get("threshold", 0.3)
        sig_dir = (1 if edge > 0 else -1) if abs(edge) >= thr else 0
        if sig_dir != 0 and sig_dir == ctx.react_dir:
            ctx.react_confirm += 1
        else:
            ctx.react_dir = sig_dir
            ctx.react_confirm = 1 if sig_dir != 0 else 0
        if sig_dir == 0 or ctx.react_confirm < max(1, self.cfg.strategy.entry_confirm_scans):
            if sig_dir != 0:
                ctx.last_entry_block = f"confirming {ctx.react_confirm}/{self.cfg.strategy.entry_confirm_scans}"
            return

        ctx.busy = True
        try:
            await self._try_enter(ctx, st, row, ev)
        finally:
            ctx.busy = False

    async def _close(self, symbol: str, reason: str) -> None:
        ctx = self.ctx.get(symbol)
        if ctx:
            ctx.set_stage("SETTLE")
        res = await self.broker.close_position(symbol, reason)
        if res.ok:
            trades = self.portfolio.trades
            if trades:
                self.settle_risk()   # exactly-once risk accounting for this close
                t = trades[-1]
                self._push_tape(symbol, "CLOSE", t.side, t.exit_price,
                                {"pnl": round(t.pnl, 4), "reason": reason})
                self._journal_trade(t, ctx.entry_ctx if ctx else {}, ctx.bars_held if ctx else 0)
            if ctx:
                ctx.bars_held = 0
                ctx.entry_ctx = {}
            if self.on_update:
                await self.on_update("trade")

    def _journal_trade(self, t, entry_ctx: dict, bars_held: int) -> None:
        if self.journal is None:
            return
        import datetime
        try:
            hour = datetime.datetime.utcfromtimestamp(t.entry_ts / 1000).hour
        except (OverflowError, OSError, ValueError):
            hour = -1
        row = {
            "ts": t.exit_ts, "symbol": t.symbol, "side": t.side, "qty": t.qty,
            "entry": t.entry_price, "exit": t.exit_price, "pnl": round(t.pnl, 6),
            "r": t.r_multiple, "mae_r": t.mae_r, "mfe_r": t.mfe_r,
            "fees": round(t.fees, 6), "bars_held": bars_held,
            "reason_open": t.reason_open, "reason_close": t.reason_close, "mode": t.mode,
            "hour": hour,
            "regime": entry_ctx.get("regime", ""), "edge": entry_ctx.get("edge", 0.0),
            "p_win": entry_ctx.get("p_win", 0.0), "mtf_align": entry_ctx.get("mtf_align", 0.0),
            "mtf_bias": entry_ctx.get("mtf_bias", 0.0), "desk": entry_ctx.get("desk", ""),
            "funding_rate": entry_ctx.get("funding_rate", 0.0), "mtf": entry_ctx.get("mtf", {}),
            "champion_id": self.active_champion_id,  # which vault champion took this trade
        }
        try:
            self.journal.record(row)
        except Exception as e:  # noqa: BLE001 - journaling must never break trading
            log.warning("journal record failed: %s", e)

    # ------------------------------------------------------------ snapshots

    def snapshot(self) -> dict:
        marks = {s: st.mark_price() for s, st in self.feed.states.items()}
        return {
            "running": self.running,
            "mode": self.portfolio.mode,
            "feed": type(self.feed).__name__,
            "feed_healthy": self.feed.healthy(),
            "started_ts": self.started_ts,
            "interval": self.cfg.strategy.interval,
            "focus": self.focus_symbol(),
            "adopted": sorted(self.adopted),
            "stages": list(STAGES),
            "tape": self.tape[-40:],
            "symbols": {
                sym: {
                    "price": marks.get(sym, 0.0),
                    "bars": len(self.feed.states[sym].candles),
                    "warmup_bars": self.cfg.strategy.warmup_bars,
                    "micro": self.feed.states[sym].micro_snapshot(),
                    "context": self.feed.states[sym].context_snapshot(),
                    "brain": c.brain.snapshot(),
                    "bars_held": c.bars_held,
                    "entry_block": c.last_entry_block,
                    "stage": c.stage,
                    "eval_ms": round(c.eval_ms, 2),
                    "mtf": c.mtf,
                    "gates": c.gates,
                    "overlay": sym in self.overlays,
                }
                for sym, c in self.ctx.items()
            },
            "portfolio": self.portfolio.to_dict(marks),
            "risk": self.risk.status(),
            "equity_curve": list(self.portfolio.equity_curve)[-600:],
            "trades": [dc_asdict(t) for t in self.portfolio.trades[-80:]],
        }

    def _live_candle(self, sym: str) -> dict | None:
        """The still-forming 1m DISPLAY bar (falls back to the signal bar) as a
        compact dict — rides the hot channel so the chart's last candle moves
        tick-by-tick instead of waiting for a REST poll."""
        st = self.feed.states.get(sym)
        if st is None:
            return None
        disp = getattr(st, "display", None)
        p = disp.partial if disp is not None else None
        if p is None and disp is not None and len(disp):
            p = disp.tail(1)[0]
        if p is None:
            p = st.candles.partial
            if p is None:
                if not len(st.candles):
                    return None
                p = st.candles.tail(1)[0]
        return {"t": p.ts // 1000, "o": p.open, "h": p.high, "l": p.low, "c": p.close}

    def hot(self) -> dict:
        """Small, fast-changing snapshot for the high-cadence UI channel: live
        prices, per-symbol edge/stage, open-position uPnL, equity — no brain
        internals, equity curve or trade history (those ride the slow channel)."""
        marks = {s: st.mark_price() for s, st in self.feed.states.items()}
        return {
            "running": self.running,
            "feed_healthy": self.feed.healthy(),
            "equity": round(self.portfolio.equity(marks), 4),
            "killed": self.risk.state.killed,
            "focus": self.focus_symbol(),
            "adopted": sorted(self.adopted),
            "symbols": {
                sym: {
                    "price": marks.get(sym, 0.0),
                    "stage": c.stage,
                    "eval_ms": round(c.eval_ms, 2),
                    "entry_block": c.last_entry_block,
                    "edge": round(c.last_eval.get("edge", 0.0), 4),
                    "p_win": round(c.last_eval.get("p_win", 0.0), 4),
                    "regime": c.last_eval.get("regime", ""),
                    "bars_held": c.bars_held,
                    "mtf": c.mtf,
                    "candle": self._live_candle(sym),
                    "gates": c.gates,
                }
                for sym, c in self.ctx.items()
            },
            "positions": {
                s: {
                    "side": p.side, "entry": p.entry_price, "stop": p.stop_price,
                    "tp": p.take_profit, "leverage": p.leverage,
                    "upnl": round(p.unrealized(marks.get(s, 0.0)), 4) if marks.get(s) else 0.0,
                }
                for s, p in self.portfolio.positions.items()
            },
            "tape": self.tape[-10:],
        }

    def _apply_brain_params(self, brain, strat, ov: dict | None) -> None:
        """Push brain scalars from the global strategy config, overridden by the
        symbol's overlay where one exists."""
        g = (lambda k, d: ov.get(k, d)) if ov else (lambda k, d: d)
        brain.base_threshold = g("base_threshold", strat.base_threshold)
        brain.cost_multiple = g("cost_multiple", strat.cost_multiple)
        brain.eta = g("hedge_eta", strat.hedge_eta)
        brain.horizon = max(1, int(g("horizon_bars", strat.horizon_bars)))
        brain.kelly_fraction = g("kelly_fraction", strat.kelly_fraction)
        brain.min_p_win = g("min_p_win", strat.min_p_win)
        brain.target_rate = max(0.1, g("target_trades_per_hour", strat.target_trades_per_hour))
        brain.threshold_adapt = strat.threshold_adapt

    def set_overlay(self, sym: str, params: dict | None) -> None:
        """Per-symbol brain overlay from the tuner: apply now if the symbol is
        live, keep it stored so adoption/restarts pick it up. None clears back
        to the global set."""
        if params:
            self.overlays[sym] = dict(params)
        else:
            self.overlays.pop(sym, None)
        c = self.ctx.get(sym)
        if c is not None:
            self._apply_brain_params(c.brain, self.cfg.strategy, self.overlays.get(sym))

    def hot_swap_params(self, strat) -> None:
        """Live-apply tuned strategy params to every symbol's brain (auto-tuner).
        Risk/exit params are read live from the shared cfg by reference, so only
        the brain's cached scalars need pushing here. Per-symbol overlays are
        re-applied on top so a global promotion never wipes them."""
        for sym, c in self.ctx.items():
            self._apply_brain_params(c.brain, strat, self.overlays.get(sym))
