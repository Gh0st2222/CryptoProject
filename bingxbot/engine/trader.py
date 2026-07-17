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
from .backtest import _entry_signal_ok
from .brokers import Broker, LiveBroker
from .portfolio import Portfolio

log = logging.getLogger("trader")

FEATURE_TAIL = 1400  # bars fed to FeatureFrame each close
REACT_MS = 850       # min gap between reactive intra-bar scans, per symbol

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
        self.busy = False  # guards against overlapping broker calls

    def set_stage(self, stage: str) -> None:
        self.stage = stage
        self.stage_ts = now_ms()


class TraderEngine:
    def __init__(self, cfg: BotConfig, feed: BaseFeed, broker: Broker, portfolio: Portfolio,
                 risk: RiskManager, specs: dict[str, ContractSpec], on_update=None, journal=None):
        self.cfg = cfg
        self.feed = feed
        self.broker = broker
        self.portfolio = portfolio
        self.risk = risk
        self.specs = specs
        self.exits = AdaptiveExitManager(cfg.risk)
        self.on_update = on_update  # async callback(kind: str) -> None
        self.journal = journal      # TradeJournal (records closed trades + context)
        self.active_champion_id: str | None = None  # vault champion currently driving trades (journal tag)
        s = cfg.strategy
        bars_per_hour = 3_600_000 / max(1, self._interval_ms())
        self.ctx: dict[str, SymbolCtx] = {
            sym: SymbolCtx(sym, TradingBrain(
                eta=s.hedge_eta, weight_floor=s.weight_floor, horizon_bars=s.horizon_bars,
                base_threshold=s.base_threshold, threshold_adapt=s.threshold_adapt,
                target_trades_per_hour=s.target_trades_per_hour,
                bars_per_hour=bars_per_hour, cost_multiple=s.cost_multiple,
                min_p_win=s.min_p_win, kelly_fraction=s.kelly_fraction,
            ))
            for sym in cfg.symbols
        }
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
        for t in (self._task, self._housekeeper, self._fast_pusher):
            if t:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        self._task = self._housekeeper = self._fast_pusher = None
        if flatten:
            await self.broker.flatten_all("engine stop")
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

    async def _housekeeping(self) -> None:
        beat = 0
        while self.running:
            await asyncio.sleep(5)
            beat += 1
            marks = {s: st.mark_price() for s, st in self.feed.states.items()}
            self.portfolio.record_equity(now_ms(), marks)
            self.risk.health.mark_equity(self.portfolio.equity(marks))
            if isinstance(self.broker, LiveBroker) and beat % 6 == 0:
                await self.broker.reconcile(self.cfg.symbols)
            if self.risk.state.killed and self.portfolio.positions:
                await self.broker.flatten_all("kill switch")
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
        ctx.set_stage("SCAN")
        marks = {s: s_st.mark_price() for s, s_st in self.feed.states.items()}
        equity = self.portfolio.equity(marks)
        spread = st.spread_bps.get(1.0)

        ok, why = self.risk.can_enter(equity, len(self.portfolio.positions), spread)
        if not ok:
            ctx.last_entry_block = why
            return
        spec = self.specs.get(symbol, ContractSpec(symbol))
        fees_rt = spec.taker_fee + (spec.maker_fee if self.cfg.strategy.entry_mode == "maker" else spec.taker_fee)
        slip = self.cfg.paper.slippage_bps if self.portfolio.mode != "live" else 1.0
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
        bracket = self.exits.initial_bracket(price, side, row.get("atr", 0.0), row, ev["regime"], style)
        if bracket is None:
            ctx.last_entry_block = "no volatility for bracket"
            return
        kelly = ctx.brain.kelly_size_mult(p_win, self.risk.payoff_ratio(style)) if self.cfg.strategy.use_kelly else 1.0
        size_mult = kelly * self.risk.health.scalar
        # correlation haircut: shrink a same-direction add across symbols (BTC/ETH
        # move together — don't quietly double the same directional bet).
        side_d = 1 if side == LONG else -1
        others = [p for s, p in self.portfolio.positions.items() if s != symbol]
        if any(p.direction() == side_d for p in others):
            size_mult *= self.cfg.risk.correlation_haircut
        ctx.set_stage("SIZE")
        sized = self.risk.size_entry(equity, price, bracket.init_risk, side, spec, size_mult)
        if sized is None:
            ctx.last_entry_block = "size below exchange minimum"
            return
        # net directional exposure cap across the whole account
        same_dir_notional = sum(p.qty * (marks.get(p.symbol, p.entry_price) or p.entry_price)
                                for p in others if p.direction() == side_d)
        if equity > 0 and (same_dir_notional + sized.qty * price) > equity * self.cfg.risk.max_net_exposure:
            ctx.last_entry_block = "net exposure cap"
            return
        sized.stop_price = bracket.stop
        sized.take_profit = bracket.take_profit
        reason = f"edge {edge:+.2f} P{p_win:.0%} k{kelly:.2f} {ev['regime']}"
        ctx.set_stage("FILL")
        res = await self.broker.open_position(symbol, side, sized, reason, bar_ts=int(row["ts"]))
        ctx.last_entry_block = "" if res.ok else f"broker: {res.error}"
        if res.ok:
            pos = self.portfolio.positions.get(symbol)
            if pos is not None:
                pos.style = style
                self.exits.attach(pos, row.get("atr", 0.0), bracket.init_risk)
            ctx.entry_ctx = self._entry_context(ctx, ev, row)
            self._push_tape(symbol, "OPEN", side, res.filled_price,
                            {"p_win": round(p_win, 3), "edge": round(edge, 3)})
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

    @staticmethod
    def _block_reason(brain, edge, p_win, row, ev, fees_rt, spread, slip) -> str:
        ok, why = brain.entry_ok(edge, p_win, row, fees_rt, spread, slip)
        if not ok:
            return why
        r = ev["regime"]
        if r == "RANGE":
            return "range (discipline: trends only)"
        if r == "VOLATILE":
            return "volatile regime (sitting out)"
        return "awaiting aligned trend"

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
        if now - ctx.react_ts < REACT_MS:
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
                marks = {s: st.mark_price() for s, st in self.feed.states.items()}
                self.risk.on_trade_closed(trades[-1], self.portfolio.equity(marks))
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
            "r": t.r_multiple, "fees": round(t.fees, 6), "bars_held": bars_held,
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
                }
                for sym, c in self.ctx.items()
            },
            "portfolio": self.portfolio.to_dict(marks),
            "risk": self.risk.status(),
            "equity_curve": list(self.portfolio.equity_curve)[-600:],
            "trades": [dc_asdict(t) for t in self.portfolio.trades[-80:]],
        }

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

    def hot_swap_params(self, strat) -> None:
        """Live-apply tuned strategy params to every symbol's brain (auto-tuner).
        Risk/exit params are read live from the shared cfg by reference, so only
        the brain's cached scalars need pushing here."""
        for c in self.ctx.values():
            b = c.brain
            b.base_threshold = strat.base_threshold
            b.cost_multiple = strat.cost_multiple
            b.eta = strat.hedge_eta
            b.horizon = max(1, strat.horizon_bars)
            b.kelly_fraction = strat.kelly_fraction
            b.min_p_win = strat.min_p_win
            b.target_rate = max(0.1, strat.target_trades_per_hour)
            b.threshold_adapt = strat.threshold_adapt
