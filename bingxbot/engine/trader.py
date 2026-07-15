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
from ..strategy.features import FeatureFrame
from ..util import now_ms
from .brokers import Broker, LiveBroker
from .portfolio import Portfolio

log = logging.getLogger("trader")

FEATURE_TAIL = 1400  # bars fed to FeatureFrame each close

# Execution-cycle stages surfaced to the UI pipeline.
STAGES = ("SCAN", "DETECT", "VALIDATE", "SIZE", "FILL", "MANAGE", "SETTLE")


class SymbolCtx:
    def __init__(self, symbol: str, brain: TradingBrain):
        self.symbol = symbol
        self.brain = brain
        self.last_row: dict = {}
        self.last_eval: dict = {}
        self.bars_held = 0
        self.last_entry_block = ""
        self.stage = "SCAN"
        self.stage_ts = 0
        self.eval_ms = 0.0
        self.busy = False  # guards against overlapping broker calls

    def set_stage(self, stage: str) -> None:
        self.stage = stage
        self.stage_ts = now_ms()


class TraderEngine:
    def __init__(self, cfg: BotConfig, feed: BaseFeed, broker: Broker, portfolio: Portfolio,
                 risk: RiskManager, specs: dict[str, ContractSpec], on_update=None):
        self.cfg = cfg
        self.feed = feed
        self.broker = broker
        self.portfolio = portfolio
        self.risk = risk
        self.specs = specs
        self.on_update = on_update  # async callback(kind: str) -> None
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
        log.info("trader started: %s %s %s", self.portfolio.mode, self.cfg.symbols, self.cfg.strategy.interval)

    async def stop(self, flatten: bool = False) -> None:
        self.running = False
        for t in (self._task, self._housekeeper):
            if t:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        self._task = self._housekeeper = None
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
        ff = FeatureFrame(st.candles.arrays(min(n, FEATURE_TAIL)))
        row = ff.row(-1)
        micro = st.micro_snapshot()
        data_ctx = st.context_snapshot()
        ev = ctx.brain.evaluate(row, micro, data_ctx)
        ctx.eval_ms = (time.perf_counter() - t0) * 1000.0
        ctx.last_row = row
        ctx.last_eval = ev

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
        symbol = ctx.symbol
        pos = self.portfolio.positions.get(symbol)
        if pos is None:
            return
        price = st.mark_price() or row["close"]
        atr_val = row.get("atr", 0.0)

        if self.risk.time_stop_hit(ctx.bars_held):
            await self._close(symbol, f"time stop ({ctx.bars_held} bars)")
            return
        edge, thr = ev["edge"], ev["threshold"]
        if edge * pos.direction() < 0 and abs(edge) >= 0.85 * thr:
            await self._close(symbol, f"opposite edge {edge:+.2f}")
            return
        if self.risk.update_trailing(pos, price, atr_val, ev["regime"]):
            log.info("%s stop -> %.6g (trail)", symbol, pos.stop_price)

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
        if abs(edge) < ev["threshold"]:
            ctx.last_entry_block = f"edge {edge:+.2f} < thr {ev['threshold']:.2f}"
            return
        ctx.set_stage("DETECT")
        spec = self.specs.get(symbol, ContractSpec(symbol))
        fees_rt = 2.0 * spec.taker_fee
        slip = self.cfg.paper.slippage_bps if self.portfolio.mode != "live" else 1.0
        ok, why = ctx.brain.entry_ok(edge, p_win, row, fees_rt, spread, slip)
        if not ok:
            ctx.last_entry_block = why
            return
        if self.cfg.strategy.micro_confirm and not ctx.brain.micro_confirms(edge, st.micro_snapshot()):
            ctx.last_entry_block = "order-flow veto"
            return
        ctx.set_stage("VALIDATE")

        side = LONG if edge > 0 else SHORT
        rt_cost = fees_rt + (spread + 2 * slip) / 10_000.0
        b = self.risk.payoff_ratio(ev["regime"])
        kelly = ctx.brain.kelly_size_mult(p_win, b) if self.cfg.strategy.use_kelly else 1.0
        size_mult = kelly * self.risk.health.scalar
        ctx.set_stage("SIZE")
        sized = self.risk.size_entry(equity, st.mark_price() or row["close"], row.get("atr", 0.0),
                                     side, spec, ev["regime"], roundtrip_cost_pct=rt_cost,
                                     size_mult=size_mult)
        if sized is None:
            ctx.last_entry_block = "size below exchange minimum"
            return
        reason = f"edge {edge:+.2f} P{p_win:.0%} k{kelly:.2f} {ev['regime']}"
        ctx.set_stage("FILL")
        res = await self.broker.open_position(symbol, side, sized, reason, bar_ts=int(row["ts"]))
        ctx.last_entry_block = "" if res.ok else f"broker: {res.error}"
        if res.ok:
            self._push_tape(symbol, "OPEN", side, res.filled_price,
                            {"p_win": round(p_win, 3), "edge": round(edge, 3)})
            if self.on_update:
                await self.on_update("trade")

    # ------------------------------------------------------------ tick logic

    async def _on_tick(self, symbol: str) -> None:
        pos = self.portfolio.positions.get(symbol)
        if pos is None:
            return
        ctx = self.ctx.get(symbol)
        st = self.feed.states.get(symbol)
        if ctx is None or st is None or ctx.busy:
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
            if ctx:
                ctx.bars_held = 0
            if self.on_update:
                await self.on_update("trade")

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
                }
                for sym, c in self.ctx.items()
            },
            "portfolio": self.portfolio.to_dict(marks),
            "risk": self.risk.status(),
            "equity_curve": list(self.portfolio.equity_curve)[-600:],
            "trades": [dc_asdict(t) for t in self.portfolio.trades[-80:]],
        }

    def hot_swap_params(self, strat) -> None:
        """Live-apply tuned strategy params to every symbol's brain (auto-tuner)."""
        for c in self.ctx.values():
            b = c.brain
            b.base_threshold = strat.base_threshold
            b.cost_multiple = strat.cost_multiple
            b.eta = strat.hedge_eta
            b.horizon = max(1, strat.horizon_bars)
            b.kelly_fraction = strat.kelly_fraction
            b.min_p_win = strat.min_p_win
