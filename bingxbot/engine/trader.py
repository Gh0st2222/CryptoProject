"""TraderEngine: the realtime decision loop shared by paper and live modes.

Bar close  -> full ensemble evaluation, entry decisions, trailing/time exits.
Trade tick -> protective exit checks (stop / take-profit / trailing cross).

The identical AlphaEnsemble + RiskManager pipeline runs in the backtester, so
what you simulate is what trades.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict as dc_asdict

from ..config import BotConfig, MODE_LIVE
from ..data.feed import BaseFeed, MarketState
from ..exchange.models import LONG, SHORT, ContractSpec
from ..risk.manager import RiskManager
from ..strategy.ensemble import AlphaEnsemble
from ..strategy.features import FeatureFrame
from ..util import now_ms
from .brokers import Broker, LiveBroker
from .portfolio import Portfolio

log = logging.getLogger("trader")

FEATURE_TAIL = 1400  # bars fed to FeatureFrame each close (widest window is 240)


class SymbolCtx:
    def __init__(self, symbol: str, ensemble: AlphaEnsemble):
        self.symbol = symbol
        self.ensemble = ensemble
        self.last_row: dict = {}
        self.bars_held = 0
        self.last_eval: dict = {}
        self.last_entry_block = ""
        self.busy = False  # guards against overlapping broker calls


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
            sym: SymbolCtx(sym, AlphaEnsemble(
                eta=s.hedge_eta, weight_floor=s.weight_floor, horizon_bars=s.horizon_bars,
                base_threshold=s.base_threshold, threshold_adapt=s.threshold_adapt,
                target_trades_per_hour=s.target_trades_per_hour,
                bars_per_hour=bars_per_hour, cost_multiple=s.cost_multiple,
            ))
            for sym in cfg.symbols
        }
        self._task: asyncio.Task | None = None
        self._housekeeper: asyncio.Task | None = None
        self.running = False
        self.started_ts = 0

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
            if isinstance(self.broker, LiveBroker) and beat % 6 == 0:
                await self.broker.reconcile(self.cfg.symbols)
            if self.risk.state.killed and self.portfolio.positions:
                await self.broker.flatten_all("kill switch")
            if self.on_update:
                await self.on_update("state")

    # ------------------------------------------------------------ bar logic

    async def _on_bar(self, symbol: str) -> None:
        ctx = self.ctx.get(symbol)
        st = self.feed.states.get(symbol)
        if ctx is None or st is None or ctx.busy:
            return
        n = len(st.candles)
        if n < self.cfg.strategy.warmup_bars:
            ctx.last_entry_block = f"warmup {n}/{self.cfg.strategy.warmup_bars}"
            return
        ff = FeatureFrame(st.candles.arrays(min(n, FEATURE_TAIL)))
        row = ff.row(-1)
        micro = st.micro_snapshot()
        ev = ctx.ensemble.evaluate(row, micro)
        ctx.last_row = row
        ctx.last_eval = ev

        pos = self.portfolio.positions.get(symbol)
        ctx.busy = True
        try:
            if pos is not None:
                ctx.bars_held += 1
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
        # strong opposite ensemble signal -> exit early instead of riding to SL
        score, thr = ev["score"], ev["threshold"]
        if score * pos.direction() < 0 and abs(score) >= 0.85 * thr:
            await self._close(symbol, f"opposite signal {score:+.2f}")
            return
        if self.risk.update_trailing(pos, price, atr_val, ev["regime"]):
            log.info("%s stop -> %.6g (trail)", symbol, pos.stop_price)

    async def _try_enter(self, ctx: SymbolCtx, st: MarketState, row: dict, ev: dict) -> None:
        symbol = ctx.symbol
        score = ev["score"]
        marks = {s: s_st.mark_price() for s, s_st in self.feed.states.items()}
        equity = self.portfolio.equity(marks)
        spread = st.spread_bps.get(1.0)

        ok, why = self.risk.can_enter(equity, len(self.portfolio.positions), spread)
        if not ok:
            ctx.last_entry_block = why
            return
        spec = self.specs.get(symbol, ContractSpec(symbol))
        fees_rt = 2.0 * spec.taker_fee
        slip = self.cfg.paper.slippage_bps if self.portfolio.mode != "live" else 1.0
        ok, why = ctx.ensemble.entry_ok(score, row, fees_rt, spread, slip)
        if not ok:
            ctx.last_entry_block = why
            return
        if self.cfg.strategy.micro_confirm and not ctx.ensemble.micro_confirms(score, st.micro_snapshot()):
            ctx.last_entry_block = "order-flow veto"
            return

        side = LONG if score > 0 else SHORT
        rt_cost = fees_rt + (spread + 2 * slip) / 10_000.0
        sized = self.risk.size_entry(equity, st.mark_price() or row["close"], row.get("atr", 0.0),
                                     side, spec, ev["regime"], roundtrip_cost_pct=rt_cost)
        if sized is None:
            ctx.last_entry_block = "size below exchange minimum"
            return
        reason = f"score {score:+.2f} thr {ev['threshold']:.2f} {ev['regime']}"
        res = await self.broker.open_position(symbol, side, sized, reason, bar_ts=int(row["ts"]))
        ctx.last_entry_block = "" if res.ok else f"broker: {res.error}"
        if res.ok and self.on_update:
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
        res = await self.broker.close_position(symbol, reason)
        if res.ok:
            trades = self.portfolio.trades
            if trades:
                marks = {s: st.mark_price() for s, st in self.feed.states.items()}
                self.risk.on_trade_closed(trades[-1], self.portfolio.equity(marks))
            ctx = self.ctx.get(symbol)
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
            "symbols": {
                sym: {
                    "price": marks.get(sym, 0.0),
                    "bars": len(self.feed.states[sym].candles),
                    "warmup_bars": self.cfg.strategy.warmup_bars,
                    "micro": self.feed.states[sym].micro_snapshot(),
                    "ensemble": c.ensemble.snapshot(),
                    "bars_held": c.bars_held,
                    "entry_block": c.last_entry_block,
                }
                for sym, c in self.ctx.items()
            },
            "portfolio": self.portfolio.to_dict(marks),
            "risk": self.risk.status(),
            "equity_curve": list(self.portfolio.equity_curve)[-600:],
            "trades": [dc_asdict(t) for t in self.portfolio.trades[-80:]],
        }
