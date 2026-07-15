"""Orchestrator: owns the config and the running engine, switches modes
(idle / paper / live), and runs backtest & optimizer jobs off the event loop.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid

from ..config import (BotConfig, FEED_SYNTHETIC, MODE_IDLE, MODE_LIVE, MODE_PAPER,
                      config_public_dict, load_config, save_config, update_config)
from ..data.feed import BaseFeed, LiveFeed, SyntheticFeed
from ..data.history import HistoryStore, synthetic_candles
from ..engine.backtest import run_backtest, run_optimizer
from ..engine.brokers import LiveBroker, PaperBroker
from ..engine.portfolio import Portfolio
from ..engine.trader import TraderEngine
from ..exchange.errors import BingXError
from ..exchange.models import ContractSpec
from ..exchange.rest import BingXRest
from ..risk.manager import RiskManager
from ..util import interval_ms, now_ms

log = logging.getLogger("orchestrator")

LIVE_CONFIRM_PHRASE = "TRADE LIVE"


class Job:
    def __init__(self, kind: str):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.progress = 0.0
        self.result: dict | None = None
        self.error: str | None = None
        self.started = time.time()

    def to_dict(self, include_result: bool = True) -> dict:
        d = {"id": self.id, "kind": self.kind, "progress": round(self.progress, 4),
             "done": self.result is not None or self.error is not None,
             "error": self.error, "started": self.started}
        if include_result:
            d["result"] = self.result
        return d


class Orchestrator:
    def __init__(self, cfg: BotConfig | None = None):
        self.cfg = cfg or load_config()
        self.engine: TraderEngine | None = None
        self.rest: BingXRest | None = None
        self.autotuner = None       # set lazily to avoid import cycle
        self.specs: dict[str, ContractSpec] = {}
        self.jobs: dict[str, Job] = {}
        self.listeners: set[asyncio.Queue] = set()
        self.mode = MODE_IDLE
        self._switch_lock = asyncio.Lock()

    # ---------------------------------------------------------------- events

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self.listeners.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.listeners.discard(q)

    async def _notify(self, kind: str) -> None:
        for q in list(self.listeners):
            try:
                q.put_nowait(kind)
            except asyncio.QueueFull:
                pass

    # ---------------------------------------------------------------- helpers

    def _make_rest(self) -> BingXRest:
        return BingXRest(
            base_url=self.cfg.exchange.base_url,
            api_key=self.cfg.api_key,
            api_secret=self.cfg.api_secret,
            recv_window_ms=self.cfg.exchange.recv_window_ms,
        )

    async def _load_specs(self, rest: BingXRest) -> dict[str, ContractSpec]:
        try:
            specs = await rest.contracts()
            found = {s: specs[s] for s in self.cfg.symbols if s in specs}
            missing = [s for s in self.cfg.symbols if s not in specs]
            for m in missing:
                log.warning("no contract spec for %s, using defaults", m)
                found[m] = ContractSpec(m)
            return found
        except BingXError as e:
            log.warning("could not fetch contracts (%s); using defaults", e)
            return {s: ContractSpec(s) for s in self.cfg.symbols}

    def _build_feed(self, rest: BingXRest | None) -> BaseFeed:
        s = self.cfg.strategy
        if self.cfg.feed == FEED_SYNTHETIC or rest is None:
            import os
            speed = float(os.getenv("BOT_SYNTH_SPEED", "1.0"))
            return SyntheticFeed(self.cfg.symbols, s.interval,
                                 warmup_bars=s.warmup_bars + 80, speed=speed)
        return LiveFeed(rest, self.cfg.exchange.ws_url, self.cfg.symbols,
                        s.interval, s.warmup_bars)

    # ---------------------------------------------------------------- modes

    async def set_mode(self, mode: str, confirm: str = "") -> tuple[bool, str]:
        async with self._switch_lock:
            if mode == self.mode:
                return True, f"already {mode}"
            if mode not in (MODE_IDLE, MODE_PAPER, MODE_LIVE):
                return False, f"unknown mode {mode}"
            if mode == MODE_LIVE:
                if not self.cfg.allow_live:
                    return False, "live trading disabled: set allow_live=true in Settings first"
                if not self.cfg.has_keys():
                    return False, "no API keys: put BINGX_API_KEY / BINGX_API_SECRET in .env"
                if self.cfg.feed == FEED_SYNTHETIC:
                    return False, "live mode requires feed=bingx (synthetic feed is demo-only)"
                if confirm != LIVE_CONFIRM_PHRASE:
                    return False, f'confirmation phrase required: type "{LIVE_CONFIRM_PHRASE}"'

            await self._stop_engine()
            if mode == MODE_IDLE:
                self.mode = MODE_IDLE
                await self._notify("mode")
                return True, "engine idle"
            try:
                await self._start_engine(mode)
            except Exception as e:  # noqa: BLE001
                log.exception("failed to start %s mode", mode)
                self.mode = MODE_IDLE
                return False, f"start failed: {e}"
            self.mode = mode
            self.cfg.mode = mode
            save_config(self.cfg)
            await self._notify("mode")
            return True, f"{mode} engine running"

    async def _start_engine(self, mode: str) -> None:
        needs_rest = self.cfg.feed != FEED_SYNTHETIC or mode == MODE_LIVE
        rest = self._make_rest() if needs_rest else None
        self.rest = rest
        if rest is not None and mode == MODE_LIVE:
            await rest.sync_time()
        self.specs = await self._load_specs(rest) if rest else {s: ContractSpec(s) for s in self.cfg.symbols}

        feed = self._build_feed(rest)
        if mode == MODE_LIVE:
            bal = await rest.balance()
            equity = bal["equity"] or bal["balance"]
            portfolio = Portfolio(equity, mode="live")
            portfolio.live_equity = equity
            broker = LiveBroker(rest, portfolio, self.specs, self.cfg)
            try:
                await rest.set_position_mode(dual_side=True)
            except BingXError as e:
                log.info("position mode: %s (often already set)", e)
        else:
            portfolio = Portfolio(self.cfg.paper.starting_balance, mode="paper")
            spec0 = self.specs[self.cfg.symbols[0]] if self.specs else None
            taker = spec0.taker_fee if spec0 else self.cfg.exchange.taker_fee
            maker = spec0.maker_fee if spec0 else self.cfg.exchange.maker_fee
            broker = PaperBroker(portfolio, feed.states, self.specs,
                                 taker_fee=taker, slippage_bps=self.cfg.paper.slippage_bps,
                                 maker_fee=maker, entry_mode=self.cfg.strategy.entry_mode)
        risk = RiskManager(self.cfg.risk)

        async def on_update(kind: str) -> None:
            await self._notify(kind)

        self.engine = TraderEngine(self.cfg, feed, broker, portfolio, risk, self.specs, on_update)
        await self.engine.start()

        from ..engine.autotuner import AutoTuner
        self.autotuner = AutoTuner(self)
        self.autotuner.start()

    async def _stop_engine(self) -> None:
        if self.autotuner is not None:
            await self.autotuner.stop()
            self.autotuner = None
        if self.engine is not None:
            try:
                await self.engine.stop(flatten=False)
            finally:
                self.engine = None
        if self.rest is not None:
            await self.rest.close()
            self.rest = None

    async def startup(self) -> None:
        """Boot into the configured mode (never straight into live)."""
        boot = self.cfg.mode if self.cfg.mode == MODE_PAPER else MODE_IDLE
        if boot != MODE_IDLE:
            ok, msg = await self.set_mode(boot)
            log.info("boot mode %s: %s", boot, msg)

    async def shutdown(self) -> None:
        await self._stop_engine()

    # ---------------------------------------------------------------- control

    async def control(self, action: str, symbol: str = "") -> tuple[bool, str]:
        eng = self.engine
        if action == "kill":
            if eng:
                eng.risk.manual_kill("manual kill switch")
                await eng.broker.flatten_all("manual kill switch")
            return True, "kill switch engaged, positions flattened"
        if action == "reset_kill":
            if eng:
                eng.risk.reset_kill()
            return True, "kill switch reset"
        if action == "flatten":
            if eng:
                await eng.broker.flatten_all("manual flatten")
            return True, "all positions closed"
        if action == "close" and symbol:
            if eng:
                res = await eng.broker.close_position(symbol, "manual close")
                return res.ok, res.error or f"{symbol} closed"
            return False, "engine not running"
        return False, f"unknown action {action}"

    # ---------------------------------------------------------------- jobs

    async def _get_backtest_candles(self, symbol: str, interval: str, days: float,
                                    synthetic: bool, job: Job) -> list:
        if synthetic or self.cfg.feed == FEED_SYNTHETIC:
            bars = int(days * 86_400_000 / interval_ms(interval))
            return synthetic_candles(symbol, interval, max(bars, 2000),
                                     seed=abs(hash(symbol)) % 100_000)
        rest = self.rest or self._make_rest()
        store = HistoryStore(rest, self.cfg.data_dir)
        end = now_ms()
        start = end - int(days * 86_400_000)

        def prog(p: float) -> None:
            job.progress = 0.35 * p  # download phase is ~third of the job
        return await store.get_range(symbol, interval, start, end, progress=prog)

    def start_backtest(self, symbol: str, interval: str, days: float,
                       synthetic: bool = False) -> Job:
        job = Job("backtest")
        self.jobs[job.id] = job

        async def runner() -> None:
            try:
                candles = await self._get_backtest_candles(symbol, interval, days, synthetic, job)
                if len(candles) < 500:
                    raise ValueError(f"only {len(candles)} bars available for {symbol} {interval}")

                def prog(p: float) -> None:
                    job.progress = 0.35 + 0.65 * p
                spec = self.specs.get(symbol, ContractSpec(symbol))
                result = await asyncio.to_thread(
                    run_backtest, candles, symbol, interval,
                    self.cfg.strategy, self.cfg.risk, spec,
                    self.cfg.paper.starting_balance, spec.taker_fee,
                    self.cfg.paper.slippage_bps, 300, prog,
                )
                job.result = result
                job.progress = 1.0
            except Exception as e:  # noqa: BLE001
                log.exception("backtest job failed")
                job.error = str(e)
            await self._notify("job")

        asyncio.get_running_loop().create_task(runner())
        return job

    def start_optimizer(self, symbol: str, interval: str, days: float,
                        trials: int, synthetic: bool = False) -> Job:
        job = Job("optimize")
        self.jobs[job.id] = job

        async def runner() -> None:
            try:
                candles = await self._get_backtest_candles(symbol, interval, days, synthetic, job)
                if len(candles) < 2000:
                    raise ValueError(f"optimizer needs 2000+ bars, got {len(candles)}")

                def prog(p: float) -> None:
                    job.progress = 0.35 + 0.65 * p
                spec = self.specs.get(symbol, ContractSpec(symbol))
                result = await asyncio.to_thread(
                    run_optimizer, candles, symbol, interval,
                    self.cfg.strategy, self.cfg.risk, spec,
                    spec.taker_fee, self.cfg.paper.slippage_bps,
                    trials, None, prog,
                )
                job.result = result
                job.progress = 1.0
            except Exception as e:  # noqa: BLE001
                log.exception("optimizer job failed")
                job.error = str(e)
            await self._notify("job")

        asyncio.get_running_loop().create_task(runner())
        return job

    def apply_params(self, params: dict) -> None:
        """Promote optimizer parameters into the running config."""
        s, r = self.cfg.strategy, self.cfg.risk
        mapping = {
            "base_threshold": (s, float), "cost_multiple": (s, float),
            "hedge_eta": (s, float), "horizon_bars": (s, int),
            "atr_sl_mult": (r, float), "atr_tp_mult": (r, float),
            "trail_atr_mult": (r, float), "breakeven_rr": (r, float),
            "time_stop_bars": (r, int), "cost_floor_mult": (r, float),
        }
        for k, v in params.items():
            if k in mapping:
                target, cast = mapping[k]
                setattr(target, k, cast(v))
        save_config(self.cfg)
        if self.engine:
            self.engine.hot_swap_params(s)

    # ---------------------------------------------------------------- status

    def status(self) -> dict:
        d = {
            "ts": now_ms(),
            "mode": self.mode,
            "config": config_public_dict(self.cfg),
            "live_confirm_phrase": LIVE_CONFIRM_PHRASE,
            "jobs": {j.id: j.to_dict(include_result=False) for j in self.jobs.values()},
        }
        if self.engine:
            d["engine"] = self.engine.snapshot()
        if self.autotuner is not None:
            d["autotuner"] = self.autotuner.snapshot()
        return d

    def update_cfg(self, patch: dict) -> dict:
        restart_keys = {"symbols", "feed", "strategy", "exchange"}
        needs_restart = bool(restart_keys & set(patch.keys())) and self.mode != MODE_IDLE
        update_config(self.cfg, patch)
        return {"ok": True, "needs_restart": needs_restart,
                "config": config_public_dict(self.cfg)}
