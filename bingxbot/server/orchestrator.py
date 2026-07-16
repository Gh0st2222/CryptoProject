"""Orchestrator: owns the config and the running engine, switches modes
(idle / paper / live), and runs backtest & optimizer jobs off the event loop.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path

from ..config import (ROOT, BotConfig, FEED_SYNTHETIC, MODE_IDLE, MODE_LIVE, MODE_PAPER,
                      config_public_dict, load_config, save_config, update_config)
from ..data.feed import BaseFeed, LiveFeed, SyntheticFeed
from ..data.history import HistoryStore, synthetic_candles
from ..engine.backtest import run_backtest, run_optimizer, run_portfolio_backtest
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
CHAMPIONS_PATH = ROOT / "data_cache" / "champions.json"
CHAMPIONS_KEEP = 12   # keep the top-N champions; prune the worst/old


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
        self.pool = None            # ProcessPoolExecutor, created lazily
        self._pool_tried = False
        self.champions: list[dict] = self._load_champions()

    # ------------------------------------------------------------- CPU offload

    def _ensure_pool(self):
        """Create a process pool on first use so heavy CPU work (tuner,
        backtests) runs off the event loop on other cores — the GIL fix for the
        UI lag. Falls back to threads if processes can't be spawned."""
        if self._pool_tried:
            return self.pool
        self._pool_tried = True
        try:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            workers = min(4, max(1, (os.cpu_count() or 2) - 1))
            # 'spawn' (not fork): safe to create from inside a running event
            # loop — fork would inherit the loop's locks and can deadlock — and
            # it's the only start method on Windows, where the user also runs.
            ctx = mp.get_context("spawn")
            self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
            log.info("process pool: %d spawn workers", workers)
        except Exception as e:  # noqa: BLE001
            log.warning("process pool unavailable (%s); using threads", e)
            self.pool = None
        return self.pool

    async def run_cpu(self, fn, *args):
        loop = asyncio.get_running_loop()
        pool = self._ensure_pool()
        if pool is not None:
            try:
                return await loop.run_in_executor(pool, fn, *args)
            except Exception as e:  # noqa: BLE001 - worker/pickling failure -> threads
                log.warning("pool task failed (%s); falling back to thread", e)
                self.pool = None
        return await asyncio.to_thread(fn, *args)

    # ------------------------------------------------------------- champions

    def _load_champions(self) -> list[dict]:
        try:
            return json.loads(CHAMPIONS_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            return []

    def record_champion(self, params: dict, fitness: float, stats: dict) -> None:
        """Save a promoted champion, keep the best CHAMPIONS_KEEP, prune the rest."""
        self.champions.append({
            "ts": now_ms(), "fitness": round(float(fitness), 3),
            "win_rate": round(stats.get("win_rate", 0.0), 4),
            "profit_factor": round(stats.get("profit_factor", 0.0), 3),
            "params": params,
        })
        # keep the top-N by fitness (auto-delete the worst/old)
        self.champions.sort(key=lambda x: x["fitness"], reverse=True)
        self.champions = self.champions[:CHAMPIONS_KEEP]
        try:
            CHAMPIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
            CHAMPIONS_PATH.write_text(json.dumps(self.champions, indent=2))
        except OSError as e:
            log.warning("could not save champions: %s", e)

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
                job.progress = 0.4
                await self._notify("job")
                spec = self.specs.get(symbol, ContractSpec(symbol))
                # heavy sim runs on another core (process pool) so the UI stays live
                result = await self.run_cpu(
                    run_backtest, candles, symbol, interval,
                    self.cfg.strategy, self.cfg.risk, spec,
                    self.cfg.paper.starting_balance, spec.taker_fee,
                    self.cfg.paper.slippage_bps, 300, None,
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
                job.progress = 0.4
                await self._notify("job")
                spec = self.specs.get(symbol, ContractSpec(symbol))
                # random search is the heaviest job — run it on another core
                result = await self.run_cpu(
                    run_optimizer, candles, symbol, interval,
                    self.cfg.strategy, self.cfg.risk, spec,
                    spec.taker_fee, self.cfg.paper.slippage_bps,
                    trials, None, None,
                )
                job.result = result
                job.progress = 1.0
            except Exception as e:  # noqa: BLE001
                log.exception("optimizer job failed")
                job.error = str(e)
            await self._notify("job")

        asyncio.get_running_loop().create_task(runner())
        return job

    def start_portfolio_backtest(self, symbols: list[str], interval: str, days: float,
                                 synthetic: bool = False) -> Job:
        """Backtest several symbols on ONE shared account (diversified sizing,
        one position cap, one kill switch, correlation haircut)."""
        job = Job("portfolio")
        self.jobs[job.id] = job
        syms = [s.strip().upper() for s in symbols if s.strip()]

        async def runner() -> None:
            try:
                if len(syms) < 2:
                    raise ValueError("portfolio backtest needs at least 2 symbols")
                candles_by_symbol: dict[str, list] = {}
                for k, sym in enumerate(syms):
                    cs = await self._get_backtest_candles(sym, interval, days, synthetic, job)
                    if len(cs) >= 500:
                        candles_by_symbol[sym] = cs
                    job.progress = 0.4 * (k + 1) / len(syms)
                    await self._notify("job")
                if len(candles_by_symbol) < 2:
                    raise ValueError("need at least 2 symbols with enough history")
                specs = {s: self.specs.get(s, ContractSpec(s)) for s in candles_by_symbol}
                spec0 = next(iter(specs.values()))
                result = await self.run_cpu(
                    run_portfolio_backtest, candles_by_symbol, interval,
                    self.cfg.strategy, self.cfg.risk, specs,
                    self.cfg.paper.starting_balance, spec0.taker_fee,
                    self.cfg.paper.slippage_bps, 300, None,
                )
                job.result = result
                job.progress = 1.0
            except Exception as e:  # noqa: BLE001
                log.exception("portfolio backtest job failed")
                job.error = str(e)
            await self._notify("job")

        asyncio.get_running_loop().create_task(runner())
        return job

    def apply_params(self, params: dict) -> None:
        """Promote tuned parameters into the running config (in place, so the
        risk manager and exit engine — which hold the same cfg by reference —
        pick them up immediately) and hot-swap the brains."""
        from ..engine.backtest import apply_tunables_inplace
        apply_tunables_inplace(self.cfg.strategy, self.cfg.risk, params)
        save_config(self.cfg)
        if self.engine:
            self.engine.hot_swap_params(self.cfg.strategy)

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
        if self.champions:
            d["champions"] = self.champions[:CHAMPIONS_KEEP]
        return d

    def hot(self) -> dict:
        """Tiny, fast-changing snapshot for the high-cadence UI channel."""
        d = {"ts": now_ms(), "mode": self.mode}
        if self.engine:
            d["engine"] = self.engine.hot()
        return d

    def update_cfg(self, patch: dict) -> dict:
        # Only data-shape changes need an engine restart; risk band, max
        # positions and auto-tune toggle all take effect live (read by ref).
        before = (tuple(self.cfg.symbols), self.cfg.feed, self.cfg.strategy.interval)
        update_config(self.cfg, patch)
        after = (tuple(self.cfg.symbols), self.cfg.feed, self.cfg.strategy.interval)
        needs_restart = before != after and self.mode != MODE_IDLE
        return {"ok": True, "needs_restart": needs_restart,
                "config": config_public_dict(self.cfg)}
