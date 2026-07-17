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
from ..engine.backtest import run_backtest, run_optimizer, run_portfolio_backtest, run_walkforward
from ..engine.brokers import LiveBroker, PaperBroker
from ..engine.journal import TradeJournal
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
CHAMPIONS_KEEP = 100         # vault capacity
CHAMPIONS_PROTECT_USED = 15  # the most-used champions are never pruned (proven, not just high-scoring)
CHAMPION_STALE_DAYS = 7      # never-used champions age out of the vault after a week


def _split_symbols(raw) -> list[str]:
    """Normalize a symbol field into a clean list — tolerates a single string,
    a list, or a comma/semicolon-joined blob (so 'BTC-USDT, ETH-USDT' can't be
    sent to the single-contract klines endpoint as one bogus symbol)."""
    items = raw if isinstance(raw, (list, tuple)) else [raw]
    out: list[str] = []
    for item in items:
        for tok in str(item).replace(";", ",").split(","):
            tok = tok.strip().upper()
            if tok and tok not in out:
                out.append(tok)
    return out


def _clean_symbol(raw) -> str:
    lst = _split_symbols(raw)
    return lst[0] if lst else ""


def _plan_cores() -> tuple[int, int]:
    """Split the host's logical cores between the two process pools. Reserve one
    for the event loop, give a small slice to on-demand user jobs (interactive),
    and hand the rest to the always-on research desk (the auto-tuner) — so the
    tuner can use several cores in parallel without ever starving the UI or a
    backtest the user just launched. Scales with whatever CPU it lands on."""
    total = os.cpu_count() or 4
    if total <= 2:
        return 1, 1
    if total <= 4:
        return 1, max(1, total - 2)              # 4 -> 1 interactive + 2 research (+1 loop)
    interactive = min(3, max(1, total // 5))      # ~20%, capped at 3
    research = max(1, min(total - 1 - interactive, 16))
    return interactive, research


class Alerter:
    """Optional push alerts to a webhook (Slack/Discord/Telegram-bridge/etc.).
    URL comes from BOT_ALERT_WEBHOOK; with none set every method is a no-op, so
    this never adds a dependency or a failure mode to trading."""

    def __init__(self) -> None:
        self.url = os.getenv("BOT_ALERT_WEBHOOK", "").strip()
        self._killed = False
        self._daily_key = ""
        self._diverged = False

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    async def _post(self, text: str) -> None:
        if not self.url:
            return
        try:
            import aiohttp
            async with aiohttp.ClientSession() as s:
                await s.post(self.url, json={"text": text},
                             timeout=aiohttp.ClientTimeout(total=8))
        except Exception as e:  # noqa: BLE001 - alerts must never break the bot
            log.debug("alert post failed: %s", e)

    async def check(self, mode: str, engine, divergence: dict | None) -> None:
        if engine is None:
            return
        killed = bool(engine.risk.state.killed)
        if killed and not self._killed:
            await self._post(f"⛔ KILL SWITCH engaged ({engine.risk.state.kill_reason or 'daily loss'}). "
                             f"Positions flattened, entries halted.")
        self._killed = killed
        if divergence and divergence.get("diverged") and not self._diverged:
            await self._post(f"⚠️ Live diverging from backtest: win rate "
                             f"{divergence.get('live_win_rate', 0):.0%} vs expected "
                             f"{divergence.get('expected_win_rate', 0):.0%} over "
                             f"{divergence.get('live_trades', 0)} trades.")
        self._diverged = bool(divergence and divergence.get("diverged"))
        # once-a-day summary
        import datetime
        key = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        if key != self._daily_key:
            if self._daily_key:  # skip the very first (startup) tick
                st = engine.portfolio.stats()
                await self._post(f"📊 Daily [{mode}] equity {st.get('equity', 0):,.2f} · "
                                 f"{st.get('trades', 0)} trades · WR {st.get('win_rate', 0):.0%} · "
                                 f"PF {st.get('profit_factor', 0):.2f}")
            self._daily_key = key


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
        self.pool = None            # interactive pool (user backtests/optimizer/walk-fwd)
        self.research_pool = None   # dedicated pool for the auto-tuner's research
        self._pools_ready = False
        self.cores = _plan_cores()  # (interactive_workers, research_workers)
        self.champions: list[dict] = self._load_champions()
        # restore which champion was driving trading before the restart (most
        # recently used) so journal tagging + the LIVE badge survive restarts.
        used = [c for c in self.champions if c.get("last_used_ts", 0) > 0]
        self.active_champion_id: str | None = (
            max(used, key=lambda c: c["last_used_ts"])["id"] if used else None)
        self.journal = TradeJournal()
        self.alerter = Alerter()
        self._alert_task: asyncio.Task | None = None
        self.scanner = None         # MarketScanner (universe radar), set on engine start
        self.carry = None           # CarryDesk (funding harvest), set on engine start

    # ------------------------------------------------------------- CPU offload

    def _ensure_pools(self):
        """Create both process pools on first use ('spawn' start method — safe
        from inside a running event loop, and the only option on Windows). The
        interactive pool serves user-triggered jobs; the research pool is the
        auto-tuner's, sized to the host so cycles run several-wide in parallel."""
        if self._pools_ready:
            return
        self._pools_ready = True
        try:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            ctx = mp.get_context("spawn")
            ni, nr = self.cores
            self.pool = ProcessPoolExecutor(max_workers=ni, mp_context=ctx)
            self.research_pool = ProcessPoolExecutor(max_workers=nr, mp_context=ctx)
            log.info("cores: %d interactive + %d research (of %d logical detected)",
                     ni, nr, os.cpu_count() or 0)
        except Exception as e:  # noqa: BLE001
            log.warning("process pools unavailable (%s); using threads", e)
            self.pool = self.research_pool = None

    def _pool_for(self, research: bool):
        self._ensure_pools()
        return self.research_pool if research else self.pool

    async def run_cpu(self, fn, *args, research: bool = False):
        loop = asyncio.get_running_loop()
        pool = self._pool_for(research)
        if pool is not None:
            try:
                return await loop.run_in_executor(pool, fn, *args)
            except Exception as e:  # noqa: BLE001 - worker/pickling failure -> threads
                log.warning("pool task failed (%s); falling back to thread", e)
        return await asyncio.to_thread(fn, *args)

    async def map_cpu(self, fn, arg_tuples: list[tuple], research: bool = True) -> list:
        """Run fn(*args) for every args tuple in PARALLEL across a pool and return
        the results in order — this is what lets one tuner cycle use many cores at
        once. Falls back to running them sequentially on threads if the pool is
        unavailable, so the tuner still works on a single core, just slower."""
        loop = asyncio.get_running_loop()
        pool = self._pool_for(research)
        if pool is not None:
            try:
                futs = [loop.run_in_executor(pool, fn, *a) for a in arg_tuples]
                return list(await asyncio.gather(*futs))
            except Exception as e:  # noqa: BLE001
                log.warning("pool map failed (%s); thread fallback", e)
        return [await asyncio.to_thread(fn, *a) for a in arg_tuples]

    @property
    def research_workers(self) -> int:
        return self.cores[1]

    # ------------------------------------------------------------- champions

    def _load_champions(self) -> list[dict]:
        try:
            raw = json.loads(CHAMPIONS_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            return []
        return [self._migrate_champion(c) for c in raw] if isinstance(raw, list) else []

    @staticmethod
    def _migrate_champion(c: dict) -> dict:
        """Bring an old-schema record up to the enriched schema so a vault written
        by an earlier build still loads — the birth eval defaults to the current
        eval (they were the same thing before we split them)."""
        c.setdefault("id", uuid.uuid4().hex[:8])
        c.setdefault("born_ts", c.get("ts", now_ms()))
        c.setdefault("cur_trades", 0)
        c.setdefault("cur_ts", c.get("ts", c["born_ts"]))
        c.setdefault("birth_fitness", c.get("fitness", 0.0))
        c.setdefault("birth_wr", c.get("win_rate", 0.0))
        c.setdefault("birth_pf", c.get("profit_factor", 0.0))
        c.setdefault("birth_trades", c.get("cur_trades", 0))
        c.setdefault("uses", 0)
        c.setdefault("last_used_ts", 0)
        c.setdefault("params", {})
        return c

    @staticmethod
    def _params_match(a: dict, b: dict) -> bool:
        keys = set(a) | set(b)
        return bool(keys) and all(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) <= 1e-9 for k in keys)

    def find_champion(self, cid: str | None) -> dict | None:
        return next((c for c in self.champions if c.get("id") == cid), None) if cid else None

    def record_champion(self, params: dict, fitness: float, stats: dict) -> str:
        """Save a freshly promoted set as a vault champion and return its id.
        Stores BOTH the birth evaluation (now, at generation time) and the current
        evaluation (identical at birth; they diverge as the set is re-validated
        against a moving market). An all-but-identical set is refreshed in place
        rather than duplicated."""
        existing = next((c for c in self.champions if self._params_match(c.get("params", {}), params)), None)
        if existing is not None:
            self.set_champion_current(existing["id"], fitness, stats)
            self.save_champions()
            return existing["id"]
        cid = uuid.uuid4().hex[:8]
        fit = round(float(fitness), 3)
        wr = round(stats.get("win_rate", 0.0), 4)
        pf = round(stats.get("profit_factor", 0.0), 3)
        tr = int(stats.get("trades", 0))
        self.champions.append({
            "id": cid, "born_ts": now_ms(),
            "birth_fitness": fit, "birth_wr": wr, "birth_pf": pf, "birth_trades": tr,
            "fitness": fit, "win_rate": wr, "profit_factor": pf, "cur_trades": tr,
            "cur_ts": now_ms(), "uses": 0, "last_used_ts": 0, "params": dict(params),
        })
        self.prune_champions()
        return cid

    def set_champion_current(self, cid: str, fitness: float, stats: dict) -> None:
        """Update a champion's CURRENT (re-validated-against-today) evaluation,
        leaving the birth evaluation untouched — the vault shows both side by side.
        Does not persist; callers batch a save after updating many."""
        c = self.find_champion(cid)
        if c is None:
            return
        c["fitness"] = round(float(fitness), 3)
        c["win_rate"] = round(stats.get("win_rate", c.get("win_rate", 0.0)), 4)
        c["profit_factor"] = round(stats.get("profit_factor", c.get("profit_factor", 0.0)), 3)
        c["cur_trades"] = int(stats.get("trades", c.get("cur_trades", 0)))
        c["cur_ts"] = now_ms()

    def mark_champion_used(self, cid: str) -> None:
        """Record that a champion is now the one driving trading: bump its use
        count, timestamp it, and tag it active so live trades journal under its id
        — that's how the vault later shows a champion's REAL trades and PnL."""
        c = self.find_champion(cid)
        if c is None:
            return
        c["uses"] = int(c.get("uses", 0)) + 1
        c["last_used_ts"] = now_ms()
        self.active_champion_id = cid
        if self.engine is not None:
            self.engine.active_champion_id = cid
        self.save_champions()

    def prune_champions(self) -> None:
        """Cap the vault at CHAMPIONS_KEEP while protecting what's PROVEN, not just
        what scores high right now: the most-used sets (top CHAMPIONS_PROTECT_USED
        by uses) and the currently-active set are always kept; never-used sets past
        CHAMPION_STALE_DAYS age out (the weekly cleanup); the rest of the room goes
        to the best remaining by CURRENT fitness."""
        champs = self.champions
        protected = {self.active_champion_id} if self.active_champion_id else set()
        used = sorted((c for c in champs if c.get("uses", 0) > 0),
                      key=lambda x: (x.get("uses", 0), x.get("last_used_ts", 0)), reverse=True)
        protected.update(c["id"] for c in used[:CHAMPIONS_PROTECT_USED])

        now = now_ms()
        stale_ms = CHAMPION_STALE_DAYS * 86_400_000
        survivors = [c for c in champs
                     if c.get("id") in protected
                     or not (c.get("uses", 0) == 0 and (now - c.get("born_ts", now)) > stale_ms)]
        # protected first, then by current fitness; cap. (Protected always fit:
        # PROTECT_USED + 1 active << KEEP.)
        survivors.sort(key=lambda x: (x.get("id") in protected, x.get("fitness", 0.0)), reverse=True)
        self.champions = survivors[:CHAMPIONS_KEEP]
        self.save_champions()

    def champion_live_stats(self) -> dict:
        """Aggregate REAL executed trades from the journal by the champion that was
        active when each was taken -> {id: {trades, wins, pnl, win_rate}}. This is a
        champion's actual live/paper track record, not its backtest score."""
        out: dict[str, dict] = {}
        for r in self.journal.rows:
            cid = r.get("champion_id")
            if not cid:
                continue
            g = out.setdefault(cid, {"trades": 0, "wins": 0, "pnl": 0.0})
            g["trades"] += 1
            g["wins"] += 1 if r.get("pnl", 0.0) > 0 else 0
            g["pnl"] += float(r.get("pnl", 0.0))
        for g in out.values():
            g["pnl"] = round(g["pnl"], 4)
            g["win_rate"] = round(g["wins"] / g["trades"], 3) if g["trades"] else 0.0
        return out

    def save_champions(self) -> None:
        try:
            CHAMPIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
            CHAMPIONS_PATH.write_text(json.dumps(self.champions, indent=2))
        except OSError as e:
            log.warning("could not save champions: %s", e)

    # ------------------------------------------------- divergence & alerts

    def _divergence(self) -> dict | None:
        """Compare live realized performance with the backtest expectation (the
        best champion's stats). Flags when live materially underperforms — the
        early-warning that the edge has stopped working, not just a bad day."""
        if self.engine is None:
            return None
        live = self.engine.portfolio.stats()
        d: dict = {"live_win_rate": live.get("win_rate", 0.0),
                   "live_profit_factor": live.get("profit_factor", 0.0),
                   "live_trades": live.get("trades", 0)}
        # reference = what's actually trading (active champion), else the best
        # current-fitness set in the vault — not just champions[0], which after
        # pruning is ordered most-used-first, not best-first.
        ref = self.find_champion(self.active_champion_id)
        if ref is None and self.champions:
            ref = max(self.champions, key=lambda c: c.get("fitness", 0.0))
        if ref:
            d["expected_win_rate"] = ref.get("win_rate", 0.0)
            d["expected_profit_factor"] = ref.get("profit_factor", 0.0)
            d["win_rate_gap"] = round(live.get("win_rate", 0.0) - ref.get("win_rate", 0.0), 3)
        if live.get("trades", 0) < 12:
            d["status"] = "gathering"
            d["diverged"] = False
        else:
            d["status"] = "tracking"
            d["diverged"] = bool(live.get("profit_factor", 0.0) < 0.9
                                 or (ref and d.get("win_rate_gap", 0.0) < -0.15))
        return d

    async def _alert_loop(self) -> None:
        while self.engine is not None:
            try:
                await self.alerter.check(self.mode, self.engine, self._divergence())
            except Exception as e:  # noqa: BLE001
                log.debug("alert check failed: %s", e)
            await asyncio.sleep(30)

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

        self.engine = TraderEngine(self.cfg, feed, broker, portfolio, risk, self.specs,
                                   on_update, journal=self.journal)
        self.engine.active_champion_id = self.active_champion_id
        await self.engine.start()

        from ..engine.autotuner import AutoTuner
        self.autotuner = AutoTuner(self)
        self.autotuner.start()
        from ..engine.carry import CarryDesk
        from ..engine.scanner import MarketScanner
        self.scanner = MarketScanner(self)
        self.scanner.start()
        self.carry = CarryDesk(self)
        self.carry.start()
        self._alert_task = asyncio.create_task(self._alert_loop(), name="alert-loop")

    async def _stop_engine(self) -> None:
        if self._alert_task is not None:
            self._alert_task.cancel()
            try:
                await self._alert_task
            except (asyncio.CancelledError, Exception):
                pass
            self._alert_task = None
        if self.carry is not None:
            await self.carry.stop()
            self.carry = None
        if self.scanner is not None:
            await self.scanner.stop()
            self.scanner = None
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
        for p in (self.pool, self.research_pool):
            if p is not None:
                p.shutdown(wait=False, cancel_futures=True)

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
        symbol = _clean_symbol(symbol) or "BTC-USDT"   # one contract only
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
        symbol = _clean_symbol(symbol) or "BTC-USDT"   # one contract only
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

    def start_walkforward(self, symbol: str, interval: str, days: float,
                          folds: int = 5, trials: int = 20, synthetic: bool = False) -> Job:
        symbol = _clean_symbol(symbol) or "BTC-USDT"
        job = Job("walkforward")
        self.jobs[job.id] = job

        async def runner() -> None:
            try:
                candles = await self._get_backtest_candles(symbol, interval, days, synthetic, job)
                if len(candles) < folds * 1200:
                    raise ValueError(f"walk-forward needs ~{folds * 1200}+ bars, got {len(candles)}")
                job.progress = 0.4
                await self._notify("job")
                spec = self.specs.get(symbol, ContractSpec(symbol))
                result = await self.run_cpu(
                    run_walkforward, candles, symbol, interval,
                    self.cfg.strategy, self.cfg.risk, spec,
                    self.cfg.paper.starting_balance, spec.taker_fee,
                    self.cfg.paper.slippage_bps, folds, trials, None,
                )
                job.result = result
                job.progress = 1.0
            except Exception as e:  # noqa: BLE001
                log.exception("walk-forward job failed")
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
        syms = _split_symbols(symbols)

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
            live = self.champion_live_stats()
            top_used = {c["id"] for c in sorted(self.champions, key=lambda x: x.get("uses", 0),
                        reverse=True)[:10] if c.get("uses", 0) > 0}
            enriched = []
            for c in self.champions[:CHAMPIONS_KEEP]:
                e = dict(c)
                e["live"] = live.get(c.get("id"), {"trades": 0, "wins": 0, "pnl": 0.0, "win_rate": 0.0})
                e["top_used"] = c.get("id") in top_used
                e["active"] = c.get("id") == self.active_champion_id
                enriched.append(e)
            d["champions"] = enriched
            d["active_champion_id"] = self.active_champion_id
        if self.engine is not None:
            d["divergence"] = self._divergence()
            d["alerts_on"] = self.alerter.enabled
        if self.scanner is not None:
            d["radar"] = self.scanner.snapshot()
        if self.carry is not None:
            d["carry"] = self.carry.snapshot()
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
