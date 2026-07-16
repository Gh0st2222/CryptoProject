"""Continuous auto-tuner — the firm's always-on research desk, now a real
optimizer instead of random-restart hill-climbing.

Each cycle:
  1. A **persistent Differential Evolution** population proposes trials over the
     full tunable space (it remembers what worked across cycles and restarts).
  2. Every member + trial is scored across several **time folds in parallel** on
     a dedicated **research pool** (one fold per core), building each fold's
     indicators once and reusing them for all candidates — so a cycle that used
     to pin one core now spreads across many and finishes far sooner.
  3. Candidates are ranked by a **risk-adjusted, recency-weighted, robust** score
     (rewards profitable frequency, punishes instability), then the population's
     best is **validated out-of-sample** on the most recent held-out window with
     an overfit penalty, and promoted into the live brains only if it clearly
     beats the running champion there.
  4. Every so often the champion vault is **re-validated on fresh data** and
     stale sets are retired.

It never touches user-owned settings.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time

from ..config import MODE_IDLE
from ..exchange.models import ContractSpec
from ..util import clamp
from .backtest import TUNABLES
from .search import (DEOptimizer, recency_weights, robust_aggregate, score_fold,
                     validate_params)

log = logging.getLogger("autotuner")

POP_SIZE = 28
IMPROVE_MARGIN = 1.06       # OOS challenger must beat champion OOS x this
MIN_ABS_FITNESS = 0.3       # ...and be clearly profitable (positive risk-adjusted score)
OVERFIT_LAMBDA = 0.5        # penalty weight on the in-sample -> OOS drop
LOOKBACK_DAYS = 60.0
DATA_TTL_S = 1800
GAP_FAST = 20               # cadence right after a promotion (keep hammering)
GAP_SLOW = 60               # cadence when stable
VAULT_REVAL_EVERY = 15      # re-validate the champion vault every N cycles
MIN_BARS = 3000


def _current_params(cfg) -> dict:
    p = {}
    for name, (_lo, _hi, grp, _kind) in TUNABLES.items():
        src = cfg.strategy if grp == "strategy" else cfg.risk
        p[name] = getattr(src, name)
    return p


def _make_folds(candles: list, f: int) -> list[list]:
    n = len(candles)
    size = max(1, n // f)
    return [candles[i * size: (n if i == f - 1 else (i + 1) * size)] for i in range(f)]


class AutoTuner:
    def __init__(self, orch):
        self.orch = orch
        self._task: asyncio.Task | None = None
        self.running = False
        self.rng = random.Random()
        self.de = DEOptimizer(pop_size=POP_SIZE, seed=self.rng.randint(0, 2**31))
        self._candles: list = []
        self._data_ts = 0.0
        self._scored_ts = -1.0      # data window the population was last fully scored on
        self.cycles = 0
        self.improvements = 0
        self.next_run_ts = 0.0
        self.champion_fitness = 0.0
        self.last_cycle: dict | None = None
        self.history: list[dict] = []

    def start(self) -> None:
        if self._task is None or self._task.done():
            self.running = True
            self._task = asyncio.create_task(self._loop(), name="autotuner")

    async def stop(self) -> None:
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def _loop(self) -> None:
        await asyncio.sleep(20)
        while self.running:
            cfg = self.orch.cfg
            promoted = False
            if cfg.strategy.auto_tune and self.orch.mode != MODE_IDLE and self.orch.engine is not None:
                try:
                    promoted = await self._cycle()
                except Exception as e:  # noqa: BLE001
                    log.warning("auto-tune cycle failed: %s", e)
            gap = GAP_FAST if promoted else GAP_SLOW
            self.next_run_ts = time.time() + gap
            await asyncio.sleep(gap)

    async def _ensure_data(self):
        if self._candles and time.time() - self._data_ts < DATA_TTL_S:
            return self._candles
        cfg = self.orch.cfg
        symbol, interval = cfg.symbols[0], cfg.strategy.interval
        synthetic = cfg.feed == "synthetic"
        self._candles = await self.orch._get_backtest_candles(symbol, interval, LOOKBACK_DAYS, synthetic, _NullJob())
        self._data_ts = time.time()
        return self._candles

    async def _cycle(self) -> bool:
        cfg = self.orch.cfg
        symbol, interval = cfg.symbols[0], cfg.strategy.interval
        candles = await self._ensure_data()
        if len(candles) < MIN_BARS:
            return False
        spec = self.orch.specs.get(symbol, ContractSpec(symbol))
        taker, slip = spec.taker_fee, cfg.paper.slippage_bps
        strat, risk = cfg.strategy, cfg.risk

        n = len(candles)
        val_cut = int(n * 0.75)
        train = candles[:val_cut]
        valid = candles[max(0, val_cut - 400):]            # recent held-out (+ warmup lead-in)

        champ = _current_params(cfg)
        if not self.de.ready():
            if not self.de.load():
                self.de.seed_population(champ)
        self.de.inject(champ)

        # folds scale with the research pool: more cores -> more (finer) folds,
        # one fold per worker, indicators built once per fold.
        nf = int(clamp(self.orch.research_workers, 3, 8))
        folds = _make_folds(train, nf)
        trials = self.de.trials()
        # Only re-score the whole population when the data window changed (every
        # ~30 min) or a member is unscored (freshly injected); otherwise member
        # fitness carries forward on the same folds and we score just the trials —
        # halving the work and roughly doubling generations-per-hour in steady state.
        need_members = (self._scored_ts != self._data_ts) or any(f <= -1e8 for f in self.de.fitness)
        self._scored_ts = self._data_ts
        candidates = (list(self.de.pop) + trials) if need_members else list(trials)
        args = [(fold, symbol, interval, spec, taker, slip, strat, risk, candidates) for fold in folds]
        fold_fits = await self.orch.map_cpu(score_fold, args, research=True)
        fold_fits = [ff for ff in fold_fits if ff and len(ff) == len(candidates)]
        if not fold_fits:
            return False
        w = recency_weights(len(fold_fits))
        robust = [robust_aggregate(list(fc), w) for fc in zip(*fold_fits)]
        p = len(self.de.pop)
        if need_members:
            member_fit, trial_fit = robust[:p], robust[p:p + len(trials)]
        else:
            member_fit, trial_fit = list(self.de.fitness), robust[:len(trials)]
        self.de.select(trials, trial_fit, member_fit)
        self.de.save()

        best_params, best_train = self.de.best()

        # out-of-sample validation of the challenger and the current champion
        best_oos = await self.orch.run_cpu(validate_params, best_params, valid, symbol, interval,
                                           spec, taker, slip, strat, risk, research=True)
        champ_oos = await self.orch.run_cpu(validate_params, champ, valid, symbol, interval,
                                            spec, taker, slip, strat, risk, research=True)
        oos_fit = best_oos["fitness"]
        # overfit penalty: dock the challenger for any drop from train to OOS
        oos_adj = oos_fit - OVERFIT_LAMBDA * max(0.0, best_train - oos_fit)
        champ_fit = champ_oos["fitness"]

        self.cycles += 1
        self.champion_fitness = round(champ_fit, 3)
        promoted = False
        different = any(abs(best_params.get(k, 0) - champ.get(k, 0)) > 1e-9 for k in champ)
        if different and oos_adj > max(champ_fit * IMPROVE_MARGIN, MIN_ABS_FITNESS):
            self.orch.apply_params(best_params)
            self.improvements += 1
            promoted = True
            vs = best_oos["stats"]
            self.history.append({
                "ts": int(time.time() * 1000),
                "from_fitness": round(champ_fit, 3), "to_fitness": round(oos_adj, 3),
                "valid_wr": round(vs.get("win_rate", 0), 3), "valid_pf": round(vs.get("profit_factor", 0), 3),
                "gen": self.de.generation,
                "params": {k: best_params[k] for k in ("base_threshold", "risk_per_trade", "sl_atr_min",
                           "trail_atr_max", "giveback_rr", "target_trades_per_hour") if k in best_params},
            })
            self.history = self.history[-25:]
            self.orch.record_champion(best_params, oos_adj, vs)
            log.info("auto-tune PROMOTED (gen %d): OOS %.2f -> %.2f", self.de.generation, champ_fit, oos_adj)

        if self.cycles % VAULT_REVAL_EVERY == 0:
            await self._revalidate_vault(valid, symbol, interval, spec, taker, slip, strat, risk)

        self.last_cycle = {
            "ts": int(time.time() * 1000), "symbol": symbol,
            "generation": self.de.generation, "population": len(self.de.pop),
            "diversity": round(self.de.diversity(), 3), "folds": len(fold_fits),
            "research_cores": self.orch.research_workers,
            "champion_fitness": round(champ_fit, 3), "best_fitness": round(oos_adj, 3),
            "promoted": promoted, "candidates": len(candidates),
        }
        if self.orch._notify:
            await self.orch._notify("autotune")
        return promoted

    async def _revalidate_vault(self, valid, symbol, interval, spec, taker, slip, strat, risk) -> None:
        """Re-score every saved champion on the freshest window and retire the
        ones that no longer hold up — the vault tracks what still works, not what
        once did."""
        vault = self.orch.champions
        if not vault:
            return
        results = await self.orch.map_cpu(
            validate_params,
            [(c.get("params", {}), valid, symbol, interval, spec, taker, slip, strat, risk) for c in vault],
            research=True)
        kept = []
        for c, res in zip(vault, results):
            fit = res.get("fitness", -1.0)
            if fit > 0:
                c["fitness"] = round(fit, 3)
                c["win_rate"] = round(res.get("stats", {}).get("win_rate", c.get("win_rate", 0.0)), 4)
                c["profit_factor"] = round(res.get("stats", {}).get("profit_factor", c.get("profit_factor", 0.0)), 3)
                kept.append(c)
        kept.sort(key=lambda x: x["fitness"], reverse=True)
        self.orch.champions = kept
        self.orch.save_champions()
        log.info("vault revalidated: kept %d/%d on recent data", len(kept), len(vault))

    def snapshot(self) -> dict:
        return {
            "enabled": self.orch.cfg.strategy.auto_tune,
            "running": self.running,
            "cycles": self.cycles,
            "improvements": self.improvements,
            "champion_fitness": self.champion_fitness,
            "generation": self.de.generation,
            "population": len(self.de.pop),
            "research_cores": self.orch.research_workers,
            "next_run_ts": int(self.next_run_ts * 1000),
            "last_cycle": self.last_cycle,
            "history": self.history[-12:][::-1],
        }


class _NullJob:
    progress = 0.0
