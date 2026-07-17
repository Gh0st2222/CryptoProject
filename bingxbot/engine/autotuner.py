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
TOP_K_VALIDATE = 5          # validate this many training-best members OOS, keep the best generalizer
VAULT_CANDIDATES = 4        # also re-validate this many top vault champions each cycle (candidate pool, not graveyard)
STALL_REINJECT = 20         # cycles without a promotion before a diversity restart
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
        self._since_improve = 0
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

        # Evaluate EVERY live candidate on the SAME current OOS window in one
        # parallel batch and run whichever wins: a freshly-evolved DE member, a
        # champion pulled back out of the vault that STILL fits today's market, or
        # the incumbent. The vault is a candidate pool, not a graveyard — the best
        # available champion drives trading, wherever it came from.
        topk = self.de.top_k(TOP_K_VALIDATE)          # (params, train_fit)
        vault = sorted(self.orch.champions, key=lambda c: c.get("fitness", 0.0),
                       reverse=True)[:VAULT_CANDIDATES]
        cands: list[dict] = [{"source": "de", "params": p, "train_fit": tfit, "cid": None}
                             for p, tfit in topk]
        cands += [{"source": "vault", "params": c.get("params", {}), "train_fit": None, "cid": c.get("id")}
                  for c in vault]

        val_args = [(c["params"], valid, symbol, interval, spec, taker, slip, strat, risk) for c in cands]
        val_args.append((champ, valid, symbol, interval, spec, taker, slip, strat, risk))
        val_res = await self.orch.map_cpu(validate_params, val_args, research=True)
        champ_fit = val_res[-1]["fitness"]

        best, best_adj, best_stats = None, -1e18, {}
        for c, res in zip(cands, val_res[:-1]):
            oos = res["fitness"]
            # DE members carry an overfit penalty (in-sample -> OOS drop); vault
            # champions are scored raw — they've already proven out-of-sample.
            adj = oos - (OVERFIT_LAMBDA * max(0.0, c["train_fit"] - oos) if c["train_fit"] is not None else 0.0)
            if c["source"] == "vault" and c["cid"]:
                self.orch.set_champion_current(c["cid"], oos, res["stats"])  # keep its CURRENT eval fresh
            if adj > best_adj:
                best, best_adj, best_stats = c, adj, res["stats"]

        self.cycles += 1
        self.champion_fitness = round(champ_fit, 3)
        promoted = False
        best_params = best["params"] if best else {}
        different = bool(best) and any(abs(best_params.get(k, 0) - champ.get(k, 0)) > 1e-9 for k in champ)
        if different and best_adj > max(champ_fit * IMPROVE_MARGIN, MIN_ABS_FITNESS):
            self.orch.apply_params(best_params)
            self.improvements += 1
            promoted = True
            vs = best_stats
            # tag the champion now driving live trades: reuse the vault entry if
            # the winner came from the vault, otherwise mint a new one.
            cid = best["cid"] if (best["source"] == "vault" and best["cid"]) \
                else self.orch.record_champion(best_params, best_adj, vs)
            self.orch.mark_champion_used(cid)
            self.history.append({
                "ts": int(time.time() * 1000),
                "from_fitness": round(champ_fit, 3), "to_fitness": round(best_adj, 3),
                "valid_wr": round(vs.get("win_rate", 0), 3), "valid_pf": round(vs.get("profit_factor", 0), 3),
                "gen": self.de.generation, "source": best["source"], "champion_id": cid,
                "params": {k: best_params[k] for k in ("base_threshold", "risk_per_trade", "sl_atr_min",
                           "trail_atr_max", "giveback_rr", "target_trades_per_hour") if k in best_params},
            })
            self.history = self.history[-25:]
            log.info("auto-tune PROMOTED (gen %d, %s): OOS %.2f -> %.2f",
                     self.de.generation, best["source"], champ_fit, best_adj)
        else:
            self.orch.save_champions()   # persist the refreshed vault current-evals

        # diversity restart: if the population has converged without finding a
        # champion for a long time, it's stuck in an overfit basin — re-inject
        # fresh explorers so it keeps searching instead of grinding the same region.
        self._since_improve = 0 if promoted else self._since_improve + 1
        if self.de.diversity() < 0.25 and self._since_improve >= STALL_REINJECT:
            k = self.de.reinject(0.4)
            self._since_improve = 0
            log.info("auto-tune: converged without a champion -> re-injected %d explorers", k)

        if self.cycles % VAULT_REVAL_EVERY == 0:
            await self._revalidate_vault(valid, symbol, interval, spec, taker, slip, strat, risk)

        self.last_cycle = {
            "ts": int(time.time() * 1000), "symbol": symbol,
            "generation": self.de.generation, "population": len(self.de.pop),
            "diversity": round(self.de.diversity(), 3), "folds": len(fold_fits),
            "research_cores": self.orch.research_workers,
            "champion_fitness": round(champ_fit, 3), "best_fitness": round(best_adj, 3),
            "promoted": promoted, "candidates": len(candidates),
            "vault_candidates": len(vault), "de_candidates": len(topk),
            "champion_source": (best["source"] if best else None),
        }
        if self.orch._notify:
            await self.orch._notify("autotune")
        return promoted

    async def _revalidate_vault(self, valid, symbol, interval, spec, taker, slip, strat, risk) -> None:
        """Re-score every saved champion on the freshest window and refresh its
        CURRENT evaluation (shown in the vault next to what it was born at). We
        DON'T drop the temporarily-cold — pruning ages out the never-used, and
        protects the most-used, so a proven champion having a bad week survives."""
        vault = self.orch.champions
        if not vault:
            return
        results = await self.orch.map_cpu(
            validate_params,
            [(c.get("params", {}), valid, symbol, interval, spec, taker, slip, strat, risk) for c in vault],
            research=True)
        for c, res in zip(vault, results):
            self.orch.set_champion_current(
                c["id"], res.get("fitness", c.get("fitness", 0.0)), res.get("stats", {}))
        self.orch.prune_champions()
        log.info("vault revalidated: %d champions re-scored on recent data", len(self.orch.champions))

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
