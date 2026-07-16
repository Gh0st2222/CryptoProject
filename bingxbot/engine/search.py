"""Smart auto-tuner search.

Two ideas make the research desk fast and sample-efficient instead of the old
random-restart hill-climbing:

1. **Parallel fold scoring with indicator reuse.** `score_fold` builds the whole
   FeatureFrame (52 indicators) ONCE for a fold and reuses it for every candidate
   — indicators depend on price, not on the parameters being tuned. One fold is
   one process-pool task, so a cycle scores every candidate on every fold across
   many cores at once.

2. **Persistent Differential Evolution.** `DEOptimizer` keeps a population that
   evolves across cycles and survives restarts (saved to disk). DE's
   difference-vector mutation adapts its step size to the landscape and keeps
   exploring, so it converges toward good regions far faster than gaussian jitter
   around a single champion.
"""
from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

from ..config import ROOT, RiskConfig, StrategyConfig
from ..util import clamp
from .backtest import (TUNABLES, _apply_params, _coerce, _fitness,
                       candles_to_arrays, run_backtest)
from ..strategy.features import FeatureFrame

STATE_PATH = ROOT / "data_cache" / "tuner_state.json"


# --------------------------------------------------- parallel fold scoring

def score_fold(fold_candles, symbol, interval, spec, taker, slip,
               base_strat: StrategyConfig, base_risk: RiskConfig, param_list) -> list[float]:
    """Score every param-set in `param_list` on ONE fold, building the fold's
    FeatureFrame once and reusing it for all of them. Module-level + picklable so
    it runs in a research-pool worker; the caller runs one of these per fold in
    parallel."""
    if len(fold_candles) < 360:
        return [-1.0] * len(param_list)
    ff = FeatureFrame(candles_to_arrays(fold_candles), interval=interval)
    out = []
    for p in param_list:
        s, r = _apply_params(base_strat, base_risk, p)
        res = run_backtest(fold_candles, symbol, interval, s, r, spec, taker_fee=taker,
                           slippage_bps=slip, collect_series=False, ff=ff)
        out.append(_fitness(res.get("stats", {})) if "error" not in res else -1.0)
    return out


def validate_params(params, valid_candles, symbol, interval, spec, taker, slip,
                    base_strat: StrategyConfig, base_risk: RiskConfig) -> dict:
    """Run one param-set on the held-out RECENT window (out-of-sample — the DE
    never trained on it) and return its fitness + stats. This is the promotion
    gate: a champion has to prove itself on the data closest to live, not on the
    window it was fitted to."""
    s, r = _apply_params(base_strat, base_risk, params)
    ff = FeatureFrame(candles_to_arrays(valid_candles), interval=interval)
    res = run_backtest(valid_candles, symbol, interval, s, r, spec, taker_fee=taker,
                       slippage_bps=slip, collect_series=False, ff=ff)
    st = res.get("stats", {})
    return {"fitness": _fitness(st) if "error" not in res else -1.0, "stats": st}


def robust_aggregate(fold_fits: list[float], weights: list[float] | None = None) -> float:
    """Combine a candidate's per-fold fitnesses into one robust score: a
    recency-weighted mean, penalized for instability and for the worst fold, so
    params that only print in one window score poorly. This is the anti-overfit
    core of what makes a good champion."""
    if not fold_fits:
        return -1.0
    w = weights if weights and len(weights) == len(fold_fits) else [1.0] * len(fold_fits)
    wmean = sum(f * wi for f, wi in zip(fold_fits, w)) / (sum(w) or 1.0)
    sd = statistics.pstdev(fold_fits) if len(fold_fits) > 1 else 0.0
    worst = min(fold_fits)
    return wmean - 0.3 * sd + 0.2 * worst


def recency_weights(n: int) -> list[float]:
    """Linear ramp giving the most recent fold ~2x the oldest fold's weight."""
    if n <= 1:
        return [1.0] * max(n, 1)
    return [1.0 + 1.0 * i / (n - 1) for i in range(n)]


# ----------------------------------------------- Differential Evolution

class DEOptimizer:
    def __init__(self, pop_size: int = 28, f: float = 0.6, cr: float = 0.85,
                 seed: int | None = None, state_path: Path = STATE_PATH):
        self.keys = list(TUNABLES)
        self.bounds = {k: (TUNABLES[k][0], TUNABLES[k][1]) for k in self.keys}
        self.pop_size = pop_size
        self.f = f
        self.cr = cr
        self.rng = random.Random(seed)
        self.state_path = state_path
        self.pop: list[dict] = []
        self.fitness: list[float] = []
        self.generation = 0

    # -- lifecycle -------------------------------------------------------
    def _rand_vec(self) -> dict:
        return {k: _coerce(k, self.rng.uniform(*self.bounds[k])) for k in self.keys}

    def _coerce_vec(self, p: dict) -> dict:
        return {k: _coerce(k, clamp(float(p.get(k, sum(self.bounds[k]) / 2)), *self.bounds[k]))
                for k in self.keys}

    def seed_population(self, champion: dict | None = None) -> None:
        self.pop = [self._coerce_vec(champion)] if champion else []
        while len(self.pop) < self.pop_size:
            self.pop.append(self._rand_vec())
        self.fitness = [-1e9] * len(self.pop)
        self.generation = 0

    def ready(self) -> bool:
        return len(self.pop) >= 4

    def inject(self, params: dict) -> None:
        """Make sure a known-good set (e.g. the live champion) is in the gene pool
        by replacing the current worst member if it isn't already present."""
        if not self.pop or not params:
            return
        vec = self._coerce_vec(params)
        if any(all(abs(m.get(k, 0) - vec[k]) < 1e-9 for k in self.keys) for m in self.pop):
            return
        worst = min(range(len(self.pop)), key=lambda j: self.fitness[j])
        self.pop[worst] = vec
        self.fitness[worst] = -1e9

    # -- one generation -------------------------------------------------
    def trials(self) -> list[dict]:
        """rand/1/bin: for each member, trial = a + F*(b-c) crossed with the
        member (at least one gene forced from the mutant)."""
        n = len(self.pop)
        out = []
        for i in range(n):
            pool = [j for j in range(n) if j != i]
            if len(pool) >= 3:
                ia, ib, ic = self.rng.sample(pool, 3)
                a, b, c = self.pop[ia], self.pop[ib], self.pop[ic]
            else:
                a = b = c = self.pop[i]
            jrand = self.rng.randrange(len(self.keys))
            trial = {}
            for ki, k in enumerate(self.keys):
                lo, hi = self.bounds[k]
                if self.rng.random() < self.cr or ki == jrand:
                    v = a[k] + self.f * (b[k] - c[k])
                else:
                    v = self.pop[i][k]
                trial[k] = _coerce(k, clamp(float(v), lo, hi))
            out.append(trial)
        return out

    def select(self, trials: list[dict], trial_fit: list[float],
               member_fit: list[float]) -> None:
        """Greedy selection on the SAME folds: a trial replaces its parent iff it
        scores at least as high; members keep their freshly measured fitness."""
        for i in range(len(self.pop)):
            self.fitness[i] = member_fit[i]
            if i < len(trials) and trial_fit[i] >= self.fitness[i]:
                self.pop[i] = trials[i]
                self.fitness[i] = trial_fit[i]
        self.generation += 1

    def best(self) -> tuple[dict, float]:
        if not self.pop:
            return {}, -1e9
        i = max(range(len(self.pop)), key=lambda j: self.fitness[j])
        return dict(self.pop[i]), self.fitness[i]

    def diversity(self) -> float:
        """Mean normalized spread across genes — a health signal (near 0 = the
        population has collapsed and should be re-seeded)."""
        if len(self.pop) < 2:
            return 0.0
        spreads = []
        for k in self.keys:
            lo, hi = self.bounds[k]
            span = (hi - lo) or 1.0
            vals = [m[k] for m in self.pop]
            spreads.append((max(vals) - min(vals)) / span)
        return sum(spreads) / len(spreads)

    # -- persistence -----------------------------------------------------
    def save(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps({
                "generation": self.generation, "keys": self.keys,
                "pop": self.pop, "fitness": self.fitness,
            }))
        except OSError:
            pass

    def load(self) -> bool:
        try:
            d = json.loads(self.state_path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        if set(d.get("keys", [])) != set(self.keys) or not d.get("pop"):
            return False   # tunable set changed between builds -> start fresh
        self.pop = [self._coerce_vec(p) for p in d["pop"]]
        self.fitness = [float(x) for x in d.get("fitness", [])] or [-1e9] * len(self.pop)
        if len(self.fitness) != len(self.pop):
            self.fitness = [-1e9] * len(self.pop)
        self.generation = int(d.get("generation", 0))
        self.pop_size = len(self.pop)
        return True
