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
import logging
import random
import statistics
from pathlib import Path

from ..config import ROOT, RiskConfig, StrategyConfig
from ..util import atomic_write, clamp
from .backtest import (TUNABLES, _apply_params, _coerce, _fitness,
                       candles_to_arrays, run_backtest, run_portfolio_backtest)
from ..strategy.features import FeatureFrame

log = logging.getLogger("search")

OBJECTIVE_VER = 2   # bump when fold_composite changes: saved scores are
                    # then on a different scale and must not be reused
STATE_PATH = ROOT / "data_cache" / "tuner_state.json"

# Worker-side cache of kernel-prepared folds (feature matrix, alpha matrix,
# regime codes — ~0.5MB per 1k-bar fold). The research pool's processes are
# persistent and the tuner re-scores the SAME data windows for ~30 minutes at
# a time; without this every cycle rebuilt identical frames, 19 alpha series
# and per-bar regime codes just to hand the kernel the same matrices.
_PREP_CACHE: dict[tuple, tuple] = {}
_PREP_MAX = 16


def _fold_key(fold_candles, interval: str) -> tuple:
    c0, cn = fold_candles[0], fold_candles[-1]
    return (interval, len(fold_candles), int(c0.ts), int(cn.ts),
            round(float(c0.open), 8), round(float(cn.close), 8))


# --------------------------------------------------- parallel fold scoring

def score_fold(fold_candles, symbol, interval, spec, taker, slip,
               base_strat: StrategyConfig, base_risk: RiskConfig, param_list) -> list[float]:
    """Score every param-set in `param_list` on ONE fold, building the fold's
    FeatureFrame once and reusing it for all of them. Module-level + picklable so
    it runs in a research-pool worker; the caller runs one of these per fold in
    parallel.

    When numba is available, candidates run on the COMPILED KERNEL — a
    parity-tested nopython port of the whole engine (~15-20x per candidate).
    The kernel ranks TRAINING candidates without the meta-labeling head
    (sklearn can't run in nopython code); OOS validation and promotion always
    use the full Python engine, meta included — the search proposes fast, the
    judge stays full-fidelity. Set BOT_NO_KERNEL=1 to force the Python path."""
    if len(fold_candles) < 360:
        return [-1.0] * len(param_list)
    import os
    if os.getenv("BOT_NO_KERNEL", "") != "1":
        try:
            from .kernel import kernel_fitness_prepped, prep_fold
            key = _fold_key(fold_candles, interval)
            prep = _PREP_CACHE.get(key)
            if prep is None:
                # transient frame: only the compact matrices are kept
                prep = prep_fold(FeatureFrame(candles_to_arrays(fold_candles),
                                              interval=interval))
                if len(_PREP_CACHE) >= _PREP_MAX:
                    _PREP_CACHE.pop(next(iter(_PREP_CACHE)))
                _PREP_CACHE[key] = prep
            out = []
            for p in param_list:
                s, r = _apply_params(base_strat, base_risk, p)
                st = kernel_fitness_prepped(prep, s, r, spec, taker, slip, interval)["stats"]
                out.append(_fitness(st))
            return out
        except Exception:  # noqa: BLE001 — the kernel is an optimization, never a dependency
            pass
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


def validate_params_portfolio(params, candles_by_symbol: dict, interval, specs: dict,
                              taker, slip, base_strat: StrategyConfig,
                              base_risk: RiskConfig, use_meta: bool = True) -> dict:
    """Score one param-set the way the ACCOUNT actually experiences it: a
    shared-portfolio backtest over the traded basket's window — one equity
    pool, one position cap, correlation haircut, one kill switch. This is the
    promotion gate's unit of evidence; a single-symbol run can flatter a set
    that the portfolio (which is what compounds) would reject.

    `use_meta=False` is for the evidence lanes that must judge the CORE brain
    only: the alt-clock trial and the regime gauntlet. The meta head is
    trained on the LIVE clock's recent bars — blending it (at its full
    measured weight) into a 5m validation or a 2021 era measures the model's
    clock/era mismatch, not the parameter set. Primary promotion keeps
    use_meta=True: live trades WITH the meta on THIS clock, so the judge must
    too (parity)."""
    s, r = _apply_params(base_strat, base_risk, params)
    res = run_portfolio_backtest(candles_by_symbol, interval, s, r, specs,
                                 taker_fee=taker, slippage_bps=slip, warmup=300,
                                 use_meta=use_meta)
    if "error" in res:
        return {"fitness": -1.0, "stats": {}}
    st = res.get("stats", {})
    return {"fitness": _fitness(st), "stats": st}


def portfolio_folds(cbs: dict[str, list], k: int = 3, tail_frac: float = 0.40,
                    warmup: int = 300) -> list[dict[str, list]]:
    """Sequential purged OOS folds over the basket's most recent `tail_frac`:
    each fold's TRADED region is disjoint (the warmup lead-in overlaps earlier
    data as indicator warmup only, never as traded bars), so one lucky window
    can't promote a champion — it must earn across all of them."""
    folds: list[dict[str, list]] = []
    for j in range(k):
        fc: dict[str, list] = {}
        for sym, cs in cbs.items():
            n = len(cs)
            a = 1.0 - tail_frac + j * tail_frac / k
            b = 1.0 - tail_frac + (j + 1) * tail_frac / k
            lo = int(n * a)
            hi = n if j == k - 1 else int(n * b)
            fc[sym] = cs[max(0, lo - warmup):hi]
        if fc and all(len(v) >= warmup + 60 for v in fc.values()):
            folds.append(fc)
    return folds


def fold_composite(fold_fits: list[float]) -> float:
    """THE one way this project turns per-fold fitnesses into a single score.

    A blend of the TYPICAL fold (median) and the WORST one. The median demands
    profit in the ordinary window rather than on average, so a single parabolic
    fold cannot buy a seat; the worst-fold term keeps the tail priced in.

    Both the SEARCH and the JUDGE call this, and they used to disagree. The
    search maximized a recency-weighted mean penalized by standard deviation
    while promotion blended median with worst -- two different functions over
    the same folds, ranking the same candidates differently. That is not a
    detail: it is why generations of climbing the training objective walked away
    from what the judge rewards.

    Measured on one 56-member population, scored once and re-aggregated both
    ways so only the objective differs:

        aggregator                  rho vs OOS return   top-10 median OOS
        weighted mean - sd (was)         +0.233               +2.72%
        this blend                       +0.338               +7.72%

    ~45% more agreement with the judge and nearly three times the out-of-sample
    quality of what the search nominates, for one function call and no extra
    CPU. There is no leak: the two still run on different windows. Sharing the
    definition is what stops them drifting apart again.

    (An evidence discount on the LOSING branch of _fitness was measured at the
    same time and REJECTED -- it dropped rho to +0.145 and took the nominated
    top-10 negative. The asymmetry between winners and losers is load-bearing.)"""
    if not fold_fits:
        return -1.0
    return 0.7 * statistics.median(fold_fits) + 0.3 * min(fold_fits)


def robust_aggregate(fold_fits: list[float], weights: list[float] | None = None) -> float:
    """The search's per-candidate score across its training folds.

    `weights` is accepted and deliberately ignored: a recency ramp is a third
    way of weighting folds, and the measurement above preferred none of them.
    The parameter stays in the signature so an un-updated caller cannot silently
    pass its weights into some other argument."""
    return fold_composite(fold_fits)


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
        """rand/1/bin with F-dither: for each member, trial = a + F*(b-c)
        crossed with the member (at least one gene forced from the mutant).
        F is drawn fresh per trial from [0.4, 0.9] — standard dither, which
        keeps both large exploratory and small refining steps in play instead
        of one fixed step size for the whole run.

        SELF-ADAPTATION WAS MEASURED AND IS NOT WORTH IT. jDE (Brest et al.
        2006) — every member carrying its own F and CR, re-rolled with
        probability tau and adopted only when the trial wins — was raced
        against this over 12 paired seeds, same folds, same 40-generation
        budget, judged on the out-of-sample return of what each run nominates:

            best training score          7/12 wins   median delta +0.153
            OOS of the single best       7/12 wins   median delta +0.017
            OOS median of the top 3      6/12 wins   median delta -0.001
            population diversity         6/12 wins   median delta -0.015

        A coin flip on every measure. An earlier 4-seed run looked like a clear
        jDE win (+6.12% vs +1.01% median OOS) and did not survive more seeds —
        the seed dominates the variance, so anything less than a paired design
        with double digits of them will just report noise.

        The lesson generalizes: when the search underperformed it was never the
        mutation operator, it was the objective being climbed (fold_composite)
        and the folds it was measured on (_train_fold_count). Fix what the
        search is aiming at before reaching for a cleverer way to aim."""
        n = len(self.pop)
        out = []
        for i in range(n):
            pool = [j for j in range(n) if j != i]
            if len(pool) >= 3:
                ia, ib, ic = self.rng.sample(pool, 3)
                a, b, c = self.pop[ia], self.pop[ib], self.pop[ic]
            else:
                a = b = c = self.pop[i]
            f_i = 0.4 + 0.5 * self.rng.random()
            jrand = self.rng.randrange(len(self.keys))
            trial = {}
            for ki, k in enumerate(self.keys):
                lo, hi = self.bounds[k]
                if self.rng.random() < self.cr or ki == jrand:
                    v = a[k] + f_i * (b[k] - c[k])
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

    def top_k(self, k: int) -> list[tuple[dict, float]]:
        """The k best members by training fitness — validated OOS by the caller so
        an overfit training-best can't hide a member that actually generalizes."""
        order = sorted(range(len(self.pop)), key=lambda j: self.fitness[j], reverse=True)
        return [(dict(self.pop[j]), self.fitness[j]) for j in order[:max(1, k)]]

    def sample_outside(self, k: int, n: int) -> list[dict]:
        """`n` members drawn from BELOW the training top-k — the part of the
        population the judge never sees.

        Training fitness ranks candidates against their out-of-sample return at
        rho +0.07, so the top-k is close to an arbitrary slice of the population
        rather than its best part. When the desk is in a drought that slice
        provably contains nothing promotable, and the only place left to look is
        the rest. Spread evenly down the ranking instead of taking the next k+1..
        k+n, which would just be more of the same neighbourhood."""
        order = sorted(range(len(self.pop)), key=lambda j: self.fitness[j], reverse=True)
        rest = order[max(1, k):]
        if not rest or n <= 0:
            return []
        step = max(1, len(rest) // n)
        return [dict(self.pop[j]) for j in rest[::step][:n]]

    def reinject(self, frac: float = 0.4) -> int:
        """Replace the worst `frac` of the population with fresh random vectors to
        escape a converged (overfit) basin — a diversity restart. Returns how many
        were replaced; they're marked unscored so they're re-evaluated next cycle."""
        n = len(self.pop)
        k = max(1, int(n * frac))
        order = sorted(range(n), key=lambda j: self.fitness[j])   # worst first
        for j in order[:k]:
            self.pop[j] = self._rand_vec()
            self.fitness[j] = -1e9
        return k

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
            atomic_write(self.state_path, json.dumps({
                "objective": OBJECTIVE_VER,
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
        # SCORES DO NOT SURVIVE AN OBJECTIVE CHANGE. Members keep their fitness
        # between cycles while the data window holds, so a population carrying
        # scores from one aggregator into selection against trials scored by
        # another would let stale numbers win on scale alone. The GENES are
        # still worth keeping -- generations of search live in them -- so the
        # population loads and only its scores are voided.
        stale = int(d.get("objective", 0)) != OBJECTIVE_VER
        if stale:
            log.info("DE state written under objective v%s, current is v%s: "
                     "keeping %d members, discarding their scores",
                     d.get("objective", 0), OBJECTIVE_VER, len(self.pop))
        self.fitness = [] if stale else [float(x) for x in d.get("fitness", [])]
        if len(self.fitness) != len(self.pop):
            self.fitness = [-1e9] * len(self.pop)
        self.generation = int(d.get("generation", 0))
        if len(self.pop) < self.pop_size:
            # the configured population grew (e.g. the compiled kernel made
            # bigger gene pools cheap): keep every saved member and top up with
            # fresh explorers, marked unscored so they're evaluated next cycle.
            while len(self.pop) < self.pop_size:
                self.pop.append(self._rand_vec())
                self.fitness.append(-1e9)
        else:
            self.pop_size = len(self.pop)
        return True
