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
import math
import random
import statistics
import time

from ..config import MODE_IDLE
from ..exchange.models import ContractSpec
from ..strategy.regime_profile import build_profile, classify_era, dominant_regime
from ..util import clamp, interval_ms
from .backtest import TUNABLES, _coerce
from .search import (STATE_PATH, DEOptimizer, portfolio_folds, recency_weights,
                     robust_aggregate, score_fold, validate_params_portfolio)

log = logging.getLogger("autotuner")

POP_SIZE = 56               # doubled once the compiled kernel made candidate
                            # scoring ~an order of magnitude cheaper: a wider
                            # gene pool searches more of the space per cycle
                            # and resists premature convergence
IMPROVE_MARGIN = 1.06       # OOS challenger must beat champion OOS x this
DEFLATE_K = 0.03            # margin inflation per decade of candidates tried on this
                            # OOS window — multiple-testing honesty: after thousands
                            # of shots at the same gate, a marginal "win" is luck
DEFLATE_CAP = 0.10          # never demand more than +10% extra margin
MARGIN_FLOOR = 0.20         # the margin is a fraction of the incumbent's
                            # magnitude, so a champion sitting near zero would
                            # set a bar only microscopically above itself; this
                            # is the smallest magnitude the margin is charged on
# THERE IS NO ABSOLUTE FITNESS FLOOR ANY MORE, and the reason is measured.
#
# At production geometry, of 121 parameter sets scored across the judged folds,
# 32 genuinely made money over the whole out-of-sample stretch with 30+ trades
# at a profit factor above one. Only 2 of those 32 — six percent — could clear a
# composite floor of 0.15. The 30 it refused had a median of +2.10% over 56
# trades at PF 1.23: ordinary good champions, blocked by a number rather than by
# their results. The live board showed the same thing from the other side, a
# champion earning +2.39% over 109 trades and scoring -0.367 against a bar of
# 0.164, for 240 cycles without a promotion.
#
# The cause is structural, not a bad constant. _fitness penalises a losing
# window roughly 1.5-2x harder than it rewards a winning one, and _oos_composite
# gives 30% weight to the WORST fold — so a set with one losing fold out of four
# is pushed negative however profitable it was overall. Raising or lowering the
# number cannot fix a scale that profitable sets do not live on.
#
# So the absolute question — "is this set any good" — is answered where it can
# be answered honestly: pooled economics over the judged stretch (made money,
# MIN_POOLED_TRADES behind it, profit factor above one), which is scale-free and
# cannot drift. The composite keeps the job it is genuinely good at, RANKING
# inside that admitted pool: top-1 by composite returned +16.81% against +2.48%
# for the pool median, monotone all the way down.
TOP_K_VALIDATE = 10         # DE members sent to the REAL (full-python, portfolio,
                            # meta-aware) judge each cycle. Was 5 of a 56-member
                            # population: the funnel, not the search, was the
                            # bottleneck on champion quality. Widening it costs
                            # ~1s of wall time per cycle on the research pool and
                            # keeps the judge unbiased — the alternative (a
                            # cheap kernel pre-screen) would have filtered
                            # candidates with a brain that has no meta head.
MIN_VETO_TRADES = 5         # a single fold below this is not a verdict — it is
                            # the placeholder ramp _fitness returns when there is
                            # nothing to judge
MIN_POOLED_TRADES = 30      # ...and the EVIDENCE FLOOR that actually decides a
                            # promotion, counted across ALL judged folds rather
                            # than the newest one alone. Measured: judging on the
                            # newest fold's trade count rejected 49% of
                            # candidates on the coin flip of whether one 6-day
                            # window happened to fire five times, and the folds
                            # that survived carried ~7 trades — below the point
                            # where the judge's verdict correlates with anything.
                            # Pooling the judged stretch turns ~7 trades into
                            # ~30-60, which is where a verdict starts to hold.
# The OOS folds trade the last OOS_TAIL_FRAC of the series, so DE TRAINING must
# stop where they begin. It used to stop at 75% while the folds started at 60%:
# fold 0 was 100% inside the training window and fold 1 50% inside, which means
# half of every "out-of-sample" verdict was scored on data the search had
# already fitted. Deriving one from the other makes the two impossible to drift
# apart again. PURGE_BARS then drops the last stretch of training so a training
# bar's outcome — which matures `horizon_bars` later — cannot land inside the
# judge's window either (standard purging; the horizon box tops out at 16).
OOS_TAIL_FRAC = 0.40
TRAIN_FRAC = 1.0 - OOS_TAIL_FRAC
PURGE_BARS = 64
TRAIN_BARS_CAP = 6000       # the SEARCH's window is bounded independently of the
                            # JUDGE's. Fold length is what the measurement
                            # constrains, and it constrains the judged folds;
                            # the DE was converging fine on ~5100 bars and
                            # doubling it only doubled the cycle. Slightly above
                            # the depth that was already working, so nothing
                            # regresses, and taken from the bars nearest the cut
                            # because those are the ones that resemble tomorrow.
WEAK_BAR_MULT = 1.25        # a challenger that loses money across the median
                            # historical era must clear a higher bar
SPECIALIST_CANDS = 5        # candidates the per-symbol bench considers (its cost
                            # is candidates x symbols x folds, so it must not
                            # scale with the widened judging funnel)
OOS_FOLDS = 4               # purged sequential portfolio folds a champion must earn
                            # across — 4 gives the median+worst composite a real
                            # median (undersized folds self-drop in portfolio_folds)
# THE FOLD LENGTH IS THE WHOLE BALL GAME. Measured on 181 parameter sets scored
# across purged portfolio folds, leaving each fold out in turn as the unseen
# future (traded bars = fold length minus the 300-bar indicator warmup):
#
#   traded bars   median trades   folds under the   best composite   lift of the
#   per fold      per fold        t<5 placeholder   anyone reaches   promoted set
#   600                8               32%              -0.39          +0.01 pp
#   960               14               14%              +0.86          +3.59 pp
#
# Same judge, same statistic, same +0.15 bar. At 600 traded bars the judge's
# output NEVER REACHES ITS OWN PROMOTION BAR — not once in 966 evaluations —
# and its ranking carries no usable signal. At 960 it clears the bar and picks
# sets that beat the field by ~3.6 points of return on a window they were never
# scored on, positive in all 8 splits. The same sweep, split by evidence rather
# than by length, puts the turn between them: a candidate's mean R over half the
# windows predicted the other half at r=-0.38 below 15 trades and r=+0.49 at
# 30-60.
#
# A 90-day lookback split four ways gave production 864 traded bars per judged
# fold — between the two rows, and the live board settles which side: every
# champion in the vault negative, best -0.115, and 618 research cycles without a
# promotion. That is the failing row, and it is a property of the geometry, not
# of the promotion logic that was repeatedly blamed for it.
MIN_OOS_TRADED_BARS = 900   # judged folds are widened (by using fewer of them)
                            # until each carries at least this many TRADED bars,
                            # rather than slicing a short history into confetti
VAULT_CANDIDATES = 4        # also re-validate this many top vault champions each cycle (candidate pool, not graveyard)
STALL_REINJECT = 20         # cycles without a promotion before a diversity restart
# THE STAND-DOWN, and why it is no longer a fitness threshold.
#
# A champion scoring above the removal floor can never be removed, and a
# challenger scoring below the promotion bar can never replace it: between the
# two is a DEAD ZONE, a seat nobody can take and nobody can revoke. A floor at
# -0.5 against a bar of 0.15 made that zone 0.65 wide, and a live incumbent
# measured at -0.177 sat in it for 450 consecutive cycles, losing on every
# re-validation while the streak counter reset each time.
#
# Raising the floor to 0.0 closed the zone and opened a worse hole. Measured
# across 966 evaluations at the old fold length, the BEST composite any
# parameter set reached was -0.18 — including every set that was genuinely
# profitable over the whole out-of-sample stretch. A floor at zero sits above
# that entire distribution, so it would have stood down every champion on a
# six-cycle timer, forever, resetting to baseline in a loop.
#
# Both failures come from the same mistake: pinning a decision to a number on a
# scale the fitness function owns and can move. The removal test now reads the
# incumbent's POOLED ECONOMICS instead — did it make or lose money across the
# windows it was judged on, with enough trades for that to be a verdict — which
# is scale-free, means the same thing in every market, and is the sentence a desk
# would actually say. The dead zone is now the promotion bar alone, which is
# where an overfit guard belongs.
DEAD_CHAMPION_TRADES = 3    # a champion that took fewer than this many trades in
                            # the ENTIRE judged stretch is not being cautious, it
                            # is not trading; that vacates the seat too
DEMOTE_PATIENCE = 6         # consecutive losing cycles before a stand-down.
                            # Re-validation reruns the same window each cycle, so
                            # consecutive readings are highly correlated and one
                            # negative stretch must not vacate the seat.
DEMOTE_PATIENCE_WEAK = 3    # a champion that also FAILED the regime gauntlet has
                            # already been told it does not survive other eras;
                            # it does not get the full benefit of the doubt
LIVE_MIN_SAMPLE = 8         # real trades before live evidence can demote: 1 win in
                            # 8 when the validation promised ~70% WR is a ~0.1%
                            # coincidence — waiting longer just pays more tuition
LIVE_RECENT_N = 20          # judge a champion on its most recent real trades
LIVE_PF_FLOOR = 0.7         # real profit factor below this (and < 1/2 expectation) -> demote
# RESEARCH HISTORY IS COUNTED IN BARS, NOT DAYS. What the fold-length table
# above measures is bars per judged fold, and that is what the search needs:
# enough bars that OOS_TAIL_FRAC of them, split OOS_FOLDS ways, clears
# MIN_OOS_TRADED_BARS. A fixed number of DAYS answers that question correctly
# for exactly one bar size and badly for every other — 180 days is 17k bars at
# 15m and 259k at 1m, and the trial clock runs at 1m, 3m or 5m. Ten symbols of
# 1m history at a day-based lookback is 2.6 MILLION candles held in the research
# cache, on a machine that is also trading.
RESEARCH_BARS = 17_280      # 4 judged folds x ~1730 traded bars, plus the
                            # training window: the geometry the measurement
                            # shows works, expressed in the unit it was
                            # measured in. (At 15m this is 180 days, double the
                            # 90 that produced 618 cycles without a champion.)
LOOKBACK_MIN_DAYS = 20.0    # ...but never ask for so little wall-clock history
LOOKBACK_MAX_DAYS = 240.0   # that regimes vanish, nor so much that a slow clock
                            # spends the session downloading. Outside this band
                            # _oos_fold_count widens the folds instead.
DATA_TTL_S = 1800
GAP_FAST = 20               # cadence right after a promotion (keep hammering)
GAP_SLOW = 60               # cadence when stable
DUTY_CYCLE = 0.28           # research may use at most ~this fraction of wall time:
                            # the sleep stretches with the measured cycle length so
                            # a slow host isn't pinned wall-to-wall by the tuner
VAULT_REVAL_EVERY = 15      # re-validate the champion vault every N cycles
META_TRAIN_EVERY = 12       # retrain the meta-labeling model every N cycles
SPECIALIST_EVERY = 5        # per-symbol specialist (overlay) pass every N cycles —
                            # it needs multi-fold statistics, not per-cycle churn
MIN_BARS = 3000
MIN_FOLD_BARS = 450         # a training fold below this can't produce a meaningful
                            # backtest (warmup + a real trading tail)


def lookback_days(interval: str) -> float:
    """How much wall-clock history RESEARCH_BARS is on this clock, bounded.

    The bound is what keeps a 1m trial clock from asking for a year and a 1h
    clock from asking for two hundred bars: below the floor there is no regime
    variety to learn from, above the ceiling the download outlasts the session.
    When the bound bites, the fold count gives way instead — see
    _oos_fold_count."""
    per_day = 86_400_000.0 / max(1, interval_ms(interval))
    return clamp(RESEARCH_BARS / per_day, LOOKBACK_MIN_DAYS, LOOKBACK_MAX_DAYS)


def _current_params(cfg) -> dict:
    p = {}
    for name, (_lo, _hi, grp, _kind) in TUNABLES.items():
        src = cfg.strategy if grp == "strategy" else cfg.risk
        p[name] = getattr(src, name)
    return p


def _train_split(candles: list) -> list:
    """The bars the DE may FIT on: everything before the OOS folds start, minus
    a purge gap. Anything after this point is the judge's, and the judge's
    windows must contain no bar the search has already seen.

    Bounded at TRAIN_BARS_CAP, taking the bars nearest the cut. Doubling the
    lookback fixed the JUDGE — that is what the fold-length measurement
    constrains — but it also doubled the SEARCH, which nothing measured said was
    broken: DE training went from 8.1s to 18.4s per worker per cycle, and with
    duty pacing that halves the number of cycles an hour. Champions the gate can
    finally promote, arriving half as often, is a bad trade when the search
    never needed the extra depth. The judge keeps its long folds; the search
    keeps its old cost and gets the bars closest to now."""
    cut = int(len(candles) * TRAIN_FRAC) - PURGE_BARS
    cut = max(MIN_FOLD_BARS, cut)
    return candles[max(0, cut - TRAIN_BARS_CAP):cut]


def _centroid(param_sets: list[dict]) -> dict:
    """Parameter-space centre of several sets, coerced back onto each tunable's
    own type (an averaged bool is a vote, an averaged int is a rounded int)."""
    if not param_sets:
        return {}
    keys = set().union(*(set(p) for p in param_sets))
    out = {}
    for k in keys:
        vals = [float(p[k]) for p in param_sets if k in p]
        if vals and k in TUNABLES:
            out[k] = _coerce(k, sum(vals) / len(vals))
    return out


def _oos_composite(fits: list[float]) -> float:
    """Blend of the TYPICAL fold (median) and the WORST fold. The old mean
    blend let one parabolic window buy the seat: fold fits of [+21, +0.9,
    -2.4] composited to 3.7 and promoted a set whose most recent window lost
    money outright. The median demands profit in the typical window; the
    worst-fold term keeps the tail priced in."""
    return 0.7 * statistics.median(fits) + 0.3 * min(fits)


def _oos_fold_count(cbs: dict[str, list], tail_frac: float = OOS_TAIL_FRAC,
                    warmup: int = 300) -> int:
    """How many judged folds this much history can actually support.

    `portfolio_folds` will happily cut any tail into k slices; what it cannot do
    is make a slice long enough to be worth judging. Below ~900 traded bars a
    fold produces so few trades that its fitness is largely _fitness's
    placeholder ramp (see the table on MIN_OOS_TRADED_BARS), and a median over
    such numbers is a verdict about nothing. So the fold COUNT gives way to the
    fold LENGTH: with thin history the tuner judges on two long windows rather
    than four short ones, and only widens back to four when there are bars to
    spare. The warmup is a lead-in, not evidence, so it is not counted.

    The shortest symbol governs: a portfolio fold is traded on the basket's
    common grid, so taking the longest series would overstate what is there."""
    if not cbs:
        return 1
    n = min(len(cs) for cs in cbs.values())
    tail = int(n * tail_frac)
    for k in range(OOS_FOLDS, 1, -1):
        # portfolio_folds also refuses a fold under warmup+60, so a count it
        # would silently drop must never be returned here either.
        if tail // k >= max(MIN_OOS_TRADED_BARS, 60):
            return k
    return 1


def _promotion_bar(champ_fit: float, margin: float) -> float:
    """The score a challenger must exceed: clear of the incumbent by `margin`.

    A RELATIVE test only. The absolute question — "is this set any good" — is
    answered on pooled economics (made money over the judged stretch, on enough
    trades, at a profit factor above one), which is scale-free. An absolute floor
    on the FITNESS scale was a second, broken answer to the same question, and it
    overrode the good one.

    Measured at production geometry on 121 parameter sets: 32 made money across
    the judged stretch with 30+ trades and PF >= 1, and only 2 of those 32 (6%)
    could clear a composite floor of 0.15. The 30 it refused had a median of
    +2.10% over 56 trades at PF 1.23 — ordinary good champions, blocked by a
    number rather than by their results. The cause is structural: _fitness
    penalises a losing window ~1.5-2x harder than it rewards a winning one, and
    the composite gives 30% weight to the WORST fold, so any set with one losing
    fold out of four is pushed negative no matter how profitable it is overall.
    The median admitted set scored -0.869.

    The composite is still an excellent RANKER inside the admitted pool — top-1
    by composite returned +16.81% against +2.48% for the pool median, monotone
    down the ranking — so it keeps that job and loses the one it was bad at.

    Written additively because the multiplicative form ran BACKWARDS on a losing
    incumbent: `champ_fit * 1.16` is -0.21 when champ_fit is -0.177, so the worse
    a champion did the lower the bar it set, and the multiple-testing inflation
    moved it the wrong way too. As a fraction of the incumbent's MAGNITUDE,
    "beat it by 6%+" means the same thing above and below zero."""
    return champ_fit + (margin - 1.0) * max(abs(champ_fit), MARGIN_FLOOR)


def _pool_stats(stats_list: list[dict]) -> dict:
    """Add several folds' verdicts into ONE account-level summary.

    Log-wealth adds (trading the same set through consecutive windows compounds),
    trades add, gross win and gross loss add; drawdown takes the worst window,
    which is the least it could have been. Unlike the fitness composite this
    stays on a FIXED scale — percent and trades — no matter how the fitness
    function is weighted, which is why the gate's absolute tests are written
    against it. A fitness floor can silently become unreachable when the scoring
    changes; "made money over the judged stretch" cannot."""
    tr = 0
    gw = gl = log_growth = worst_dd = 0.0
    for s in stats_list:
        t = int(s.get("trades", 0) or 0)
        tr += t
        gw += float(s.get("gross_win", 0.0) or 0.0)
        gl += float(s.get("gross_loss", 0.0) or 0.0)
        log_growth += math.log1p(clamp(float(s.get("total_return", 0.0) or 0.0), -0.95, 20.0))
        worst_dd = max(worst_dd, float(s.get("max_drawdown", 0.0) or 0.0))
    pf = (gw / gl) if gl > 0 else (999.0 if gw > 0 else 0.0)
    return {"trades": tr, "total_return": math.expm1(log_growth), "profit_factor": pf,
            "max_drawdown": worst_dd, "gross_win": gw, "gross_loss": gl}


# Brain-only scalars that may differ PER SYMBOL. Risk/exit geometry stays
# global — one account, one risk policy; but BTC and a hot adopted mid-cap
# genuinely should not share one edge threshold.
BRAIN_PARAMS = ("base_threshold", "target_trades_per_hour", "cost_multiple",
                "hedge_eta", "horizon_bars", "min_p_win", "kelly_fraction",
                "desk_tilt")   # a hot mid-cap can genuinely want a different
                               # KIND of desk than BTC, not just a different bar


def overlay_of(params: dict) -> dict:
    """The brain-scalar slice of a full parameter set — what an overlay stores."""
    return {k: params[k] for k in BRAIN_PARAMS if k in params}


def select_specialists(sym_results: dict, margin: float = IMPROVE_MARGIN) -> dict:
    """Pick each symbol's specialist overlay from FOLD-VALIDATED evidence.

    sym_results: {sym: {"applied": (params, fit, pf), "overlay": (params, fit, pf) | None,
                        "cands": [(params, fit, pf), ...]}}
    where every fit is the median+worst composite across the SAME purged OOS
    folds the global promotion uses, and pf is the most recent fold's profit
    factor.

    Rules (the old single-window picker flapped SET/CLEARED every cycle and
    converged to zero overlays — a specialist bench needs statistics and
    hysteresis):
      - a challenger must be profitable where it matters (pooled pf >= 1 over
        the judged folds, with real evidence behind it) and clearly better than
        the applied global set. The absolute FITNESS floor is gone for the same
        reason it left the global gate: measured at production geometry, only
        6% of parameter sets that genuinely made money could reach it. Profit
        is the absolute test; the composite is the ranking;
      - an INCUMBENT overlay keeps its seat while it still beats the global
        set — a challenger must beat the incumbent by the margin, not just
        the global;
      - an overlay that stops beating the global set (or turns unprofitable)
        is cleared — the global set is the best known there.
    Returns {sym: {"params", "fitness", "vs", "pf"}} (missing sym = no overlay)."""
    out: dict[str, dict] = {}
    for sym, r in sym_results.items():
        applied_params, base_fit, _base_pf = r["applied"]
        base_ov = overlay_of(applied_params)
        # sign-safe, exactly like the global bar: base_fit * margin runs
        # backwards below zero and made a losing global set EASIER to overlay
        bar = _promotion_bar(float(base_fit), margin)
        best = None
        for params, fit, pf in r.get("cands", []):
            if pf < 1.0 or fit <= bar:
                continue
            if overlay_of(params) == base_ov:
                continue    # identical brain scalars to the global set — pointless overlay
            if best is None or fit > best[1]:
                best = (params, fit, pf)
        base_fit = float(base_fit)
        inc = r.get("overlay")
        if inc is not None:
            inc_params, inc_fit, inc_pf = inc
            keeps_seat = inc_pf >= 1.0 and inc_fit > base_fit
            if keeps_seat and (best is None or best[1] <= _promotion_bar(inc_fit, margin)):
                out[sym] = {"params": overlay_of(inc_params), "fitness": round(inc_fit, 3),
                            "vs": round(base_fit, 3), "pf": round(inc_pf, 3)}
                continue
        if best is not None:
            out[sym] = {"params": overlay_of(best[0]), "fitness": round(best[1], 3),
                        "vs": round(base_fit, 3), "pf": round(best[2], 3)}
    return out


def _best_fallback(champions: list[dict], interval: str,
                   exclude: str | None = None) -> dict | None:
    """The best vault set to fall back to when the incumbent is stood down.

    Ranked by `fitness`, which now means one thing everywhere — the same
    portfolio composite the promotion gate uses. It used to be written by two
    different judges (see _revalidate_vault) on different scales, so WHICH
    champion the account fell back to depended on which of them had run last.

    Admission is ECONOMIC, not by fitness: a set is worth falling back to when it
    MADE MONEY over the judged stretch on enough trades — the same test a
    promotion has to pass. Ranking among the admitted is by fitness, which is
    what fitness is good at.

    "Positive fitness" used to be the admission test, and on this scale that pool
    is empty in exactly the situation it exists for, since most profitable sets
    score negative. Returning None is correct and deliberate: when the whole
    vault is under water there is nothing to fall back to but the code defaults,
    and swapping one losing set for another is not a defence."""
    pool = [c for c in champions
            if c.get("params") and not c.get("live_flag")
            and c.get("id") != exclude
            and (c.get("clock") or interval) == interval
            and float(c.get("oos_return", 0.0) or 0.0) > 0.0
            and int(c.get("oos_trades", 0) or 0) >= MIN_POOLED_TRADES
            and float(c.get("oos_pf", 0.0) or 0.0) >= 1.0]
    return max(pool, key=lambda c: c.get("fitness", -1e18), default=None)


def _default_params() -> dict:
    """The code-default baseline for every tunable — the safe harbor the
    stand-down falls back to when the whole vault has gone cold."""
    from ..config import RiskConfig, StrategyConfig
    s, r = StrategyConfig(), RiskConfig()
    return {name: getattr(s if grp == "strategy" else r, name)
            for name, (_lo, _hi, grp, _kind) in TUNABLES.items()}


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
        self._cache: dict[str, tuple[list, float]] = {}   # symbol -> (candles, fetched_ts)
        self._rot_idx = -1
        self.research_symbol = ""   # rotates across the top-volume board each window
        self._data_ts = 0.0
        self._scored_ts = -1.0      # data window the population was last fully scored on
        self.cycles = 0
        self.improvements = 0
        self._since_improve = 0
        self._tested_oos = 0        # candidates tried against the current OOS window
        self._champ_bad_streak = 0  # consecutive cycles the incumbent scored toxic on traded symbols
        self.next_run_ts = 0.0
        self.champion_fitness = 0.0
        self._val_cache: dict[tuple, dict] = {}   # OOS validations memoized per data window
        self.last_cycle: dict | None = None
        self.last_meta: dict | None = None   # latest meta-model training result
        self.history: list[dict] = []
        # clock trial: the alternate-interval research track (own gene pool,
        # own candle cache — never mixed with the primary clock's)
        self.trial_de: DEOptimizer | None = None
        self._trial_cache: dict[str, tuple[list, float]] = {}
        self._trial_scored_ts = -1.0
        self._turn = 0
        self.last_trial: dict | None = None
        self._meta_task: asyncio.Task | None = None   # meta training runs in the
        # background: awaiting a 35-70s pool task inline stalled every 12th
        # cycle, and duty pacing then amplified the stall ~2.6x into the gap
        # regime gauntlet: (params-sig, interval, window) -> result. Windows
        # are immutable history, so entries never expire.
        self._gauntlet_cache: dict[tuple, dict] = {}
        # (era, interval) -> (dominant regime, shares). An era's CHARACTER does
        # not depend on which champion is being tested, so it is classified once.
        self._era_regime_cache: dict[tuple, tuple] = {}

    def start(self) -> None:
        if self._task is None or self._task.done():
            self.running = True
            self._task = asyncio.create_task(self._loop(), name="autotuner")

    async def stop(self) -> None:
        self.running = False
        for t in (self._task, self._meta_task):
            if t:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        self._task = self._meta_task = None

    async def _loop(self) -> None:
        await asyncio.sleep(20)
        while self.running:
            cfg = self.orch.cfg
            promoted = False
            t0 = time.monotonic()
            # turning the clock trial OFF takes effect within one loop tick,
            # not at the next restart: the shadow flattens, finalizes its
            # record and stops — "off" must mean off.
            if not cfg.strategy.clock_trial and getattr(self.orch, "shadow", None) is not None:
                try:
                    await self.orch.stop_shadow()
                except Exception as e:  # noqa: BLE001
                    log.warning("shadow shutdown failed: %s", e)
            if cfg.strategy.auto_tune and self.orch.mode != MODE_IDLE and self.orch.engine is not None:
                try:
                    # clock trial ON: research time alternates between the live
                    # clock and the trial clock (each cycle is one or the other,
                    # so the CPU envelope never doubles — the honest cost is
                    # that each clock advances at half cadence while trialing).
                    self._turn += 1
                    trial_turn = (cfg.strategy.clock_trial
                                  and cfg.strategy.trial_interval != cfg.strategy.interval
                                  and self._turn % 2 == 0)
                    if trial_turn:
                        await self._trial_cycle()
                    else:
                        promoted = await self._cycle()
                except Exception as e:  # noqa: BLE001
                    log.warning("auto-tune cycle failed: %s", e)
            cycle_s = time.monotonic() - t0
            # duty-cycle pacing: the sleep stretches with how long the cycle
            # actually took, so research never monopolizes a slow host — a 4-min
            # cycle is followed by a breather sized by the LIVE research_duty
            # setting. The sleep runs in short slices that re-read the setting,
            # so the dashboard slider re-paces even a gap already in progress:
            # turning the dial acts within seconds, never after the old gap.
            slept = time.monotonic()
            while self.running:
                duty = clamp(float(getattr(cfg.strategy, "research_duty", DUTY_CYCLE) or DUTY_CYCLE),
                             0.10, 0.50)
                gap = max(GAP_FAST if promoted else GAP_SLOW,
                          cycle_s * (1.0 - duty) / duty)
                remaining = gap - (time.monotonic() - slept)
                self.next_run_ts = time.time() + max(0.0, remaining)
                if remaining <= 0:
                    break
                await asyncio.sleep(min(5.0, remaining))

    def _universe(self) -> list[str]:
        """The research universe: the radar's ACTUAL top-10 BingX perps by 24h
        USDT volume (clean majors — no index products, no long-tail memes; see
        scanner.top_volume_universe), plus the user's own symbols. Falls back to
        the configured symbols offline."""
        sc = getattr(self.orch, "scanner", None)
        uni = list(sc.top_volume) if sc is not None and sc.top_volume else []
        for s in self.orch.cfg.symbols:
            if s not in uni:
                uni.append(s)
        return uni or list(self.orch.cfg.symbols)

    async def _get_candles(self, symbol: str, interval: str | None = None,
                           cache: dict | None = None) -> list:
        """Research candles for one symbol. The primary clock uses the shared
        cache; the trial clock passes its OWN interval + cache so the two never
        collide (same key, different bar sizes = silent corruption)."""
        cache = self._cache if cache is None else cache
        hit = cache.get(symbol)
        if hit and time.time() - hit[1] < DATA_TTL_S:
            return hit[0]
        cfg = self.orch.cfg
        iv = interval or cfg.strategy.interval
        # history sized in BARS for THIS clock: the trial lane runs at 1m-5m,
        # where a fixed day count is either a rounding error or a gigabyte
        candles = await self.orch._get_backtest_candles(
            symbol, iv, lookback_days(iv), cfg.feed == "synthetic", _NullJob())
        cache[symbol] = (candles, time.time())
        if len(cache) > 10:  # bound the cache to the working set
            oldest = min(cache, key=lambda s: cache[s][1])
            if oldest != symbol:
                cache.pop(oldest, None)
        return candles

    async def _ensure_data(self) -> list:
        """Rotate the research symbol across the universe each data window: every
        ~30 min the DE trains against a different top-volume perp, so surviving
        parameters must work on the BOARD, not on one symbol's quirks."""
        uni = self._universe()
        rotate = (not self.research_symbol
                  or self.research_symbol not in uni
                  or time.time() - self._data_ts >= DATA_TTL_S)
        if rotate:
            self._rot_idx = (self._rot_idx + 1) % len(uni)
            self.research_symbol = uni[self._rot_idx]
        candles = await self._get_candles(self.research_symbol)
        self._data_ts = self._cache[self.research_symbol][1]
        return candles

    def _valid_window(self, candles: list) -> list:
        """The held-out recent window, with a lead-in EXACTLY equal to the
        backtester's warmup (300) so OOS trading starts precisely where training
        stops. The old 400-bar lead-in started trading 100 bars early — 100 bars
        of TRAINING data silently counted toward every 'out-of-sample' score.

        The cut is DERIVED from OOS_TAIL_FRAC rather than written down again. It
        used to be a hard-coded 0.75 against a tail that is 0.40, which was safe
        only by accident: it threw away a third of the held-out data, and the day
        anyone tightened the tail below 0.25 it would have started scoring vault
        champions on bars the search had already fitted, silently."""
        val_cut = int(len(candles) * TRAIN_FRAC)
        return candles[max(0, val_cut - 300):]

    async def _specialist_pass(self, cands: list[dict], applied_params: dict,
                               folds_cbs: list[dict], interval: str, slip: float,
                               strat, risk) -> dict:
        """Score (applied set, incumbent overlay, every candidate's brain
        scalars merged onto the applied risk geometry) PER SYMBOL across the
        same purged OOS folds, in one parallel batch. Near-clone candidates
        collapse after the brain-scalar merge, so the batch stays small.
        Returns the sym_results shape select_specialists() consumes."""
        syms = sorted({s for fc in folds_cbs for s in fc})
        args: list[tuple] = []
        index: list[tuple] = []          # (sym, tag, params, n_folds)
        for sym in syms:
            folds = [{sym: fc[sym]} for fc in folds_cbs if fc.get(sym)]
            if not folds:
                continue
            variants: list[tuple[str, dict]] = [("applied", dict(applied_params))]
            seen = {tuple(sorted(overlay_of(applied_params).items()))}
            cur = self.orch.symbol_overlays.get(sym)
            if cur and cur.get("params"):
                merged = {**applied_params, **cur["params"]}
                key = tuple(sorted(overlay_of(merged).items()))
                if key not in seen:
                    seen.add(key)
                    variants.append(("overlay", merged))
            for c in cands:
                merged = {**applied_params, **overlay_of(c["params"])}
                key = tuple(sorted(overlay_of(merged).items()))
                if key in seen:
                    continue
                seen.add(key)
                variants.append(("cand", merged))
            spec = self.orch.specs.get(sym, ContractSpec(sym))
            for tag, params in variants:
                for fc in folds:
                    args.append((params, fc, interval, {sym: spec}, spec.taker_fee,
                                 slip, strat, risk))
                index.append((sym, tag, params, len(folds)))
        res = await self.orch.map_cpu(validate_params_portfolio, args, research=True)
        out: dict[str, dict] = {}
        i = 0
        for sym, tag, params, nf in index:
            rs = res[i:i + nf]
            i += nf
            fits = [r["fitness"] for r in rs]
            fit = _oos_composite(fits)
            # POOLED, like the global gate. Reading the newest fold's profit
            # factor alone made a specialist seat turn on the coin flip of
            # whether one window happened to fire, and 999-on-three-trades sailed
            # through a `>= 1.0` test here exactly as it once did upstairs.
            pool = _pool_stats([r["stats"] for r in rs])
            pf = (float(pool["profit_factor"])
                  if int(pool["trades"]) >= MIN_POOLED_TRADES and pool["total_return"] > 0.0
                  else 0.0)
            slot = out.setdefault(sym, {"applied": None, "overlay": None, "cands": []})
            if tag == "applied":
                slot["applied"] = (params, fit, pf)
            elif tag == "overlay":
                slot["overlay"] = (params, fit, pf)
            else:
                slot["cands"].append((params, fit, pf))
        return {s: r for s, r in out.items() if r["applied"] is not None}

    def _traded_symbols(self) -> list[str]:
        """What the engine is actually running right now — the set promotions
        must be judged on."""
        eng = self.orch.engine
        if eng is not None and getattr(eng, "ctx", None):
            return list(eng.ctx.keys())
        return list(self.orch.cfg.symbols)

    async def _cycle(self) -> bool:
        cfg = self.orch.cfg
        interval = cfg.strategy.interval
        candles = await self._ensure_data()
        symbol = self.research_symbol
        if len(candles) < MIN_BARS:
            return False
        spec = self.orch.specs.get(symbol, ContractSpec(symbol))
        taker, slip = spec.taker_fee, cfg.paper.slippage_bps
        strat, risk = cfg.strategy, cfg.risk

        n = len(candles)
        train = _train_split(candles)
        valid = self._valid_window(candles)

        # validation BASKET = the symbols the engine is ACTUALLY TRADING (user
        # symbols + adopted). Training may rotate across the majors board for
        # generality, but promotion answers one question only: "is this better
        # on what we are trading right now?" — a champion brilliant on BTC and
        # toxic on an adopted symbol must not win.
        basket: list[tuple[str, list]] = []
        for tsym in self._traded_symbols()[:4]:
            try:
                tc = await self._get_candles(tsym)
                if len(tc) >= MIN_BARS:
                    basket.append((tsym, self._valid_window(tc)))
            except Exception as e:  # noqa: BLE001 — basket breadth is best-effort
                log.debug("basket data %s: %s", tsym, e)
        if not basket:
            basket = [(symbol, valid)]

        champ = _current_params(cfg)
        if not self.de.ready():
            if not self.de.load():
                self.de.seed_population(champ)
                # cold start: the vault's best sets join the gene pool so the
                # search resumes from everything already proven, not from noise.
                for c in sorted(self.orch.champions, key=lambda c: c.get("fitness", 0.0),
                                reverse=True)[:3]:
                    if c.get("params"):
                        self.de.inject(c["params"])
        self.de.inject(champ)

        # folds scale with the research pool: more cores -> more (finer) folds,
        # one fold per worker, indicators built once per fold — but never so
        # many that a fold drops below what a backtest needs (score_fold
        # returns -1 under 360 bars; on an 8-core host with minimal data that
        # used to zero out EVERY fold and turn DE selection into pure noise).
        max_folds_by_data = max(1, len(train) // MIN_FOLD_BARS)
        nf = int(clamp(min(self.orch.research_workers, max_folds_by_data), 1, 8))
        folds = _make_folds(train, nf)
        trials = self.de.trials()
        # Only re-score the whole population when the data window changed (every
        # ~30 min) or a member is unscored (freshly injected); otherwise member
        # fitness carries forward on the same folds and we score just the trials —
        # halving the work and roughly doubling generations-per-hour in steady state.
        need_members = (self._scored_ts != self._data_ts) or any(f <= -1e8 for f in self.de.fitness)
        if self._scored_ts != self._data_ts:
            self._tested_oos = 0    # fresh OOS window -> the multiple-testing meter resets
            self._val_cache.clear()  # ...and cached OOS validations expire with it
        self._scored_ts = self._data_ts
        candidates = (list(self.de.pop) + trials) if need_members else list(trials)
        args = [(fold, symbol, interval, spec, taker, slip, strat, risk, candidates) for fold in folds]
        # CO-TRAINING. The DE used to rank candidates on ONE rotating symbol
        # while promotion judged them on the traded PORTFOLIO — two different
        # objectives, so most of the search's progress never survived contact
        # with the judge and the top-k it nominated was close to arbitrary.
        # A second traded symbol in the training objective costs one more
        # kernel pass (the compiled path makes this nearly free) and makes
        # "best in training" mean something much closer to "best for the
        # account". Symbol-specific quirks now have to be paid for twice.
        co_sym, co_folds = "", []
        for tsym in self._traded_symbols():
            if tsym == symbol:
                continue
            hit = self._cache.get(tsym)
            if hit and len(hit[0]) >= MIN_BARS:
                cf = _make_folds(_train_split(hit[0]), nf)
                # a fold below the backtester's floor scores -1.0 for EVERY
                # candidate: harmless to the ordering but pure wasted CPU, so
                # only co-train when the second symbol can carry the same
                # fold count the primary is using.
                if cf and min(len(f) for f in cf) >= MIN_FOLD_BARS:
                    co_sym, co_folds = tsym, cf
                break
        if co_folds:
            co_spec = self.orch.specs.get(co_sym, ContractSpec(co_sym))
            args += [(fold, co_sym, interval, co_spec, co_spec.taker_fee, slip,
                      strat, risk, candidates) for fold in co_folds]
        raw_fits = await self.orch.map_cpu(score_fold, args, research=True)
        ok = lambda ff: bool(ff) and len(ff) == len(candidates)   # noqa: E731
        fold_fits = [ff for ff in raw_fits[:len(folds)] if ok(ff)]
        co_fits = [ff for ff in raw_fits[len(folds):] if ok(ff)]
        if not fold_fits:
            return False
        robust = [robust_aggregate(list(fc), recency_weights(len(fold_fits)))
                  for fc in zip(*fold_fits)]
        if co_fits:   # equal say to each symbol, so neither can carry a set alone
            robust_co = [robust_aggregate(list(fc), recency_weights(len(co_fits)))
                         for fc in zip(*co_fits)]
            robust = [0.5 * (a + b) for a, b in zip(robust, robust_co)]
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
        # only same-clock champions are candidates: a 5m-born set's bar-count
        # parameters (horizon, time stops, maker windows) mean different real
        # time on a 15m engine — validating it here would burn basket runs on
        # a set that could never be honestly applied. (Entries without a clock
        # tag predate the trial feature and are treated as current-clock.)
        vault = sorted((c for c in self.orch.champions
                        if not c.get("live_flag")
                        and (c.get("clock") or interval) == interval),
                       key=lambda c: c.get("fitness", 0.0), reverse=True)[:VAULT_CANDIDATES]
        cands: list[dict] = [{"source": "de", "params": p, "train_fit": tfit, "cid": None}
                             for p, tfit in topk]
        cands += [{"source": "vault", "params": c.get("params", {}), "train_fit": None, "cid": c.get("id")}
                  for c in vault]
        # CONSENSUS candidate: the centroid of the population's best members.
        # Picking the single highest-scoring set is picking the luckiest draw
        # from a noisy landscape; the centre of a good REGION is usually the
        # sturdier choice, which is exactly the property that survives contact
        # with a moving market. It earns nothing for being the average — it
        # faces the same OOS folds, the same PF and evidence vetoes and the
        # same deflated margin as everyone else.
        if len(topk) >= 3:
            mid = _centroid([p for p, _ in topk[:5]])
            if mid and not any(self.orch._params_match(mid, c["params"]) for c in cands):
                cands.append({"source": "consensus", "params": mid, "train_fit": None, "cid": None})
        # dedupe identical parameter sets (a converged population's top-k are
        # often clones, and a vault champion may equal a DE member): validating
        # duplicates wastes basket runs and double-counts the multiple-testing
        # meter for what is really one candidate.
        uniq: list[dict] = []
        for c in cands:
            if not any(self.orch._params_match(c["params"], u["params"]) for u in uniq):
                uniq.append(c)
        cands = uniq

        # OOS validation is now PORTFOLIO fitness across purged sequential
        # folds: every candidate runs the shared-account simulator over the
        # traded basket for each of the last K disjoint windows. Promotion
        # answers "does this make the ACCOUNT richer, consistently" — not
        # "does it flatter one symbol in one lucky window".
        cbs_full: dict[str, list] = {}
        for tsym in self._traded_symbols()[:4]:
            hit = self._cache.get(tsym)
            if hit and len(hit[0]) >= MIN_BARS:
                cbs_full[tsym] = hit[0]
        if not cbs_full:
            cbs_full = {symbol: candles}
        folds_cbs = portfolio_folds(cbs_full, k=_oos_fold_count(cbs_full),
                                    tail_frac=OOS_TAIL_FRAC)
        if not folds_cbs:
            return False
        nf_oos = len(folds_cbs)
        specs_map = {s: self.orch.specs.get(s, ContractSpec(s)) for s in cbs_full}
        # MEMOIZED within the OOS window: the champion and the vault candidates
        # are re-validated every cycle on the SAME folds (the window refreshes
        # every ~30 min) — identical inputs, identical outputs. Only genuinely
        # NEW candidates pay for full-fidelity python backtests; stasis cycles
        # drop from ~40 portfolio runs to a handful, and the saved duty budget
        # becomes more DE generations of actual search.
        def _psig(params: dict) -> tuple:
            return tuple(sorted((k, round(float(v), 10)) for k, v in params.items()))
        all_keys, miss_args, miss_keys = [], [], []
        for c in cands + [{"params": champ}]:
            sig = _psig(c["params"])
            for fi, fc in enumerate(folds_cbs):
                key = (sig, fi)
                all_keys.append(key)
                if key not in self._val_cache and key not in miss_keys:
                    miss_args.append((c["params"], fc, interval, specs_map, taker, slip, strat, risk))
                    miss_keys.append(key)
        if miss_args:
            miss_res = await self.orch.map_cpu(validate_params_portfolio, miss_args, research=True)
            for key, r in zip(miss_keys, miss_res):
                self._val_cache[key] = r
        val_res = [self._val_cache[k] for k in all_keys]

        def cand_fit(idx: int) -> tuple[float, dict, list[float], dict]:
            rs = val_res[idx * nf_oos:(idx + 1) * nf_oos]
            fits = [r["fitness"] for r in rs]
            stats = [r["stats"] for r in rs]
            # `stats[-1]` (newest fold) still travels for the champion record and
            # the dashboard; the GATE reads the pooled column instead.
            return _oos_composite(fits), stats[-1], fits, _pool_stats(stats)

        champ_fit, _, champ_folds, champ_pool = cand_fit(len(cands))
        best, best_i, best_adj, best_stats = None, -1, -1e18, {}
        best_folds: list[float] = []
        best_pool: dict = {}
        pf_passed = 0    # candidates whose pooled economics cleared (promotable pool)
        thin_rejected = 0  # ...and those refused for judging on too little evidence
        ranked: list[tuple[float, int]] = []   # (OOS score, index) of the promotable
        for i, c in enumerate(cands):
            oos, stats0, fold_fits_oos, pool = cand_fit(i)
            # NO in-sample-vs-OOS value penalty anymore: training fitness is a
            # SINGLE-SYMBOL score and OOS is a PORTFOLIO composite — different
            # simulators, different units. Subtracting them charged every DE
            # candidate (the only source of NEW champions) a handicap
            # proportional to a unit mismatch, not to measured overfitting.
            # Overfit protection lives where the units are consistent: the
            # median+worst composite, the majority-fold beat, the PF >= 1
            # veto and the deflated margin — all pure OOS.
            adj = oos
            if c["source"] == "vault" and c["cid"]:
                # keep its CURRENT eval fresh, pooled economics included
                self.orch.set_champion_current(c["cid"], oos, stats0, pool)
            # ABSOLUTE-PROFIT VETO, on the POOLED judged stretch. Whatever the
            # fitness composite says, a set that did not actually make the
            # account money across the windows it was judged on cannot take the
            # seat — and "made money" is read off summed log-wealth and summed
            # gross win/loss, not off one window.
            #
            # This used to read the NEWEST FOLD ALONE, which had two failures.
            # A set could be promoted off three trades with no losers (PF 999
            # sails past a `< 1.0` test) — that hole was walked through by a
            # real champion. And measurement showed the opposite edge doing more
            # damage: requiring five trades in one 6-day window rejected 49% of
            # all candidates on the coin flip of whether that particular window
            # happened to fire, regardless of how the set did everywhere else.
            # Pooling answers both — 999 needs real losing trades to survive
            # alongside it, and the evidence floor is met by the whole stretch.
            if int(pool.get("trades", 0) or 0) < MIN_POOLED_TRADES:
                thin_rejected += 1
                continue
            if (float(pool.get("profit_factor", 0.0) or 0.0) < 1.0
                    or float(pool.get("total_return", 0.0) or 0.0) <= 0.0):
                continue
            pf_passed += 1
            ranked.append((adj, i))
            if adj > best_adj:
                best, best_i, best_adj, best_stats = c, i, adj, stats0
                best_folds, best_pool = fold_fits_oos, pool

        self.cycles += 1
        self.champion_fitness = round(champ_fit, 3)
        promoted = False
        best_params = best["params"] if best else {}
        different = bool(best) and any(abs(best_params.get(k, 0) - champ.get(k, 0)) > 1e-9 for k in champ)
        # deflated margin: the more candidates have taken a shot at THIS OOS
        # window, the more a challenger must win by — a marginal beat after
        # thousands of tries is selection bias, not signal.
        self._tested_oos += len(cands)
        margin = IMPROVE_MARGIN + min(DEFLATE_CAP, DEFLATE_K * math.log10(1 + self._tested_oos / 10))
        # rank stability: beyond beating the champion's blended score, the
        # challenger must beat it in a MAJORITY of the purged folds — a set
        # that wins on average but loses most windows is one lucky window.
        beat = sum(1 for a, b in zip(best_folds, champ_folds) if a > b)
        bar = _promotion_bar(champ_fit, margin)
        gaunt = None
        gauntlet_blocked = False
        qualifies = different and best_adj > bar and beat * 2 > nf_oos
        if qualifies:
            # REGIME GAUNTLET: score the incoming champion across five years of
            # Binance regime eras (2021 top, 2022 crash, 2023 chop, ...) —
            # same portfolio simulator, same BingX fees, Binance prices.
            #
            # It now runs BEFORE the decision instead of after it. The evidence
            # was already being paid for and then ignored at the only moment it
            # could have mattered: a set that beats the incumbent by a hair on
            # 36 days but loses money across the median historical era is far
            # more likely to be fitting the recent window than to have found
            # something. Such a challenger must clear a HIGHER bar. It is still
            # never an outright veto — a 2022-shaped failure says little about a
            # 2026 edge — and with no internet there is no verdict and nothing
            # changes, so research never depends on Binance being reachable.
            try:
                gaunt = await self._gauntlet(best_params, interval, taker, slip, strat, risk)
            except Exception as e:  # noqa: BLE001 — evidence, never a dependency
                log.debug("gauntlet failed: %s", e)
            if gaunt is not None and gaunt.get("weak"):
                bar = _promotion_bar(champ_fit, 1.0 + (margin - 1.0) * WEAK_BAR_MULT)
                qualifies = best_adj > bar
                gauntlet_blocked = not qualifies
                if gauntlet_blocked:
                    log.info("promotion held: challenger %.2f fails history "
                             "(median era %.2f) and misses the raised bar %.2f",
                             best_adj, gaunt.get("median", 0.0), bar)
        if qualifies:
            self.orch.apply_params(best_params)
            self.improvements += 1
            promoted = True
            self._tested_oos = 0    # a promotion resets the bias meter
            vs = best_stats
            # tag the champion now driving live trades: reuse the vault entry if
            # the winner came from the vault, otherwise mint a new one.
            cid = best["cid"] if (best["source"] == "vault" and best["cid"]) \
                else self.orch.record_champion(best_params, best_adj, vs, clock=interval)
            ch = self.orch.find_champion(cid)
            if ch is not None:
                ch["clock"] = interval
                if gaunt is not None:
                    ch["gauntlet"] = gaunt
                    ch["gauntlet_weak"] = bool(gaunt.get("weak"))
            self.orch.mark_champion_used(cid)
            self.history.append({
                "ts": int(time.time() * 1000),
                "from_fitness": round(champ_fit, 3), "to_fitness": round(best_adj, 3),
                "folds_beaten": f"{beat}/{nf_oos}",
                "fold_fits": [round(f, 2) for f in best_folds],
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

        # DEFENSIVE STAND-DOWN: the promotion gate only swaps for something
        # BETTER — it never removed something TOXIC. If the incumbent keeps
        # LOSING MONEY on the symbols we actually trade and nothing beats the
        # bar, stop trading it: fall back to the best still-positive vault set,
        # else to the code-default baseline.
        #
        # The trigger is the incumbent's POOLED ECONOMICS over the judged
        # stretch, not a fitness threshold. A fitness floor is a number on a
        # scale the scoring function owns, and that scale moves: measured across
        # 966 evaluations at the old fold length, NO parameter set — including
        # every one that was genuinely profitable — ever scored above -0.18, so
        # a floor at 0.0 would have stood every champion down on a six-cycle
        # timer forever. "Lost money over the windows we judged it on, with
        # enough trades for that to mean something" cannot drift that way, and
        # it is the same sentence a desk would use.
        act_now = self.orch.find_champion(self.orch.active_champion_id) or {}
        weak_now = bool(act_now.get("gauntlet_weak"))
        patience = DEMOTE_PATIENCE_WEAK if weak_now else DEMOTE_PATIENCE
        champ_trades = int(champ_pool.get("trades", 0) or 0)
        champ_ret = float(champ_pool.get("total_return", 0.0) or 0.0)
        # ...and a champion that barely trades at all is not "safe", it is a seat
        # doing nothing. Below the evidence floor there is no verdict to appeal.
        champ_losing = (champ_ret < 0.0 and champ_trades >= MIN_POOLED_TRADES) \
            or champ_trades < DEAD_CHAMPION_TRADES
        if promoted or not champ_losing:
            self._champ_bad_streak = 0
        else:
            self._champ_bad_streak += 1
            if self._champ_bad_streak >= patience:
                self._champ_bad_streak = 0
                alt = _best_fallback(self.orch.champions, interval)
                fb_params = alt["params"] if alt else _default_params()
                fb_name = "vault fallback" if alt else "baseline reset"
                if any(abs(fb_params.get(k, 0) - champ.get(k, 0)) > 1e-9 for k in champ):
                    self.orch.apply_params(fb_params)
                    if alt:
                        self.orch.mark_champion_used(alt["id"])
                    self.history.append({
                        "ts": int(time.time() * 1000),
                        "from_fitness": round(champ_fit, 3),
                        "to_fitness": round(alt.get("fitness", 0.0), 3) if alt else 0.0,
                        "valid_wr": 0.0, "valid_pf": 0.0, "gen": self.de.generation,
                        "source": f"defensive ({fb_name})",
                        "params": {k: fb_params[k] for k in ("base_threshold", "risk_per_trade",
                                   "target_trades_per_hour") if k in fb_params},
                    })
                    self.history = self.history[-25:]
                    log.warning("auto-tune STAND-DOWN: incumbent %.2f on traded basket -> %s",
                                champ_fit, fb_name)

        # PER-SYMBOL SPECIALIST BENCH (every SPECIALIST_EVERY cycles): the
        # global promotion judges the whole basket, so a set that is brilliant
        # on ONE symbol but average elsewhere can never win the global seat —
        # the "a good BTC champion blocks a better SOL specialist" failure.
        # Here each traded symbol runs its own contest: every candidate's
        # brain scalars merged onto the applied risk geometry, validated on
        # that symbol alone across the SAME purged OOS folds, PF-gated, with
        # incumbent hysteresis. (The old picker judged one window every cycle
        # with no gates — it flapped SET/CLEARED endlessly, burned a ~30-
        # backtest batch per cycle, and still converged to zero overlays.)
        if self.cycles % SPECIALIST_EVERY == 0:
            try:
                applied_params = best_params if promoted else champ
                # the bench costs candidates x symbols x folds, so it takes the
                # best few by OOS rather than everything the widened funnel
                # judged — its job is finding per-symbol overlays, and a
                # near-clone of a mediocre set has nothing to teach a symbol.
                top = [cands[i] for _s, i in sorted(ranked, reverse=True)[:SPECIALIST_CANDS]]
                sym_results = await self._specialist_pass(top or cands[:SPECIALIST_CANDS],
                                                          applied_params, folds_cbs,
                                                          interval, slip, strat, risk)
                overlays = select_specialists(sym_results)
                self.orch.update_symbol_overlays(overlays, traded=list(sym_results))
            except Exception as e:  # noqa: BLE001 — specialists are an optimization, never fatal
                log.warning("specialist pass failed: %s", e)

        # LIVE-EVIDENCE DEMOTION: backtests propose, live results dispose. A
        # champion with a real sample whose actual profit factor collapsed vs
        # its validation expectation stops trading NOW, whatever backtests say.
        act = self.orch.find_champion(self.orch.active_champion_id)
        if act is not None and not act.get("live_flag"):
            lv = self.orch.champion_live_stats(recent_n=LIVE_RECENT_N).get(act["id"])
            if (lv and lv["trades"] >= LIVE_MIN_SAMPLE and lv["pf"] < LIVE_PF_FLOOR
                    and lv["pf"] < 0.5 * max(act.get("profit_factor", 1.0), 1.0)):
                act["live_flag"] = {"pf": lv["pf"], "trades": lv["trades"],
                                    "ts": int(time.time() * 1000)}
                alt = _best_fallback(self.orch.champions, interval, exclude=act["id"])
                fb = alt["params"] if alt else _default_params()
                self.orch.apply_params(fb)
                if alt is not None:
                    self.orch.mark_champion_used(alt["id"])
                else:
                    self.orch.save_champions()
                self.history.append({
                    "ts": int(time.time() * 1000),
                    "from_fitness": round(champ_fit, 3),
                    "to_fitness": round(alt.get("fitness", 0.0), 3) if alt else 0.0,
                    "valid_wr": round(lv["win_rate"], 3), "valid_pf": round(lv["pf"], 3),
                    "gen": self.de.generation,
                    "source": f"live-evidence demotion (real PF {lv['pf']:.2f} on {lv['trades']} trades)",
                    "params": {},
                })
                self.history = self.history[-25:]
                log.warning("LIVE-EVIDENCE DEMOTION: champion %s real PF %.2f over %d trades",
                            act["id"], lv["pf"], lv["trades"])

        # diversity restart: if the population has converged without finding a
        # champion for a long time, it's stuck in an overfit basin — re-inject
        # fresh explorers so it keeps searching instead of grinding the same region.
        self._since_improve = 0 if promoted else self._since_improve + 1
        if self.de.diversity() < 0.25 and self._since_improve >= STALL_REINJECT:
            k = self.de.reinject(0.4)
            self._since_improve = 0
            log.info("auto-tune: converged without a champion -> re-injected %d explorers", k)

        if self.cycles % VAULT_REVAL_EVERY == 0:
            await self._revalidate_vault(folds_cbs, specs_map, interval, taker,
                                         slip, strat, risk)

        # meta-labeling research: retrain the P(win) model on the basket's full
        # history every so often (walk-forward credentialed; persists only if
        # it beats the incumbent's held-out AUC). Runs on the research pool.
        if self.cycles % META_TRAIN_EVERY == 0 and (self._meta_task is None
                                                    or self._meta_task.done()):
            try:
                # train on a WIDER basket than we trade: the traded symbols
                # plus top-volume universe perps, up to 8 — the meta model's
                # features are all symbol-relative (ATR units, percentiles),
                # so pooled history means ~2x the samples and a sturdier AUC
                # without diluting what it learns.
                cbs = {}
                meta_syms = list(self._traded_symbols()[:4])
                for extra in self._universe():
                    if len(meta_syms) >= 8:
                        break
                    if extra not in meta_syms:
                        meta_syms.append(extra)
                for tsym in meta_syms:
                    try:
                        tc0 = await self._get_candles(tsym)
                    except Exception:  # noqa: BLE001 — basket breadth is best-effort
                        tc0 = None
                    if tc0 and len(tc0) >= MIN_BARS:
                        cbs[tsym] = tc0
                if not cbs:
                    self.last_meta = {"trained": False, "reason": "no cached history >= MIN_BARS",
                                      "ts": int(time.time() * 1000)}
                else:
                    # BACKGROUND: the training occupies one research worker for
                    # 35-70s. Awaiting it inline froze every 12th cycle — and
                    # the duty-cycle governor then stretched the following gap
                    # by the same stall again. The cycle moves on; the result
                    # lands in last_meta when the task finishes; a still-
                    # running task simply skips the next due slot.
                    import copy
                    self._meta_task = asyncio.create_task(
                        self._meta_train(cbs, interval, copy.deepcopy(strat),
                                         copy.deepcopy(risk)),
                        name="meta-train")
            except Exception as e:  # noqa: BLE001 — ML must never break tuning
                # ALWAYS leave a trace: a silent null in the snapshot hid five
                # straight failed trainings from an entire live session's resume.
                self.last_meta = {"trained": False, "error": f"{type(e).__name__}: {e}",
                                  "ts": int(time.time() * 1000)}
                log.warning("meta training failed: %s", e)

        self.last_cycle = {
            "ts": int(time.time() * 1000), "symbol": symbol,
            "generation": self.de.generation, "population": len(self.de.pop),
            "diversity": round(self.de.diversity(), 3), "folds": len(fold_fits),
            "research_cores": self.orch.research_workers,
            # best=None means NO candidate passed the profit veto this cycle —
            # surface null, not the -1e18 selection sentinel
            "champion_fitness": round(champ_fit, 3),
            "best_fitness": round(best_adj, 3) if best is not None else None,
            # promotion transparency: how close is anything to taking the seat?
            "pf_passed": pf_passed, "cands_judged": len(cands),
            "thin_rejected": thin_rejected, "co_symbol": co_sym or None,
            "gauntlet_blocked": gauntlet_blocked,
            "bar": round(bar, 3),   # the bar as actually applied, history raise included
            "promoted": promoted, "candidates": len(candidates),
            "vault_candidates": len(vault), "de_candidates": len(topk),
            "champion_source": (best["source"] if best else None),
            "research_symbol": symbol,
            "basket": [s for s, _ in basket],
            "clock": interval,
            # HOW MUCH EVIDENCE stood behind this verdict. A bar nothing can
            # reach and a floor nothing can clear both look identical from
            # outside — "no promotion again" — and both hid here for 618 cycles.
            # These are the numbers that tell the two apart, and the report's
            # self-check reads them.
            "oos_folds": nf_oos,
            "oos_traded_bars": max(0, min(len(v) for v in folds_cbs[0].values()) - 300),
            "champ_oos_trades": champ_trades,
            "champ_oos_return": round(champ_ret, 4),
            "best_oos_trades": int(best_pool.get("trades", 0) or 0) if best else None,
            "best_oos_return": (round(float(best_pool.get("total_return", 0.0) or 0.0), 4)
                                if best else None),
            "gauntlet": ({"median": gaunt["median"], "pf_ge1": gaunt["pf_ge1"],
                          "n": gaunt["n"], "weak": gaunt["weak"]}
                         if gaunt else None),
        }
        if self.orch._notify:
            await self.orch._notify("autotune")
        return promoted

    async def _meta_train(self, cbs: dict, interval: str, strat, risk) -> None:
        """The meta-model retrain, off the cycle's critical path. Snapshot
        configs travel with the task so a mid-training promotion can't mutate
        the labeler's inputs under it."""
        try:
            from ..ml.meta import train_from_candles
            try:
                res = (await self.orch.map_cpu(train_from_candles,
                                               [(cbs, interval, strat, risk)],
                                               research=True))[0]
            except Exception as e:  # noqa: BLE001 — a pool hiccup must not cost the model
                log.warning("meta training on the research pool failed (%s) — in-process retry", e)
                res = await asyncio.to_thread(train_from_candles, cbs, interval, strat, risk)
            self.last_meta = {**res, "ts": int(time.time() * 1000)}
            log.info("meta-model training: %s", res)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            self.last_meta = {"trained": False, "error": f"{type(e).__name__}: {e}",
                              "ts": int(time.time() * 1000)}
            log.warning("meta training failed: %s", e)

    async def _gauntlet(self, params: dict, interval: str, taker: float, slip: float,
                        strat, risk) -> dict | None:
        """Regime stress-test: run one parameter set through the shared-account
        portfolio simulator over five years of Binance regime eras (free
        data.binance.vision archive, disk-cached forever — a finished month
        never changes), with BingX fee assumptions. Memoized per (params,
        interval, window): a champion is only ever gauntleted once per era.
        Returns a summary dict, or None when fewer than 3 eras have data
        (offline, unlisted symbol, first run without internet)."""
        from ..data.binance_hist import GAUNTLET_WINDOWS, load_window

        def _psig(p: dict) -> tuple:
            return tuple(sorted((k, round(float(v), 10)) for k, v in p.items()))

        syms = [s for s in self._traded_symbols()[:2]] or ["BTC-USDT"]
        sig = _psig(params)
        results: dict[str, dict] = {}
        args, keys = [], []
        for name, months in GAUNTLET_WINDOWS:
            ck = (sig, interval, name)
            hit = self._gauntlet_cache.get(ck)
            if hit is not None:
                results[name] = hit
                continue
            cbs = await load_window(syms, interval, months)
            if not cbs:
                continue
            specs_map = {s: self.orch.specs.get(s, ContractSpec(s)) for s in cbs}
            # meta-free: applying TODAY'S meta model to a 2021 era would mix
            # impossible hindsight into the score — eras judge the core brain
            args.append((params, cbs, interval, specs_map, taker, slip, strat, risk, False))
            keys.append((name, ck, cbs))
        if args:
            res = await self.orch.map_cpu(validate_params_portfolio, args, research=True)
            for (name, ck, cbs), r in zip(keys, res):
                st = r.get("stats", {}) or {}
                entry = {"fit": round(float(r.get("fitness", -1.0)), 3),
                         "pf": round(float(st.get("profit_factor", 0.0) or 0.0), 3),
                         "trades": int(st.get("trades", 0) or 0),
                         # the era's own trades, bucketed by the market each was
                         # opened into — this is what the profile is built from
                         "by_regime": st.get("by_regime") or {},
                         # ...and the era's overall character, kept as readable
                         # context in the vault. It decides nothing.
                         "regime": await self._era_regime(name, interval, cbs)}
                self._gauntlet_cache[ck] = entry
                results[name] = entry
        if len(results) < 3:
            return None
        fits = [r["fit"] for r in results.values()]
        pf_ge1 = sum(1 for r in results.values() if r["pf"] >= 1.0)
        med = statistics.median(fits)
        # PER-REGIME PROFILE — the part the live engine actually uses. This set's
        # own trades, pooled across five years and bucketed by the market each
        # was opened into, turn "it died in 2022" from a mark against it into an
        # instruction: stand down when the tape looks like that again. Losing
        # money in a crash is not a defect in a trend set, it is a description
        # of when that set should be flat.
        by_regime = build_profile({n: {**r, "name": n} for n, r in results.items()})
        return {"n": len(results), "median": round(med, 3), "worst": round(min(fits), 3),
                "pf_ge1": pf_ge1, "weak": med < 0.0, "windows": results,
                "by_regime": by_regime, "symbols": syms, "meta_free": True}

    async def _era_regime(self, name: str, interval: str, cbs: dict) -> str | None:
        """What KIND of market an era was, by the live classifier.

        Independent of any parameter set and of immutable history, so it is
        computed once per (era, interval) and reused for every champion the
        gauntlet ever runs — the classification pass is a bar loop over three
        months and would otherwise be paid once per candidate for no reason."""
        key = (name, interval)
        hit = self._era_regime_cache.get(key)
        if hit is not None:
            return hit[0]
        # the first (largest) series is representative: the eras are named for
        # the market as a whole, and the gauntlet's symbols are correlated majors
        series = max(cbs.values(), key=len) if cbs else []
        try:
            shares = await asyncio.to_thread(classify_era, series, interval)
        except Exception as e:  # noqa: BLE001 — a label, never a dependency
            log.debug("era classification failed for %s: %s", name, e)
            return None
        reg = dominant_regime(shares)
        self._era_regime_cache[key] = (reg, shares)
        return reg

    async def _trial_cycle(self) -> None:
        """One research cycle on the TRIAL clock (clock_trial setting): its own
        gene pool and candle cache, the same fold machinery and OOS standards —
        but lean: no specialists, no meta training, no vault revalidation, and
        its champions are TAGGED with their clock and never applied to the live
        engine. The point is a fair, continuously-updated answer to 'which bar
        clock earns more OOS?', visible in the tuner panel and the vault."""
        cfg = self.orch.cfg
        interval = cfg.strategy.trial_interval
        strat, risk = cfg.strategy, cfg.risk
        syms = self._traded_symbols()[:2] or list(cfg.symbols)[:2]
        if not syms:
            return
        symbol = syms[0]
        candles = await self._get_candles(symbol, interval=interval, cache=self._trial_cache)
        if len(candles) < MIN_BARS:
            self.last_trial = {"ts": int(time.time() * 1000), "clock": interval,
                               "note": f"insufficient data ({len(candles)} bars)"}
            return
        spec = self.orch.specs.get(symbol, ContractSpec(symbol))
        taker, slip = spec.taker_fee, cfg.paper.slippage_bps

        if self.trial_de is None:
            self.trial_de = DEOptimizer(pop_size=POP_SIZE,
                                        state_path=STATE_PATH.with_name("tuner_state_trial.json"))
            if not self.trial_de.load():
                self.trial_de.seed_population(_current_params(cfg))
        de = self.trial_de

        # THE SAME SPLIT THE PRIMARY CLOCK USES. This was a hard-coded 0.75 while
        # the judge below builds its folds from OOS_TAIL_FRAC (0.40), i.e. from
        # 60% of the series onward — so the trial DE was fitting bars up to 75%
        # and then being scored on folds that began at 60%. The first fold sat
        # entirely inside its own training data and the second half-way in, with
        # no purge gap either: precisely the leak the primary clock's comment
        # describes as fixed, still live on the clock whose whole job is to
        # answer "which bar size earns more out-of-sample". Its champions are
        # tagged and kept in the vault for the day the user switches interval,
        # so the inflated numbers were not harmless.
        train = _train_split(candles)
        max_folds_by_data = max(1, len(train) // MIN_FOLD_BARS)
        nf = int(clamp(min(self.orch.research_workers, max_folds_by_data), 1, 8))
        folds = _make_folds(train, nf)
        trials = de.trials()
        # same steady-state economy as the primary clock: members keep their
        # fitness on unchanged folds — only trials are scored, halving the work
        data_ts = self._trial_cache.get(symbol, (None, 0.0))[1]
        need_members = (self._trial_scored_ts != data_ts) or any(f <= -1e8 for f in de.fitness)
        self._trial_scored_ts = data_ts
        candidates = (list(de.pop) + trials) if need_members else list(trials)
        args = [(fold, symbol, interval, spec, taker, slip, strat, risk, candidates) for fold in folds]
        fold_fits = await self.orch.map_cpu(score_fold, args, research=True)
        fold_fits = [ff for ff in fold_fits if ff and len(ff) == len(candidates)]
        if not fold_fits:
            return
        w = recency_weights(len(fold_fits))
        robust = [robust_aggregate(list(fc), w) for fc in zip(*fold_fits)]
        p = len(de.pop)
        if need_members:
            member_fit, trial_fit = robust[:p], robust[p:p + len(trials)]
        else:
            member_fit, trial_fit = list(de.fitness), robust[:len(trials)]
        de.select(trials, trial_fit, member_fit)
        de.save()

        # OOS judging on the trial clock's own basket folds — same standards
        # as the live clock (portfolio fitness, median+worst composite, PF>=1
        # on the newest fold), so the two clocks' numbers are comparable.
        cbs_full: dict[str, list] = {}
        for tsym in syms:
            try:
                tc = await self._get_candles(tsym, interval=interval, cache=self._trial_cache)
                if len(tc) >= MIN_BARS:
                    cbs_full[tsym] = tc
            except Exception as e:  # noqa: BLE001
                log.debug("trial basket %s: %s", tsym, e)
        if not cbs_full:
            return
        folds_cbs = portfolio_folds(cbs_full, k=_oos_fold_count(cbs_full),
                                    tail_frac=OOS_TAIL_FRAC)
        if not folds_cbs:
            return
        specs_map = {s: self.orch.specs.get(s, ContractSpec(s)) for s in cbs_full}
        topk = de.top_k(3)
        # meta-free: the GBM head is trained on the LIVE clock's labels —
        # blending it (at weight ~0.85) into another clock's validation would
        # score the mismatch, not the parameters. The trial judges CORE brains.
        vargs = [(prm, fc, interval, specs_map, taker, slip, strat, risk, False)
                 for prm, _ in topk for fc in folds_cbs]
        vres = await self.orch.map_cpu(validate_params_portfolio, vargs, research=True)
        nfo = len(folds_cbs)
        best_fit, best_params, best_stats = None, None, {}
        for i, (prm, _tf) in enumerate(topk):
            rs = vres[i * nfo:(i + 1) * nfo]
            fit = _oos_composite([r["fitness"] for r in rs])
            stats0 = rs[-1]["stats"]
            # same pooled economics the live clock admits on, so a trial
            # champion in the vault means what a live one means
            pool = _pool_stats([r["stats"] for r in rs])
            admitted = (int(pool.get("trades", 0) or 0) >= MIN_POOLED_TRADES
                        and float(pool.get("profit_factor", 0.0) or 0.0) >= 1.0
                        and float(pool.get("total_return", 0.0) or 0.0) > 0.0)
            if (best_fit is None or fit > best_fit) and admitted:
                best_fit, best_params, best_stats = fit, prm, stats0
        recorded = None
        if best_params is not None and best_fit is not None:
            # Admitted on the same POOLED ECONOMICS as the live clock (checked
            # above) — the vault only ever holds sets that made money on real
            # evidence. There is no absolute fitness floor here either: only 6%
            # of genuinely profitable sets could reach one. Tagged with the
            # trial clock,
            # they become instant candidates the day the user switches
            # interval. Full newest-fold stats travel with the record: the old
            # pf-only dict made the vault show wr 0% / 0 trades / PF 999 —
            # numbers that read like a bug instead of like evidence.
            recorded = self.orch.record_champion(best_params, best_fit,
                                                 best_stats, clock=interval)
        self.last_trial = {
            "ts": int(time.time() * 1000), "clock": interval,
            "generation": de.generation, "population": len(de.pop),
            "best_fitness": round(best_fit, 3) if best_fit is not None else None,
            "recorded": recorded, "basket": list(cbs_full), "meta_free": True,
        }
        # the live half of the trial: (re)start or hot-swap the shadow paper
        # account the moment a (better) trial-clock champion exists.
        try:
            await self.orch.maybe_refresh_shadow()
        except Exception as e:  # noqa: BLE001 — the shadow must never break tuning
            log.warning("shadow refresh failed: %s", e)
        if self.orch._notify:
            await self.orch._notify("autotune")

    async def _revalidate_vault(self, folds_cbs, specs_map, interval, taker, slip,
                                strat, risk) -> None:
        """Re-score every saved champion with THE SAME JUDGE that promotes, and
        refresh its CURRENT evaluation (shown next to what it was born at).

        It used to run a different one. The promotion gate scores the shared-
        account PORTFOLIO across purged OOS folds and blends median with worst;
        this pass ran the SINGLE-SYMBOL validator over one continuous recent
        window and took a plain mean across the basket. Both wrote the same
        `fitness` field, so a champion's headline number meant whichever judge
        had touched it last — on a live board, 0.159 from this path sitting
        beside -0.367 from the gate, for the same parameter set in the same
        cycle.

        That is not only confusing to read. `prune_champions` ranks the vault by
        `fitness`, and the stand-down picks its fallback from champions whose
        `fitness` is positive — so which champions survived, and which one the
        account fell back to, depended on which judge had last run. One judge,
        one scale, one meaning.

        Cheap now, too: the folds are memoized for the window, so any champion
        already judged as a candidate this cycle costs nothing to re-read."""
        vault = [c for c in self.orch.champions
                 # other-clock champions are skipped: scoring a 5m-born set on
                 # 15m folds would overwrite its honest evaluation with nonsense
                 if (c.get("clock") or interval) == interval and c.get("params")]
        if not vault or not folds_cbs:
            return
        nf = len(folds_cbs)

        def _psig(params: dict) -> tuple:
            return tuple(sorted((k, round(float(v), 10)) for k, v in params.items()))

        keys, miss_args, miss_keys = [], [], []
        for c in vault:
            sig = _psig(c["params"])
            for fi, fc in enumerate(folds_cbs):
                key = (sig, fi)
                keys.append(key)
                if key not in self._val_cache and key not in miss_keys:
                    miss_args.append((c["params"], fc, interval, specs_map,
                                      taker, slip, strat, risk))
                    miss_keys.append(key)
        if miss_args:
            res = await self.orch.map_cpu(validate_params_portfolio, miss_args, research=True)
            for key, r in zip(miss_keys, res):
                self._val_cache[key] = r
        for i, c in enumerate(vault):
            rs = [self._val_cache[k] for k in keys[i * nf:(i + 1) * nf]]
            fit = _oos_composite([r["fitness"] for r in rs])
            self.orch.set_champion_current(
                c["id"], fit, rs[-1].get("stats", {}),
                _pool_stats([r.get("stats", {}) for r in rs]))
        self.orch.prune_champions()
        log.info("vault revalidated on %d purged OOS folds: %d champions",
                 nf, len(self.orch.champions))

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
            "research_symbol": self.research_symbol,
            "next_run_ts": int(self.next_run_ts * 1000),
            "duty": round(clamp(float(getattr(self.orch.cfg.strategy, "research_duty",
                                              DUTY_CYCLE) or DUTY_CYCLE), 0.10, 0.50), 2),
            "last_cycle": self.last_cycle,
            "last_trial": self.last_trial,
            "clock_trial": self.orch.cfg.strategy.clock_trial,
            "meta": self._meta_status(),
            "history": self.history[-12:][::-1],
        }

    def _meta_status(self) -> dict:
        try:
            from ..ml.meta import get_meta
            m = get_meta()
            if m is None:
                return {"model": None, "last_training": self.last_meta}
            return {"model": {"auc": round(m.auc, 4), "n": m.n, "ready": m.ready,
                              "weight": round(m.blend_weight, 3),
                              "age_h": round((time.time() - m.trained_ts) / 3600, 1)},
                    "last_training": self.last_meta}
        except Exception as e:  # noqa: BLE001
            return {"model": None, "error": str(e)}


class _NullJob:
    progress = 0.0
