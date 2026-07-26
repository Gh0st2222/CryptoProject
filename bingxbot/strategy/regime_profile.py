"""What kind of market was that, and does this champion trade in it?

The regime gauntlet already scores a parameter set across five years of market
personalities. Until now the only thing done with that evidence was to raise the
promotion bar for a set that lost across the median era — a champion that could
not survive 2022 was treated as a slightly worse champion everywhere.

That is the wrong reading. Losing money in a crash is not a defect in a
trend-following set; it is a description of when that set should be flat. The
useful question is not "did it survive every era" but "which markets does it earn
in, and is this one of them". So an era's SCORE is filed under the era's
CHARACTER, and the profile becomes an operating manual: when the live tape looks
like a market this champion loses in, it stands down and waits, instead of being
rejected at birth or trading into weather it has already failed.

The classification runs through the SAME `detect_regime` the live brain uses on
every bar, over the era's own candles, so "TREND_DOWN" means one thing in 2022
and in the next ten minutes. Nothing here is a new model.
"""
from __future__ import annotations

import statistics

from .features import FeatureFrame
from .regime import REGIMES, detect_regime

# A regime needs at least this share of an era's bars before that era is filed
# under it. Below this the era is a mixture and naming it would be a fiction.
DOMINANT_SHARE = 0.34

# An era's fitness this far below zero means the set actually bled in that
# market — not that it merely underperformed. Only that earns a stand-down.
LOSING_ERA_FIT = -0.5

# ...and one bad era is an anecdote. A regime must be judged on at least this
# many eras before the live gate will act on it. With six eras in the gauntlet
# and four regimes, most regimes get one or two — so this stays at one and the
# magnitude threshold above carries the weight. Raising it makes the gate
# quieter, never wrong.
MIN_ERAS_PER_REGIME = 1


WARMUP = 300


def classify_era(candles: list, interval: str) -> dict[str, float]:
    """The share of an era's bars spent in each regime.

    Uses the live classifier on the live feature frame, so the labels are the
    same objects the trading gate compares against — not a parallel definition
    that could drift from it. An era's character does not depend on any
    parameter set, so this is computed once per era and cached forever: a
    finished month never changes its mind about what it was.
    """
    # deferred: engine imports strategy, so a module-level import here would
    # close the cycle. The frame builder lives on the engine side because the
    # backtester owns it; this is the one strategy-side caller.
    from ..engine.backtest import candles_to_arrays

    shares = {r: 0.0 for r in REGIMES}
    if not candles or len(candles) < WARMUP + 20:
        return shares
    ff = FeatureFrame(candles_to_arrays(candles), interval=interval)
    counted = 0
    # start past the indicator warmup: an unwarmed row has non-finite features,
    # which detect_regime answers with RANGE at zero confidence — counting those
    # would tilt every era toward the same label
    for i in range(WARMUP, ff.n):
        reg, _conf = detect_regime(ff.row_cached(i))
        if reg in shares:
            shares[reg] += 1.0
            counted += 1
    if counted:
        for r in shares:
            shares[r] /= counted
    return shares


def dominant_regime(shares: dict[str, float]) -> str | None:
    """The one regime an era is mostly made of, or None when it is a mixture."""
    if not shares:
        return None
    best = max(shares, key=lambda r: shares[r])
    return best if shares[best] >= DOMINANT_SHARE else None


def build_profile(era_results: dict[str, dict]) -> dict[str, dict]:
    """Fold per-era gauntlet results into a per-regime verdict.

    `era_results` maps era name -> {"fit": float, "pf": float, "regime": str}.
    Returns regime -> {"fit": median fitness, "eras": n, "losing": bool}.
    """
    by: dict[str, list[dict]] = {}
    for name, r in era_results.items():
        reg = r.get("regime")
        if reg in REGIMES:
            by.setdefault(reg, []).append(r)
    out: dict[str, dict] = {}
    for reg, rs in by.items():
        fits = [float(r.get("fit", 0.0) or 0.0) for r in rs]
        med = statistics.median(fits)
        out[reg] = {
            "fit": round(med, 3),
            "eras": len(rs),
            "names": sorted(r.get("name", "") for r in rs if r.get("name")),
            "losing": bool(med <= LOSING_ERA_FIT and len(rs) >= MIN_ERAS_PER_REGIME),
        }
    return out


def stands_down(profile: dict | None, regime: str) -> bool:
    """Should the champion holding this profile refuse new entries right now?

    Silent by default. No profile, no era for this regime, or a regime the set
    merely underperformed in are all reasons to keep trading — the gate only
    fires where history says this specific market cost this specific set real
    money. An absent verdict is not a bad one.
    """
    if not profile or not regime:
        return False
    entry = profile.get(regime)
    return bool(entry and entry.get("losing"))
