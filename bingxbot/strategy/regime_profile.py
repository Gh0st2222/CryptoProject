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


MIN_REGIME_TRADES = 25   # below this there is no verdict, only an anecdote
LOSING_PF = 0.85         # ...and a verdict is only a CONVICTION when the set
                         # gave back real money: a profit factor this far under
                         # one, not merely under-performance.


def build_profile(era_results: dict[str, dict]) -> dict[str, dict]:
    """Fold the gauntlet's eras into a per-regime verdict, from the champion's
    OWN TRADES rather than from a handful of era scores.

    `era_results` maps era name -> {"fit", "pf", "regime", "by_regime"} where
    `by_regime` is the era's per-regime trade ledger. Trades pool across every
    era, so a regime's record is built from hundreds of fills spanning five
    years instead of a median over six numbers.

    This replaces labelling each era with one dominant regime. On live data that
    dropped four of six eras — "2023 chop" and "2025 range" named no regime at
    all, because a mixed market has no majority — and the two that survived were
    medians of two eras that cancelled each other out. The markets a champion is
    most likely to bleed in were precisely the ones producing no verdict.

    Returns regime -> {"pf", "trades", "eras", "names", "losing"}.
    """
    agg: dict[str, dict] = {}
    for name, r in era_results.items():
        for reg, b in (r.get("by_regime") or {}).items():
            if reg not in REGIMES:
                continue          # "UNKNOWN" and anything a future build adds
            a = agg.setdefault(reg, {"trades": 0, "gw": 0.0, "gl": 0.0,
                                     "pnl": 0.0, "names": set()})
            a["trades"] += int(b.get("trades", 0) or 0)
            a["gw"] += float(b.get("gross_win", 0.0) or 0.0)
            a["gl"] += float(b.get("gross_loss", 0.0) or 0.0)
            a["pnl"] += float(b.get("pnl", 0.0) or 0.0)
            if b.get("trades"):
                a["names"].add(name)
    out: dict[str, dict] = {}
    for reg, a in agg.items():
        pf = (a["gw"] / a["gl"]) if a["gl"] > 0 else (999.0 if a["gw"] > 0 else 0.0)
        enough = a["trades"] >= MIN_REGIME_TRADES
        out[reg] = {
            "pf": round(pf, 3),
            "pnl": round(a["pnl"], 4),
            "trades": a["trades"],
            "eras": len(a["names"]),
            "names": sorted(a["names"]),
            # a conviction needs BOTH: enough trades to mean something, and a
            # loss big enough to be a loss rather than a flat patch
            "losing": bool(enough and pf <= LOSING_PF and a["pnl"] < 0.0),
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
