"""Market regime classification and per-desk suitability gating."""
from __future__ import annotations

import math

TREND_UP = "TREND_UP"
TREND_DOWN = "TREND_DOWN"
RANGE = "RANGE"
VOLATILE = "VOLATILE"

REGIMES = (TREND_UP, TREND_DOWN, RANGE, VOLATILE)

# How much each *desk* matters per regime. Trend desks lead trends, mean-rev
# desks lead ranges, microstructure works everywhere, everything shrinks in
# chaos. The meta-allocator adjusts on top of this from live performance.
REGIME_DESK_MULT: dict[str, dict[str, float]] = {
    TREND_UP:   {"trend": 1.5, "meanrev": 0.28, "micro": 1.0, "vol": 1.0, "carry": 0.9},
    TREND_DOWN: {"trend": 1.5, "meanrev": 0.28, "micro": 1.0, "vol": 1.0, "carry": 0.9},
    RANGE:      {"trend": 0.5, "meanrev": 1.4, "micro": 1.05, "vol": 1.05, "carry": 1.0},
    VOLATILE:   {"trend": 0.75, "meanrev": 0.6, "micro": 0.85, "vol": 0.7, "carry": 0.8},
}

# STRATEGY ARCHETYPES — what KIND of desk this champion is, as opposed to which
# constants it uses. Every champion until now was the same strategy with
# different numbers: the desk mix was learned online from a uniform start and
# reset with every brain, so nothing in the parameter set could say "I trade
# trends" or "I fade extremes". The tuner owns this choice like any other, and
# out-of-sample validation decides which identity actually pays on this board.
#
# It is a STANDING tilt applied at the same site as REGIME_DESK_MULT, so the
# three layers compose in the obvious way: the archetype says what this champion
# is, the regime says what today's market rewards, and the online allocator
# still moves weight toward whatever is actually working. Index 0 is exactly
# uniform, so every champion promoted before this existed behaves identically.
#
# Only desks the JUDGE can see are tilted. Historical klines carry no book, no
# tape and no funding, so the micro and carry desks are dormant in every
# backtest — an archetype leaning on them would be a knob the tuner selects at
# random because nothing out-of-sample can tell the settings apart.
# The archetype is a PREFERENCE; the regime is a fact about the market, and
# facts win. Every tilt below is deliberately weaker than the regime spread it
# competes with (RANGE already favours mean-reversion 1.4 : 0.5, a ratio of
# 2.8), so a trend-led champion is still outvoted by mean-reversion inside a
# range and a mean-reversion champion still cannot fade a decided trend. Making
# the archetype the stronger term would re-open exactly the failure the regime
# multipliers were introduced for — the allocator over-weighting mean-reversion
# into an uptrend — with the tuner able to select it.
DESK_TILT_NAMES = ("balanced", "trend-led", "meanrev-led", "volatility-led")
DESK_TILTS: tuple[dict[str, float], ...] = (
    {"trend": 1.00, "meanrev": 1.00, "micro": 1.00, "vol": 1.00, "carry": 1.00},
    {"trend": 1.45, "meanrev": 0.70, "micro": 1.00, "vol": 1.00, "carry": 1.00},
    {"trend": 0.70, "meanrev": 1.45, "micro": 1.00, "vol": 1.00, "carry": 1.00},
    {"trend": 0.90, "meanrev": 0.80, "micro": 1.00, "vol": 1.50, "carry": 1.00},
)


def desk_tilt_weights(idx) -> dict[str, float]:
    """The archetype's standing desk multipliers.

    Rounds rather than truncates, because `_coerce` rounds every int tunable and
    a hand-edited 1.7 must not mean archetype 1 here and archetype 2 to the rest
    of the pipeline. Anything unrecognisable falls back to balanced: this number
    arrives from a tuned parameter set and from champion JSON on disk, and an
    unknown archetype must mean "no opinion", never an IndexError in the fusion
    loop or an out-of-bounds read in the compiled one."""
    try:
        f = float(idx)
    except (TypeError, ValueError):
        return DESK_TILTS[0]
    # written as a positive test so NaN (every comparison False) falls back too
    if not 0.0 <= f <= len(DESK_TILTS) - 1:
        return DESK_TILTS[0]
    return DESK_TILTS[int(round(f))]


# Exit geometry adapts to regime as well.
REGIME_EXIT_MULT: dict[str, dict[str, float]] = {
    TREND_UP:   {"sl": 1.0, "tp": 1.25, "trail": 1.1},
    TREND_DOWN: {"sl": 1.0, "tp": 1.25, "trail": 1.1},
    RANGE:      {"sl": 0.9, "tp": 0.8, "trail": 0.85},
    VOLATILE:   {"sl": 1.3, "tp": 1.1, "trail": 1.3},
}


def detect_regime(row: dict[str, float]) -> tuple[str, float]:
    """Return (regime, confidence in [0,1]) — now genuinely multi-timeframe.

    A trend is only declared when the base momentum AND the higher-timeframe
    consensus (15m/1h, via `mtf_bias`) agree, and the trend's DIRECTION is taken
    from that stable higher-TF consensus — so a shallow pullback on the base bar
    no longer flips the regime (which used to un-mute mean-reversion and let the
    account fade an uptrend). A strong higher-TF bias also keeps the regime
    'trending' through a base wobble."""
    adx = row.get("adx", 0.0)
    slope = row.get("ema21_slope", 0.0)
    atr_pctile = row.get("atr_pctile", 0.5)
    er = row.get("eff_ratio", 0.0)
    align = row.get("mtf_align", 0.0)
    bias = row.get("mtf_bias", 0.0)          # 15m/1h consensus (stable backdrop)
    e8, e21, e55 = row.get("ema_8", 0.0), row.get("ema_21", 0.0), row.get("ema_55", 0.0)

    if not all(map(math.isfinite, (adx, slope, atr_pctile, er, align, bias, e8, e21, e55))):
        return RANGE, 0.0

    if atr_pctile > 0.90:
        return VOLATILE, min(1.0, (atr_pctile - 0.90) / 0.10 + 0.5)

    stacked_up = e8 > e21 > e55
    stacked_dn = e8 < e21 < e55
    trend_strength = (clamp01((adx - 18) / 22) + clamp01(er / 0.45) + clamp01(abs(align))) / 3.0
    # higher timeframes must lean the same way for a trend to count...
    htf_agree = abs(bias) >= 0.25
    # ...but a strong higher-TF bias sustains the regime through a base wobble.
    trending = adx > 16 and htf_agree and (trend_strength > 0.42 or abs(bias) >= 0.45)

    if trending:
        # direction from the higher-TF consensus (stable), falling back to the
        # base EMA stack only when the higher-TF read is weak.
        if abs(bias) >= 0.2:
            up = bias > 0
        else:
            up = stacked_up or (align > 0 and not stacked_dn)
        return (TREND_UP if up else TREND_DOWN), min(1.0, trend_strength + 0.25)

    conf = clamp01((0.42 - trend_strength) / 0.42 + 0.2)
    return RANGE, conf


def clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x
