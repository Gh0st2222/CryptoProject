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
    TREND_UP:   {"trend": 1.4, "meanrev": 0.45, "micro": 1.05, "vol": 1.0, "carry": 0.9},
    TREND_DOWN: {"trend": 1.4, "meanrev": 0.45, "micro": 1.05, "vol": 1.0, "carry": 0.9},
    RANGE:      {"trend": 0.5, "meanrev": 1.4, "micro": 1.05, "vol": 1.05, "carry": 1.0},
    VOLATILE:   {"trend": 0.75, "meanrev": 0.6, "micro": 0.85, "vol": 0.7, "carry": 0.8},
}

# Exit geometry adapts to regime as well.
REGIME_EXIT_MULT: dict[str, dict[str, float]] = {
    TREND_UP:   {"sl": 1.0, "tp": 1.25, "trail": 1.1},
    TREND_DOWN: {"sl": 1.0, "tp": 1.25, "trail": 1.1},
    RANGE:      {"sl": 0.9, "tp": 0.8, "trail": 0.85},
    VOLATILE:   {"sl": 1.3, "tp": 1.1, "trail": 1.3},
}


def detect_regime(row: dict[str, float]) -> tuple[str, float]:
    """Return (regime, confidence in [0,1]) from ADX, EMA stack, trend quality
    (Kaufman efficiency ratio), multi-timeframe alignment and ATR percentile."""
    adx = row.get("adx", 0.0)
    slope = row.get("ema21_slope", 0.0)
    atr_pctile = row.get("atr_pctile", 0.5)
    er = row.get("eff_ratio", 0.0)
    align = row.get("mtf_align", 0.0)
    e8, e21, e55 = row.get("ema_8", 0.0), row.get("ema_21", 0.0), row.get("ema_55", 0.0)

    if not all(map(math.isfinite, (adx, slope, atr_pctile, er, align, e8, e21, e55))):
        return RANGE, 0.0

    if atr_pctile > 0.90:
        return VOLATILE, min(1.0, (atr_pctile - 0.90) / 0.10 + 0.5)

    stacked_up = e8 > e21 > e55
    stacked_dn = e8 < e21 < e55
    # A real trend: ADX up, clean efficiency ratio, and MTF agreement.
    trend_strength = (clamp01((adx - 18) / 22) + clamp01(er / 0.45) + clamp01(abs(align))) / 3.0
    trending = trend_strength > 0.42 and adx > 18

    if trending and (stacked_up or stacked_dn or abs(align) > 0.35):
        up = stacked_up or (align > 0 and not stacked_dn)
        return (TREND_UP if up else TREND_DOWN), min(1.0, trend_strength + 0.25)

    conf = clamp01((0.42 - trend_strength) / 0.42 + 0.2)
    return RANGE, conf


def clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x
