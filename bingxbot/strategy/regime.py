"""Market regime classification from a feature row."""
from __future__ import annotations

import math

TREND_UP = "TREND_UP"
TREND_DOWN = "TREND_DOWN"
RANGE = "RANGE"
VOLATILE = "VOLATILE"

REGIMES = (TREND_UP, TREND_DOWN, RANGE, VOLATILE)

# How much each alpha matters per regime. Rows are learned truths of scalping:
# momentum works in trends, fading works in ranges, everything shrinks in chaos.
REGIME_ALPHA_MULT: dict[str, dict[str, float]] = {
    TREND_UP:   {"momentum": 1.35, "breakout": 1.2, "vwap_pullback": 1.15, "meanrev_bb": 0.45,
                 "rsi_fade": 0.45, "squeeze": 0.9, "obi": 1.0, "flow": 1.1},
    TREND_DOWN: {"momentum": 1.35, "breakout": 1.2, "vwap_pullback": 1.15, "meanrev_bb": 0.45,
                 "rsi_fade": 0.45, "squeeze": 0.9, "obi": 1.0, "flow": 1.1},
    RANGE:      {"momentum": 0.5, "breakout": 0.55, "vwap_pullback": 0.8, "meanrev_bb": 1.4,
                 "rsi_fade": 1.35, "squeeze": 1.1, "obi": 1.05, "flow": 0.9},
    VOLATILE:   {"momentum": 0.7, "breakout": 0.8, "vwap_pullback": 0.5, "meanrev_bb": 0.6,
                 "rsi_fade": 0.6, "squeeze": 0.4, "obi": 0.8, "flow": 0.9},
}

# Exit geometry adapts to regime as well.
REGIME_EXIT_MULT: dict[str, dict[str, float]] = {
    TREND_UP:   {"sl": 1.0, "tp": 1.25, "trail": 1.1},
    TREND_DOWN: {"sl": 1.0, "tp": 1.25, "trail": 1.1},
    RANGE:      {"sl": 0.9, "tp": 0.8, "trail": 0.85},
    VOLATILE:   {"sl": 1.3, "tp": 1.1, "trail": 1.3},
}


def detect_regime(row: dict[str, float]) -> tuple[str, float]:
    """Return (regime, confidence in [0,1])."""
    adx = row.get("adx", 0.0)
    slope = row.get("ema21_slope", 0.0)
    atr_pctile = row.get("atr_pctile", 0.5)
    e8, e21, e55 = row.get("ema_8", 0.0), row.get("ema_21", 0.0), row.get("ema_55", 0.0)

    if not all(map(math.isfinite, (adx, slope, atr_pctile, e8, e21, e55))):
        return RANGE, 0.0

    if atr_pctile > 0.88:
        return VOLATILE, min(1.0, (atr_pctile - 0.88) / 0.12 + 0.5)

    stacked_up = e8 > e21 > e55
    stacked_dn = e8 < e21 < e55
    trending = adx > 21
    if trending and (stacked_up or stacked_dn):
        conf = min(1.0, (adx - 21) / 20 + 0.4)
        return (TREND_UP if stacked_up else TREND_DOWN), conf
    if trending and abs(slope) > 1e-5:
        conf = min(1.0, (adx - 21) / 25 + 0.3)
        return (TREND_UP if slope > 0 else TREND_DOWN), conf
    conf = min(1.0, (21 - min(adx, 21)) / 21 + 0.2)
    return RANGE, conf
