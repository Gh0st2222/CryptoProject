"""Alpha signals. Each is a pure function
    (feature row, micro snapshot) -> score in [-1, 1]
where sign is direction and magnitude is conviction. Micro alphas read live
order-flow and return 0 when that data doesn't exist (historical backtests).
"""
from __future__ import annotations

import math

from ..util import clamp


def _fin(*vals: float) -> bool:
    return all(math.isfinite(v) for v in vals)


def alpha_momentum(row: dict, micro: dict) -> float:
    """EMA-stack momentum normalized by ATR, confirmed by medium-term ROC."""
    e8, e21, atr_ = row.get("ema_8", 0), row.get("ema_21", 0), row.get("atr", 0)
    roc12 = row.get("roc_12", 0)
    if not _fin(e8, e21, atr_, roc12) or atr_ <= 0:
        return 0.0
    raw = (e8 - e21) / (1.5 * atr_)
    confirm = clamp(roc12 / max(3.0 * row.get("atr_pct", 1e-9), 1e-9), -1, 1)
    s = math.tanh(raw) * (0.6 + 0.4 * abs(confirm)) * (1 if raw * confirm >= 0 else 0.4)
    return clamp(s, -1, 1)


def alpha_meanrev_bb(row: dict, micro: dict) -> float:
    """Fade Bollinger extremes: %B near 0 -> long, near 1 -> short."""
    pctb, rsi3 = row.get("bb_pctb", 0.5), row.get("rsi_3", 50)
    if not _fin(pctb, rsi3):
        return 0.0
    stretch = clamp((0.5 - pctb) * 2.5, -1, 1)          # >0 when below band mid
    kicker = clamp((50 - rsi3) / 50, -1, 1)             # oversold boosts longs
    s = stretch * (0.55 + 0.45 * abs(kicker)) if stretch * kicker >= 0 else stretch * 0.35
    return clamp(s, -1, 1)


def alpha_breakout(row: dict, micro: dict) -> float:
    """Donchian-channel breakout with volume confirmation."""
    c, hi, lo, atr_ = row.get("close", 0), row.get("dc_hi", 0), row.get("dc_lo", 0), row.get("atr", 0)
    vol_z = row.get("vol_z", 0)
    if not _fin(c, hi, lo, atr_) or atr_ <= 0:
        return 0.0
    vol_boost = 0.6 + 0.4 * clamp(vol_z / 2.5, 0, 1)
    if c >= hi - 0.1 * atr_:
        return clamp(((c - (hi - 0.1 * atr_)) / atr_ + 0.35) * vol_boost, 0, 1)
    if c <= lo + 0.1 * atr_:
        return -clamp((((lo + 0.1 * atr_) - c) / atr_ + 0.35) * vol_boost, 0, 1)
    return 0.0


def alpha_vwap_pullback(row: dict, micro: dict) -> float:
    """In an EMA-defined drift, buy pullbacks toward VWAP (trend-following
    entry timing rather than raw fading)."""
    c, vwap, e21, e55, atr_ = (row.get("close", 0), row.get("vwap", 0),
                               row.get("ema_21", 0), row.get("ema_55", 0), row.get("atr", 0))
    if not _fin(c, vwap, e21, e55, atr_) or atr_ <= 0 or vwap <= 0:
        return 0.0
    drift = 1 if e21 > e55 else -1 if e21 < e55 else 0
    if drift == 0:
        return 0.0
    gap = (c - vwap) / atr_
    if drift > 0 and -2.2 < gap < -0.3:
        return clamp(0.4 + (-gap - 0.3) * 0.45, 0, 1)
    if drift < 0 and 0.3 < gap < 2.2:
        return -clamp(0.4 + (gap - 0.3) * 0.45, 0, 1)
    return 0.0


def alpha_rsi_fade(row: dict, micro: dict) -> float:
    """Classic RSI(14) exhaustion fade, scaled continuously."""
    r = row.get("rsi_14", 50)
    if not _fin(r):
        return 0.0
    if r <= 32:
        return clamp((32 - r) / 22, 0, 1)
    if r >= 68:
        return -clamp((r - 68) / 22, 0, 1)
    return 0.0


def alpha_squeeze(row: dict, micro: dict) -> float:
    """Volatility-squeeze expansion: tight bands breaking with direction."""
    wp, roc3, atr_pct = row.get("bb_width_pctile", 0.5), row.get("roc_3", 0), row.get("atr_pct", 0)
    if not _fin(wp, roc3, atr_pct) or atr_pct <= 0:
        return 0.0
    if wp > 0.22:
        return 0.0
    tightness = (0.22 - wp) / 0.22
    push = clamp(roc3 / (2.0 * atr_pct), -1, 1)
    return clamp(tightness * push * 1.3, -1, 1)


def alpha_obi(row: dict, micro: dict) -> float:
    """Order-book imbalance (live only)."""
    obi = micro.get("obi", 0.0)
    return clamp(obi * 1.15, -1, 1) if abs(obi) > 0.12 else 0.0


def alpha_flow(row: dict, micro: dict) -> float:
    """Aggressor trade-flow imbalance + CVD slope agreement (live only)."""
    flow = micro.get("flow", 0.0)
    slope = micro.get("cvd_slope", 0.0)
    if abs(flow) < 0.1:
        return 0.0
    agree = 1.0 if flow * slope > 0 else 0.55
    return clamp(flow * agree * 1.2, -1, 1)


ALPHAS: dict[str, callable] = {
    "momentum": alpha_momentum,
    "meanrev_bb": alpha_meanrev_bb,
    "breakout": alpha_breakout,
    "vwap_pullback": alpha_vwap_pullback,
    "rsi_fade": alpha_rsi_fade,
    "squeeze": alpha_squeeze,
    "obi": alpha_obi,
    "flow": alpha_flow,
}

MICRO_ALPHAS = {"obi", "flow"}
