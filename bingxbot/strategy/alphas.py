"""The analyst floor: ~18 alpha signals grouped into five desks.

Each alpha is a pure function
    (feature row, micro snapshot, context) -> score in [-1, 1]
where sign is direction and magnitude is conviction. Alphas return 0 when
their setup isn't present (event alphas) or their data source is absent
(microstructure/carry alphas in historical backtests).

ALPHA_META tags every alpha with its desk and whether it is `continuous`
(usually speaking) or `event` (fires only on a specific setup) so the UI can
show a dormant / armed / firing state instead of a dead "+0.00".
"""
from __future__ import annotations

import math

from ..util import clamp

CONT, EVENT = "continuous", "event"


def _fin(*vals: float) -> bool:
    return all(math.isfinite(v) for v in vals)


# ========================================================== TREND DESK

def alpha_momentum(row, micro, ctx) -> float:
    """EMA-stack momentum normalized by ATR, confirmed by medium-term ROC."""
    e8, e21, atr_ = row.get("ema_8", 0), row.get("ema_21", 0), row.get("atr", 0)
    roc12 = row.get("roc_12", 0)
    if not _fin(e8, e21, atr_, roc12) or atr_ <= 0:
        return 0.0
    raw = (e8 - e21) / (1.5 * atr_)
    confirm = clamp(roc12 / max(3.0 * row.get("atr_pct", 1e-9), 1e-9), -1, 1)
    s = math.tanh(raw) * (0.6 + 0.4 * abs(confirm)) * (1 if raw * confirm >= 0 else 0.4)
    return clamp(s, -1, 1)


def alpha_macd(row, micro, ctx) -> float:
    """MACD histogram sign + expansion (histogram accelerating)."""
    hist, rising, atr_pct = row.get("macd_hist", 0), row.get("macd_rising", 0), row.get("atr_pct", 0)
    if not _fin(hist, rising, atr_pct) or atr_pct <= 0:
        return 0.0
    body = clamp(hist / (0.6 * atr_pct), -1, 1)
    accel = 1.0 if hist * rising > 0 else 0.55       # histogram growing in its direction
    return clamp(body * accel, -1, 1)


def alpha_mtf_trend(row, micro, ctx) -> float:
    """Multi-timeframe trend alignment gated by trend quality (Kaufman ER)."""
    align, er = row.get("mtf_align", 0), row.get("eff_ratio", 0)
    if not _fin(align, er):
        return 0.0
    return clamp(align * (0.4 + 0.6 * clamp(er / 0.5, 0, 1)), -1, 1)


def alpha_breakout(row, micro, ctx) -> float:
    """Donchian-channel breakout with volume confirmation. [EVENT]"""
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


def alpha_roc_accel(row, micro, ctx) -> float:
    """Rate-of-change acceleration: momentum of momentum."""
    acc, atr_pct = row.get("roc_accel", 0), row.get("atr_pct", 0)
    if not _fin(acc, atr_pct) or atr_pct <= 0:
        return 0.0
    return clamp(acc / (1.2 * atr_pct), -1, 1)


# ========================================================== MEAN-REVERSION DESK

def alpha_meanrev_bb(row, micro, ctx) -> float:
    """Fade Bollinger extremes: %B near 0 -> long, near 1 -> short."""
    pctb, rsi3 = row.get("bb_pctb", 0.5), row.get("rsi_3", 50)
    if not _fin(pctb, rsi3):
        return 0.0
    stretch = clamp((0.5 - pctb) * 2.5, -1, 1)
    kicker = clamp((50 - rsi3) / 50, -1, 1)
    s = stretch * (0.55 + 0.45 * abs(kicker)) if stretch * kicker >= 0 else stretch * 0.35
    return clamp(s, -1, 1)


def alpha_capitulation(row, micro, ctx) -> float:
    """Liquidation-cascade reversion: a violent, huge-range bar on heavy volume
    whose extreme wick gets RECLAIMED by the close is forced liquidation
    exhausting itself — fade it. Pure OHLCV (fully backtestable); live funding /
    open-interest context boosts conviction when it confirms a squeeze. One of
    the few fast edges that survives retail latency: we are not racing anyone,
    we provide liquidity after the race ends. [EVENT]"""
    o, h, l, c = row.get("open", 0), row.get("high", 0), row.get("low", 0), row.get("close", 0)
    atr_, vol_z = row.get("atr", 0), row.get("vol_z", 0)
    if not _fin(o, h, l, c, atr_, vol_z) or atr_ <= 0:
        return 0.0
    rng = h - l
    if rng < 2.2 * atr_ or vol_z < 1.5:            # needs a violent, high-volume bar
        return 0.0
    lower, upper = min(o, c) - l, h - max(o, c)
    pos_in_bar = (c - l) / max(rng, 1e-12)
    strength = clamp((rng / atr_ - 2.2) / 2.0 + 0.4, 0.0, 1.0)
    if lower > 0.55 * rng and pos_in_bar > 0.6:    # flush down, close reclaimed -> long
        s = strength
    elif upper > 0.55 * rng and pos_in_bar < 0.4:  # squeeze up, close rejected -> short
        s = -strength
    else:
        return 0.0
    boost = 1.0
    fr = ctx.get("funding_rate")
    if fr is not None and math.isfinite(fr) and fr * s > 0 and abs(fr) > 0.0002:
        boost *= 1.25                              # crowded side just got flushed
    oi = ctx.get("oi_change_pct")
    if oi is not None and math.isfinite(oi) and oi < -0.005:
        boost *= 1.15                              # open interest actually left the market
    return clamp(s * boost, -1, 1)


def alpha_rsi_fade(row, micro, ctx) -> float:
    """RSI(14) exhaustion fade. [EVENT]"""
    r = row.get("rsi_14", 50)
    if not _fin(r):
        return 0.0
    if r <= 32:
        return clamp((32 - r) / 22, 0, 1)
    if r >= 68:
        return -clamp((r - 68) / 22, 0, 1)
    return 0.0


def alpha_stoch_fade(row, micro, ctx) -> float:
    """Stochastic overbought/oversold reversal with %K/%D cross bias. [EVENT]"""
    k, d = row.get("stoch_k", 50), row.get("stoch_d", 50)
    if not _fin(k, d):
        return 0.0
    if k <= 20:
        return clamp((20 - k) / 20 * (1.0 if k >= d else 0.6), 0, 1)
    if k >= 80:
        return -clamp((k - 80) / 20 * (1.0 if k <= d else 0.6), 0, 1)
    return 0.0


def alpha_vwap_revert(row, micro, ctx) -> float:
    """Fade extreme deviation from rolling VWAP (in ATR units)."""
    dev = row.get("vwap_dev", 0)
    if not _fin(dev):
        return 0.0
    if abs(dev) < 1.3:
        return 0.0
    return clamp(-dev / 3.0, -1, 1)          # far above VWAP -> short, far below -> long


def alpha_vwap_pullback(row, micro, ctx) -> float:
    """In an EMA-defined drift, buy pullbacks toward VWAP (trend entry timing)."""
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


# ========================================================== MICROSTRUCTURE DESK

def alpha_obi(row, micro, ctx) -> float:
    """Order-book imbalance, top 10 levels. [live]"""
    obi = micro.get("obi", 0.0)
    return clamp(obi * 1.15, -1, 1) if abs(obi) > 0.12 else 0.0


def alpha_flow(row, micro, ctx) -> float:
    """Aggressor trade-flow imbalance + CVD-slope agreement. [live]"""
    flow = micro.get("flow", 0.0)
    slope = micro.get("cvd_slope", 0.0)
    if abs(flow) < 0.1:
        return 0.0
    agree = 1.0 if flow * slope > 0 else 0.55
    return clamp(flow * agree * 1.2, -1, 1)


def alpha_cvd_trend(row, micro, ctx) -> float:
    """Cumulative-volume-delta slope as a standalone momentum-of-flow. [live]"""
    slope = micro.get("cvd_slope", 0.0)
    tps = micro.get("ticks_per_s", 0.0)
    if abs(slope) < 1e-9 or tps <= 0:
        return 0.0
    return clamp(math.tanh(slope * 2.5), -1, 1)


def alpha_spread_pressure(row, micro, ctx) -> float:
    """Tight spread + one-sided flow = clean directional pressure. [live]"""
    spread, flow, obi = micro.get("spread_bps", 99), micro.get("flow", 0), micro.get("obi", 0)
    if spread > 4.0:
        return 0.0
    lean = 0.6 * flow + 0.4 * obi
    if abs(lean) < 0.18:
        return 0.0
    tight = clamp((4.0 - spread) / 4.0, 0, 1)
    return clamp(lean * (0.5 + 0.5 * tight), -1, 1)


# ========================================================== VOLATILITY DESK

def alpha_squeeze(row, micro, ctx) -> float:
    """Volatility-squeeze expansion: tight bands breaking with direction. [EVENT]"""
    wp, roc3, atr_pct = row.get("bb_width_pctile", 0.5), row.get("roc_3", 0), row.get("atr_pct", 0)
    if not _fin(wp, roc3, atr_pct) or atr_pct <= 0:
        return 0.0
    if wp > 0.22:
        return 0.0
    tightness = (0.22 - wp) / 0.22
    push = clamp(roc3 / (2.0 * atr_pct), -1, 1)
    return clamp(tightness * push * 1.3, -1, 1)


def alpha_vol_breakout(row, micro, ctx) -> float:
    """TTM squeeze release: coiled energy (BB inside KC) firing directionally. [EVENT]"""
    on, roc3, atr_pct = row.get("squeeze_on", 0), row.get("roc_3", 0), row.get("atr_pct", 0)
    macd_h = row.get("macd_hist", 0)
    if not _fin(roc3, atr_pct, macd_h) or atr_pct <= 0:
        return 0.0
    # Fire as the squeeze is on/just releasing, in the MACD-confirmed direction.
    coiled = 0.7 if on > 0.5 else 0.3
    direction = clamp((roc3 / (2.0 * atr_pct)) * 0.6 + math.copysign(min(abs(macd_h) / (0.5 * atr_pct), 1), macd_h) * 0.4, -1, 1)
    return clamp(coiled * direction, -1, 1) if abs(direction) > 0.25 else 0.0


# ========================================================== CARRY DESK

def alpha_funding_skew(row, micro, ctx) -> float:
    """Contrarian funding: crowded longs (high +funding) pay to hold -> short
    bias, and vice versa. Fires only where funding data exists (live). [EVENT]"""
    fr = ctx.get("funding_rate")
    if fr is None or not math.isfinite(fr):
        return 0.0
    z = ctx.get("funding_z")
    signal = z if (z is not None and math.isfinite(z)) else fr / 0.0005
    if abs(signal) < 0.6:
        return 0.0
    return clamp(-signal / 2.0, -1, 1)


def alpha_oi_divergence(row, micro, ctx) -> float:
    """Open-interest / price agreement: fresh money confirming the move is
    continuation; OI bleeding into a move is exhaustion. [EVENT, live]"""
    oi_chg = ctx.get("oi_change_pct")
    if oi_chg is None or not math.isfinite(oi_chg):
        return 0.0
    ret = row.get("roc_3", 0)
    atr_pct = row.get("atr_pct", 1e-9)
    if abs(oi_chg) < 0.003 or atr_pct <= 0:
        return 0.0
    move = clamp(ret / (2.0 * atr_pct), -1, 1)
    if abs(move) < 0.2:
        return 0.0
    conf = clamp(abs(oi_chg) / 0.01, 0, 1)
    return clamp(move * conf * (1.0 if oi_chg > 0 else -0.5), -1, 1)


# ------------------------------------------------------------------ registry

ALPHAS: dict[str, callable] = {
    "momentum": alpha_momentum,
    "macd": alpha_macd,
    "mtf_trend": alpha_mtf_trend,
    "breakout": alpha_breakout,
    "roc_accel": alpha_roc_accel,
    "meanrev_bb": alpha_meanrev_bb,
    "capitulation": alpha_capitulation,
    "rsi_fade": alpha_rsi_fade,
    "stoch_fade": alpha_stoch_fade,
    "vwap_revert": alpha_vwap_revert,
    "vwap_pullback": alpha_vwap_pullback,
    "obi": alpha_obi,
    "flow": alpha_flow,
    "cvd_trend": alpha_cvd_trend,
    "spread_pressure": alpha_spread_pressure,
    "squeeze": alpha_squeeze,
    "vol_breakout": alpha_vol_breakout,
    "funding_skew": alpha_funding_skew,
    "oi_divergence": alpha_oi_divergence,
}

# desk, kind
ALPHA_META: dict[str, tuple[str, str]] = {
    "momentum": ("trend", CONT), "macd": ("trend", CONT),
    "mtf_trend": ("trend", CONT), "breakout": ("trend", EVENT),
    "roc_accel": ("trend", CONT),
    "meanrev_bb": ("meanrev", CONT), "capitulation": ("meanrev", EVENT),
    "rsi_fade": ("meanrev", EVENT),
    "stoch_fade": ("meanrev", EVENT), "vwap_revert": ("meanrev", EVENT),
    "vwap_pullback": ("meanrev", EVENT),
    "obi": ("micro", CONT), "flow": ("micro", CONT),
    "cvd_trend": ("micro", CONT), "spread_pressure": ("micro", EVENT),
    "squeeze": ("vol", EVENT), "vol_breakout": ("vol", EVENT),
    "funding_skew": ("carry", EVENT), "oi_divergence": ("carry", EVENT),
}

DESKS: dict[str, list[str]] = {}
for _name, (_desk, _kind) in ALPHA_META.items():
    DESKS.setdefault(_desk, []).append(_name)

DESK_ORDER = ["trend", "meanrev", "micro", "vol", "carry"]
MICRO_ALPHAS = {"obi", "flow", "cvd_trend", "spread_pressure"}
