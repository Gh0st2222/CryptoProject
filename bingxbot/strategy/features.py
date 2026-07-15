"""FeatureFrame: every indicator series the alphas need, computed vectorized.

Live mode recomputes over the candle tail on each closed bar (~2000 bars,
sub-millisecond). Backtests compute once over the whole dataset and then walk
row by row — identical numbers on identical data, by construction.
"""
from __future__ import annotations

import numpy as np

from . import indicators as ta

WARMUP_MIN = 260  # rows before this index contain NaNs from the widest window


class FeatureFrame:
    __slots__ = ("n", "f")

    def __init__(self, arrays: dict[str, np.ndarray]):
        c, h, l, v = arrays["close"], arrays["high"], arrays["low"], arrays["volume"]
        self.n = len(c)
        f: dict[str, np.ndarray] = {"ts": arrays["ts"], "close": c, "high": h, "low": l, "volume": v}

        f["ema_8"] = ta.ema(c, 8)
        f["ema_21"] = ta.ema(c, 21)
        f["ema_55"] = ta.ema(c, 55)
        f["ema21_slope"] = np.gradient(f["ema_21"]) / np.maximum(c, 1e-12)
        f["atr"] = ta.atr(h, l, c, 14)
        f["atr_pct"] = f["atr"] / np.maximum(c, 1e-12)
        f["atr_pctile"] = ta.rolling_percentile_rank(np.nan_to_num(f["atr_pct"]), 240)
        f["rsi_14"] = ta.rsi(c, 14)
        f["rsi_3"] = ta.rsi(c, 3)
        f["adx"] = ta.adx(h, l, c, 14)
        bb_mid, bb_up, bb_dn, bb_w = ta.bollinger(c, 20, 2.0)
        f["bb_mid"], f["bb_up"], f["bb_dn"] = bb_mid, bb_up, bb_dn
        f["bb_pctb"] = (c - bb_dn) / np.maximum(bb_up - bb_dn, 1e-12)
        f["bb_width_pctile"] = ta.rolling_percentile_rank(np.nan_to_num(bb_w), 240)
        dc_hi, dc_lo = ta.donchian(h, l, 34)
        f["dc_hi"], f["dc_lo"] = dc_hi, dc_lo
        f["roc_3"] = ta.roc(c, 3)
        f["roc_12"] = ta.roc(c, 12)
        f["vwap"] = ta.rolling_vwap(c, v, 240)
        f["vwap_z"] = ta.zscore(c - np.nan_to_num(f["vwap"], nan=c.mean() if self.n else 0.0), 60)
        f["vol_z"] = ta.zscore(v, 96)
        self.f = f

    def row(self, i: int) -> dict[str, float]:
        if i < 0:
            i += self.n
        return {k: float(a[i]) for k, a in self.f.items()}

    def ready(self, i: int | None = None) -> bool:
        i = self.n - 1 if i is None else i
        return i >= WARMUP_MIN
