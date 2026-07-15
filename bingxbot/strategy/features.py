"""FeatureFrame: the firm's shared perception layer.

Every indicator series the desks read, computed vectorized over the candle
tail on each closed bar (live) or once over the whole dataset (backtest) — so
live and historical see identical numbers on identical data.

v2 adds multi-timeframe context (higher-TF trend built by resampling the base
series, no extra downloads), MACD/Stochastic/Keltner, Kaufman efficiency ratio
(trend quality), rolling regression slope, volatility-of-volatility, TTM-style
squeeze detection, and normalized VWAP deviation.
"""
from __future__ import annotations

import numpy as np

from . import indicators as ta

WARMUP_MIN = 260  # rows before this contain NaNs from the widest rolling window

# Higher-timeframe factors relative to the base bar (e.g. base 5m -> 20m, 60m).
MTF_MED = 4
MTF_HI = 12


def _broadcast(htf: np.ndarray, factor: int, n: int) -> np.ndarray:
    """Map a higher-TF series back onto base bars (base bar i -> HTF bar i//factor)."""
    idx = np.arange(n) // factor
    idx = np.clip(idx, 0, len(htf) - 1)
    return htf[idx]


class FeatureFrame:
    __slots__ = ("n", "f")

    def __init__(self, arrays: dict[str, np.ndarray]):
        o = arrays.get("open", arrays["close"])
        c, h, l, v = arrays["close"], arrays["high"], arrays["low"], arrays["volume"]
        self.n = len(c)
        f: dict[str, np.ndarray] = {"ts": arrays["ts"], "open": o, "close": c,
                                    "high": h, "low": l, "volume": v}

        # --- trend / moving averages
        f["ema_8"] = ta.ema(c, 8)
        f["ema_21"] = ta.ema(c, 21)
        f["ema_55"] = ta.ema(c, 55)
        f["ema_100"] = ta.ema(c, 100)
        f["ema21_slope"] = np.gradient(f["ema_21"]) / np.maximum(c, 1e-12)
        f["ema55_slope"] = np.gradient(f["ema_55"]) / np.maximum(c, 1e-12)
        f["linreg_slope"] = ta.linreg_slope(c, 20)
        f["eff_ratio"] = ta.efficiency_ratio(c, 20)          # Kaufman trend quality

        macd_line, macd_sig, macd_hist = ta.macd(c, 12, 26, 9)
        f["macd_hist"] = macd_hist / np.maximum(c, 1e-12)
        f["macd_line"] = macd_line / np.maximum(c, 1e-12)
        f["macd_rising"] = np.gradient(np.nan_to_num(macd_hist))

        # --- volatility
        f["atr"] = ta.atr(h, l, c, 14)
        f["atr_pct"] = f["atr"] / np.maximum(c, 1e-12)
        f["atr_pctile"] = ta.rolling_percentile_rank(np.nan_to_num(f["atr_pct"]), 240)
        f["vol_of_vol"] = ta.rolling_std(np.nan_to_num(f["atr_pct"]), 60)

        # --- oscillators
        f["rsi_14"] = ta.rsi(c, 14)
        f["rsi_7"] = ta.rsi(c, 7)
        f["rsi_3"] = ta.rsi(c, 3)
        stk, std = ta.stochastic(h, l, c, 14, 3)
        f["stoch_k"], f["stoch_d"] = stk, std
        f["adx"] = ta.adx(h, l, c, 14)

        # --- bands / channels
        bb_mid, bb_up, bb_dn, bb_w = ta.bollinger(c, 20, 2.0)
        f["bb_mid"], f["bb_up"], f["bb_dn"] = bb_mid, bb_up, bb_dn
        f["bb_pctb"] = (c - bb_dn) / np.maximum(bb_up - bb_dn, 1e-12)
        f["bb_width"] = bb_w
        f["bb_width_pctile"] = ta.rolling_percentile_rank(np.nan_to_num(bb_w), 240)
        kc_mid, kc_up, kc_dn = ta.keltner(h, l, c, 20, 1.5)
        # TTM squeeze: Bollinger bands inside Keltner => energy coiling
        f["squeeze_on"] = ((bb_up < kc_up) & (bb_dn > kc_dn)).astype(np.float64)
        dc_hi, dc_lo = ta.donchian(h, l, 34)
        f["dc_hi"], f["dc_lo"] = dc_hi, dc_lo
        f["dc_pos"] = (c - dc_lo) / np.maximum(dc_hi - dc_lo, 1e-12)   # 0..1 within range

        # --- momentum / returns
        f["roc_3"] = ta.roc(c, 3)
        f["roc_12"] = ta.roc(c, 12)
        roc3 = np.nan_to_num(f["roc_3"])
        f["roc_accel"] = roc3 - np.concatenate(([0.0], roc3[:-1]))
        f["ret_1"] = np.concatenate(([0.0], np.diff(c) / np.maximum(c[:-1], 1e-12)))

        # --- VWAP
        f["vwap"] = ta.rolling_vwap(c, v, 240)
        vwap_filled = np.nan_to_num(f["vwap"], nan=float(c.mean()) if self.n else 0.0)
        f["vwap_dev"] = (c - vwap_filled) / np.maximum(f["atr"], 1e-12)  # in ATR units
        f["vwap_z"] = ta.zscore(c - vwap_filled, 60)
        f["vol_z"] = ta.zscore(v, 96)

        # --- multi-timeframe context (resample base -> medium & high TF)
        self._add_mtf(f, o, h, l, c, v)

        self.f = f

    def _add_mtf(self, f, o, h, l, c, v) -> None:
        n = self.n
        for factor, tag in ((MTF_MED, "med"), (MTF_HI, "hi")):
            if n < factor * 12:
                f[f"htf_{tag}_slope"] = np.zeros(n)
                f[f"htf_{tag}_dir"] = np.zeros(n)
                continue
            _o, _h, _l, _c, _v = ta.resample_ohlc(f["ts"], o, h, l, c, v, factor)
            he = ta.ema(_c, 21)
            hslope = np.gradient(he) / np.maximum(_c, 1e-12)
            hrsi = ta.rsi(_c, 14)
            f[f"htf_{tag}_slope"] = _broadcast(np.nan_to_num(hslope), factor, n)
            f[f"htf_{tag}_dir"] = _broadcast(np.nan_to_num(np.tanh(hslope * 4000)), factor, n)
            f[f"htf_{tag}_rsi"] = _broadcast(np.nan_to_num(hrsi, nan=50.0), factor, n)
        # alignment across base + medium + high TF, in [-1, 1]
        base_dir = np.tanh(np.nan_to_num(f["linreg_slope"]) * 4000)
        f["mtf_align"] = (base_dir + f.get("htf_med_dir", np.zeros(n))
                          + f.get("htf_hi_dir", np.zeros(n))) / 3.0

    def row(self, i: int) -> dict[str, float]:
        if i < 0:
            i += self.n
        return {k: float(a[i]) for k, a in self.f.items()}

    def ready(self, i: int | None = None) -> bool:
        i = self.n - 1 if i is None else i
        return i >= WARMUP_MIN
