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

# The multi-timeframe ladder (label, minutes). Rungs finer than the base bar are
# skipped (you can't see below your data); the rung equal to the base is the base
# series itself; coarser rungs are exact aggregations of the base bars — so a 1m
# base gives a true 1m / 5m / 15m / 1h view with no extra data pulled.
LADDER = (("1m", 1), ("5m", 5), ("15m", 15), ("1h", 60))


def _bdiff(x: np.ndarray) -> np.ndarray:
    """Backward difference: slope at i from bars i-1 -> i only. np.gradient's
    central difference reads bar i+1 — one bar of future — which both leaked
    into backtests and disagreed with the live edge row (where gradient falls
    back to a one-sided difference)."""
    return np.diff(x, prepend=x[:1]) if len(x) else x


def _causal_read_idx(ts: np.ndarray, bidx: np.ndarray, base_ms: int, bucket_ms: int,
                     n_buckets: int) -> np.ndarray:
    """For each base bar i, the index of the last higher-TF bucket whose value
    was fully known at bar i's close: the bar's own bucket when bar i is that
    bucket's final base bar (its close IS the bucket close), otherwise the
    previous bucket. Purely a function of timestamps, so live and backtest
    agree exactly — and no read ever includes bars beyond i (no lookahead)."""
    completes_bucket = ((ts + base_ms) % bucket_ms) == 0
    idx = np.where(completes_bucket, bidx, bidx - 1)
    return np.clip(idx, 0, max(n_buckets - 1, 0))


def _base_minutes(interval: str | None, ts: np.ndarray) -> int:
    if interval:
        from ..util import interval_ms
        return max(1, int(round(interval_ms(interval) / 60_000)))
    if len(ts) >= 3:
        d = int(np.median(np.diff(ts[-50:] if len(ts) > 50 else ts)))
        return max(1, int(round(d / 60_000)))
    return 1


def mtf_from_row(row: dict, ladder) -> dict:
    """Compact per-timeframe view assembled from a feature row — what the brain
    reads for cross-TF context and what the terminal shows for each rung."""
    out = {}
    for lab in ladder:
        d = row.get(f"tf_{lab}_dir")
        if d is None or not np.isfinite(d):
            continue
        out[lab] = {"dir": round(float(d), 3),
                    "rsi": round(float(row.get(f"tf_{lab}_rsi", 50.0)), 1),
                    "adx": round(float(row.get(f"tf_{lab}_adx", 0.0)), 1)}
    return out


class FeatureFrame:
    __slots__ = ("n", "f", "ladder", "_rows", "_alpha", "_kmats")

    def __init__(self, arrays: dict[str, np.ndarray], interval: str | None = None):
        self.ladder: list[str] = []
        self._rows = None    # lazy per-bar row-dict cache (candidate-invariant)
        self._alpha = None   # lazy per-bar alpha-score cache (set by the backtester)
        self._kmats = None   # lazy compiled-kernel matrices (features/alphas/regimes)
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
        f["ema21_slope"] = _bdiff(f["ema_21"]) / np.maximum(c, 1e-12)
        f["ema55_slope"] = _bdiff(f["ema_55"]) / np.maximum(c, 1e-12)
        f["linreg_slope"] = ta.linreg_slope(c, 20)
        f["eff_ratio"] = ta.efficiency_ratio(c, 20)          # Kaufman trend quality

        macd_line, macd_sig, macd_hist = ta.macd(c, 12, 26, 9)
        f["macd_hist"] = macd_hist / np.maximum(c, 1e-12)
        f["macd_line"] = macd_line / np.maximum(c, 1e-12)
        f["macd_rising"] = _bdiff(np.nan_to_num(macd_hist))

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

        # --- true multi-timeframe ladder (1m/5m/15m/1h relative to the base)
        self._add_mtf(f, o, h, l, c, v, _base_minutes(interval, arrays["ts"]))

        self.f = f

    def _add_mtf(self, f, o, h, l, c, v, base_min: int) -> None:
        """Build a real timeframe ladder: for each standard rung at/above the base
        bar, an EMA-stack + slope direction, RSI and ADX, mapped back onto base
        bars. Rungs are epoch-anchored aggregations of the base (real 5m/15m/1h
        buckets), and every base bar reads only the last rung bar that had fully
        CLOSED by that base bar's close — strictly causal, so the backtester and
        tuner can never peek at a higher-TF close that hasn't printed yet, and
        live sees exactly the same numbers on the same data."""
        n = self.n
        base_ms = base_min * 60_000
        base_dir = np.clip(np.tanh(np.nan_to_num(f["linreg_slope"]) * 4000), -1, 1)
        dirs = [base_dir]          # base counted once for alignment
        higher: list[np.ndarray] = []
        present: list[str] = []
        for label, mins in LADDER:
            if mins < base_min:
                continue           # can't see finer than the base data
            if mins == base_min:
                di = base_dir
                ri = np.nan_to_num(f["rsi_14"], nan=50.0)
                ai = np.nan_to_num(f["adx"])
                sl = np.nan_to_num(f["ema21_slope"])
            else:
                factor = max(2, int(round(mins / base_min)))
                if n < factor * 8:
                    continue
                bucket_ms = mins * 60_000
                _o, _h, _l, _c, _v, bidx = ta.resample_ohlc(f["ts"], o, h, l, c, v, bucket_ms)
                if len(_c) < 8:
                    continue
                he21, he55 = ta.ema(_c, 21), ta.ema(_c, 55)
                hslope = _bdiff(he21) / np.maximum(_c, 1e-12)
                stack = np.sign(_c - he21) + np.sign(he21 - he55)   # price/ema alignment
                hdir = 0.6 * np.tanh(hslope * 4000) + 0.4 * np.clip(stack / 2.0, -1, 1)
                read = _causal_read_idx(f["ts"], bidx, base_ms, bucket_ms, len(_c))
                di = np.clip(np.nan_to_num(hdir)[read], -1, 1)
                ri = np.nan_to_num(ta.rsi(_c, 14), nan=50.0)[read]
                ai = np.nan_to_num(ta.adx(_h, _l, _c, 14))[read]
                sl = np.nan_to_num(hslope)[read]
            f[f"tf_{label}_dir"], f[f"tf_{label}_rsi"] = di, ri
            f[f"tf_{label}_adx"], f[f"tf_{label}_slope"] = ai, sl
            present.append(label)
            if mins > base_min:
                dirs.append(di)
                higher.append(di)
        self.ladder = present
        f["mtf_align"] = np.clip(np.mean(np.vstack(dirs), axis=0), -1, 1) if len(dirs) > 1 else base_dir
        # consensus of the rungs strictly above the base — the trend backdrop
        f["mtf_bias"] = np.clip(np.mean(np.vstack(higher), axis=0), -1, 1) if higher else np.zeros(n)
        # legacy aliases so existing MTF alphas read the real ladder unchanged
        f["htf_med_dir"] = higher[0] if higher else np.zeros(n)
        f["htf_hi_dir"] = higher[-1] if higher else np.zeros(n)

    def row(self, i: int) -> dict[str, float]:
        if i < 0:
            i += self.n
        return {k: float(a[i]) for k, a in self.f.items()}

    def row_cached(self, i: int) -> dict[str, float]:
        """Same values as row(i), but built ONCE for every bar and reused.
        Rows depend on price only — never on strategy parameters — so a frame
        shared across tuner candidates pays this cost once instead of once per
        candidate. Callers must treat cached rows as read-only."""
        if self._rows is None:
            keys = list(self.f)
            cols = [self.f[k] for k in keys]
            self._rows = [dict(zip(keys, (float(c[j]) for c in cols)))
                          for j in range(self.n)]
        if i < 0:
            i += self.n
        return self._rows[i]

    def ready(self, i: int | None = None) -> bool:
        i = self.n - 1 if i is None else i
        return i >= WARMUP_MIN
