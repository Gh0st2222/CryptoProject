"""Vectorized indicators. Every function returns an array aligned to the
input (NaN where the window is not yet full)."""
from __future__ import annotations

import numpy as np


def ema(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float64)
    if len(x) == 0:
        return out
    alpha = 2.0 / (n + 1.0)
    acc = x[0]
    out[0] = acc
    for i in range(1, len(x)):
        acc += alpha * (x[i] - acc)
        out[i] = acc
    return out


def sma(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float64)
    if len(x) >= n:
        c = np.cumsum(np.insert(x, 0, 0.0))
        out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def rolling_std(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float64)
    if len(x) >= n:
        w = np.lib.stride_tricks.sliding_window_view(x, n)
        out[n - 1:] = w.std(axis=1, ddof=0)
    return out


def rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    out = np.full_like(close, np.nan, dtype=np.float64)
    if len(close) <= n:
        return out
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag = np.empty_like(delta)
    al = np.empty_like(delta)
    ag[n - 1] = gain[:n].mean()
    al[n - 1] = loss[:n].mean()
    for i in range(n, len(delta)):
        ag[i] = (ag[i - 1] * (n - 1) + gain[i]) / n
        al[i] = (al[i - 1] * (n - 1) + loss[i]) / n
    rs = ag[n - 1:] / np.maximum(al[n - 1:], 1e-12)
    out[n:] = 100.0 - 100.0 / (1.0 + rs)
    return out


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    tr = np.empty_like(high)
    tr[0] = high[0] - low[0]
    if len(high) > 1:
        pc = close[:-1]
        tr[1:] = np.maximum.reduce([high[1:] - low[1:], np.abs(high[1:] - pc), np.abs(low[1:] - pc)])
    return tr


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    return ema(true_range(high, low, close), 2 * n - 1)  # Wilder smoothing


def bollinger(close: np.ndarray, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    sd = rolling_std(close, n)
    up, dn = mid + k * sd, mid - k * sd
    width = (up - dn) / np.maximum(mid, 1e-12)
    return mid, up, dn, width


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    out = np.full_like(close, np.nan, dtype=np.float64)
    if len(close) <= n + 1:
        return out
    up = high[1:] - high[:-1]
    dn = low[:-1] - low[1:]
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(high, low, close)[1:]
    atr_s = ema(tr, 2 * n - 1)
    pdi = 100.0 * ema(plus_dm, 2 * n - 1) / np.maximum(atr_s, 1e-12)
    mdi = 100.0 * ema(minus_dm, 2 * n - 1) / np.maximum(atr_s, 1e-12)
    dx = 100.0 * np.abs(pdi - mdi) / np.maximum(pdi + mdi, 1e-12)
    out[1:] = ema(dx, 2 * n - 1)
    return out


def donchian(high: np.ndarray, low: np.ndarray, n: int = 20):
    hi = np.full_like(high, np.nan, dtype=np.float64)
    lo = np.full_like(low, np.nan, dtype=np.float64)
    if len(high) >= n:
        hi[n - 1:] = np.lib.stride_tricks.sliding_window_view(high, n).max(axis=1)
        lo[n - 1:] = np.lib.stride_tricks.sliding_window_view(low, n).min(axis=1)
    return hi, lo


def roc(close: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(close, np.nan, dtype=np.float64)
    if len(close) > n:
        out[n:] = close[n:] / np.maximum(close[:-n], 1e-12) - 1.0
    return out


def rolling_vwap(close: np.ndarray, volume: np.ndarray, n: int = 240) -> np.ndarray:
    out = np.full_like(close, np.nan, dtype=np.float64)
    if len(close) >= n:
        pv = np.cumsum(np.insert(close * volume, 0, 0.0))
        vv = np.cumsum(np.insert(volume, 0, 0.0))
        out[n - 1:] = (pv[n:] - pv[:-n]) / np.maximum(vv[n:] - vv[:-n], 1e-12)
    return out


def rolling_percentile_rank(x: np.ndarray, n: int) -> np.ndarray:
    """Rank of x[i] within its trailing n-window, in [0, 1]."""
    out = np.full_like(x, np.nan, dtype=np.float64)
    if len(x) >= n:
        w = np.lib.stride_tricks.sliding_window_view(x, n)
        out[n - 1:] = (w < w[:, -1:]).sum(axis=1) / (n - 1)
    return out


def zscore(x: np.ndarray, n: int) -> np.ndarray:
    m = sma(x, n)
    sd = rolling_std(x, n)
    return (x - m) / np.maximum(sd, 1e-12)
