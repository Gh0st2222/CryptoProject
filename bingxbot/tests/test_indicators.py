import numpy as np

from bingxbot.strategy import indicators as ta


def test_ema_converges_to_constant():
    x = np.full(200, 42.0)
    e = ta.ema(x, 20)
    assert abs(e[-1] - 42.0) < 1e-9


def test_rsi_extremes():
    up = np.cumsum(np.ones(100)) + 100.0
    dn = 200.0 - np.cumsum(np.ones(100))
    assert ta.rsi(up, 14)[-1] > 95
    assert ta.rsi(dn, 14)[-1] < 5


def test_atr_positive_and_scaled():
    rng = np.random.default_rng(7)
    c = 100 + np.cumsum(rng.normal(0, 1, 500))
    h = c + np.abs(rng.normal(0, 0.5, 500))
    l = c - np.abs(rng.normal(0, 0.5, 500))
    a = ta.atr(h, l, c, 14)
    assert np.all(a[50:] > 0)
    assert 0.3 < a[-1] < 6.0


def test_bollinger_contains_price_mostly():
    rng = np.random.default_rng(3)
    c = 100 + np.cumsum(rng.normal(0, 0.3, 800))
    _, up, dn, _ = ta.bollinger(c, 20, 2.0)
    valid = ~np.isnan(up)
    inside = np.mean((c[valid] <= up[valid]) & (c[valid] >= dn[valid]))
    assert inside > 0.82  # 2-sigma bands on an autocorrelated walk


def test_donchian_bounds():
    rng = np.random.default_rng(11)
    c = 100 + np.cumsum(rng.normal(0, 1, 300))
    h, l = c + 0.5, c - 0.5
    hi, lo = ta.donchian(h, l, 20)
    m = ~np.isnan(hi)
    assert np.all(hi[m] >= h[m] - 1e-9)
    assert np.all(lo[m] <= l[m] + 1e-9)


def test_rolling_percentile_rank_bounds():
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 500)
    r = ta.rolling_percentile_rank(x, 100)
    m = ~np.isnan(r)
    assert np.all((r[m] >= 0) & (r[m] <= 1))
