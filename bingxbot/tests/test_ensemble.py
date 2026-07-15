import numpy as np

from bingxbot.data.history import synthetic_candles
from bingxbot.engine.backtest import candles_to_arrays
from bingxbot.strategy import alphas as alpha_mod
from bingxbot.strategy.ensemble import AlphaEnsemble
from bingxbot.strategy.features import FeatureFrame

MICRO0 = {"obi": 0.0, "flow": 0.0, "cvd_slope": 0.0, "spread_bps": 1.0, "ticks_per_s": 0.0}


def _frame(seed=1, bars=2000):
    return FeatureFrame(candles_to_arrays(synthetic_candles("BTC-USDT", "1m", bars, seed=seed)))


def test_weights_always_normalized_and_floored():
    ff = _frame()
    ens = AlphaEnsemble(weight_floor=0.04)
    for i in range(300, ff.n):
        ens.evaluate(ff.row(i), MICRO0)
        s = sum(ens.weights.values())
        assert abs(s - 1.0) < 1e-6
        assert min(ens.weights.values()) >= 0.04 - 1e-9


def test_hedge_rewards_prescient_alpha(monkeypatch):
    """An alpha that literally peeks at the future must accumulate weight."""
    candles = synthetic_candles("BTC-USDT", "1m", 2500, seed=9)
    ff = FeatureFrame(candles_to_arrays(candles))
    closes = ff.f["close"]
    H = 5
    state = {"i": 0}

    def oracle(row, micro):
        i = state["i"]
        if i + H >= len(closes):
            return 0.0
        return 1.0 if closes[i + H] > closes[i] else -1.0

    monkeypatch.setitem(alpha_mod.ALPHAS, "squeeze", oracle)  # hijack one slot
    ens = AlphaEnsemble(horizon_bars=H, eta=0.5)
    for i in range(300, ff.n):
        state["i"] = i
        ens.evaluate(ff.row(i), MICRO0)
    ranked = sorted(ens.weights.items(), key=lambda kv: -kv[1])
    assert ranked[0][0] == "squeeze"
    assert ens.stats["squeeze"].hit_rate > 0.8


def test_scores_bounded_and_threshold_sane():
    ff = _frame(seed=4)
    ens = AlphaEnsemble()
    for i in range(300, ff.n):
        snap = ens.evaluate(ff.row(i), MICRO0)
        assert -1.0 <= snap["score"] <= 1.0
        assert 0.1 <= snap["threshold"] <= 0.95


def test_cost_gate_blocks_thin_edges():
    ens = AlphaEnsemble(base_threshold=0.3, threshold_adapt=False, cost_multiple=1.5)
    row = {"atr_pct": 0.0002, "close": 100.0}   # 2bps ATR: tiny expected move
    ok, why = ens.entry_ok(0.9, row, fees_roundtrip=0.001, spread_bps=1.0, slippage_bps=1.0)
    assert not ok and "cost" in why
    row_fat = {"atr_pct": 0.02, "close": 100.0}
    ok2, _ = ens.entry_ok(0.9, row_fat, fees_roundtrip=0.001, spread_bps=1.0, slippage_bps=1.0)
    assert ok2


def test_micro_veto():
    ens = AlphaEnsemble()
    assert not ens.micro_confirms(0.8, {"flow": -0.8, "obi": -0.5})
    assert ens.micro_confirms(0.8, {"flow": 0.5, "obi": 0.2})
    assert ens.micro_confirms(0.8, {"flow": 0.0, "obi": 0.0})
