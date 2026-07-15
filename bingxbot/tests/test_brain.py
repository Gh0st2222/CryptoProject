import numpy as np

from bingxbot.data.history import synthetic_candles
from bingxbot.engine.backtest import candles_to_arrays
from bingxbot.strategy import alphas as alpha_mod
from bingxbot.strategy.alphas import DESKS
from bingxbot.strategy.brain import TradingBrain
from bingxbot.strategy.calibration import ProbabilityCalibrator
from bingxbot.strategy.features import FeatureFrame

MICRO0 = {"obi": 0.0, "flow": 0.0, "cvd_slope": 0.0, "spread_bps": 1.0, "ticks_per_s": 0.0}
CTX0 = {}


def _frame(seed=1, bars=2500):
    return FeatureFrame(candles_to_arrays(synthetic_candles("BTC-USDT", "5m", bars, seed=seed)))


def test_within_desk_weights_normalized_and_floored():
    ff = _frame()
    brain = TradingBrain(weight_floor=0.05)
    for i in range(300, ff.n):
        brain.evaluate(ff.row(i), MICRO0, CTX0)
    for desk, names in DESKS.items():
        total = sum(brain.alpha_w[n] for n in names)
        assert abs(total - 1.0) < 1e-6, f"{desk} weights must sum to 1"
        assert min(brain.alpha_w[n] for n in names) >= 0.05 - 1e-9


def test_edge_and_pwin_bounded():
    ff = _frame(seed=4)
    brain = TradingBrain()
    for i in range(300, ff.n):
        d = brain.evaluate(ff.row(i), MICRO0, CTX0)
        assert -1.0 <= d["edge"] <= 1.0
        assert 0.0 <= d["p_win"] <= 1.0
        assert 0.1 <= d["threshold"] <= 0.95


def test_hedge_rewards_prescient_alpha(monkeypatch):
    """An alpha that peeks at the future must accumulate weight within its desk."""
    candles = synthetic_candles("BTC-USDT", "5m", 3000, seed=9)
    ff = FeatureFrame(candles_to_arrays(candles))
    closes = ff.f["close"]
    H = 5
    state = {"i": 0}

    def oracle(row, micro, ctx):
        i = state["i"]
        if i + H >= len(closes):
            return 0.0
        return 1.0 if closes[i + H] > closes[i] else -1.0

    monkeypatch.setitem(alpha_mod.ALPHAS, "squeeze", oracle)  # squeeze is in the vol desk
    brain = TradingBrain(horizon_bars=H, eta=0.5)
    for i in range(300, ff.n):
        state["i"] = i
        brain.evaluate(ff.row(i), MICRO0, CTX0)
    vol_names = DESKS["vol"]
    best = max(vol_names, key=lambda n: brain.alpha_w[n])
    assert best == "squeeze"
    assert brain.alpha_stats["squeeze"].hit_rate > 0.8


def test_allocator_backs_the_winning_desk(monkeypatch):
    """A desk whose signal is always right should win allocation and no losing
    desk should dominate."""
    candles = synthetic_candles("BTC-USDT", "5m", 3500, seed=15)
    ff = FeatureFrame(candles_to_arrays(candles))
    closes = ff.f["close"]
    H = 5
    state = {"i": 0}

    def oracle(row, micro, ctx):
        i = state["i"]
        if i + H >= len(closes):
            return 0.0
        return 0.9 if closes[i + H] > closes[i] else -0.9

    # replace the whole trend desk with prescient copies
    for nm in DESKS["trend"]:
        monkeypatch.setitem(alpha_mod.ALPHAS, nm, oracle)
    brain = TradingBrain(horizon_bars=H)
    for i in range(300, ff.n):
        state["i"] = i
        brain.evaluate(ff.row(i), MICRO0, CTX0)
    w = brain.allocator.weights()
    assert w["trend"] == max(w.values()), f"trend should lead allocation, got {w}"


def test_calibrator_learns_direction():
    """Feed outcomes correlated with edge; Brier should beat the 0.25 baseline."""
    cal = ProbabilityCalibrator()
    rng = np.random.default_rng(0)
    for _ in range(1500):
        edge = rng.uniform(-1, 1)
        # true P(win) rises with |edge|
        p = 0.5 + 0.35 * abs(edge)
        won = rng.random() < p
        cal.update(edge, "RANGE", 0.5, won)
    assert cal.brier < 0.24, f"calibrator failed to learn (brier={cal.brier})"
    assert cal.predict(0.9, "RANGE", 0.5) > cal.predict(0.1, "RANGE", 0.5)


def test_kelly_sizing_monotone_and_bounded():
    brain = TradingBrain(kelly_fraction=0.3)
    # higher win prob -> bigger size; negative-edge bet -> zero
    lo = brain.kelly_size_mult(0.52, 1.2)
    hi = brain.kelly_size_mult(0.75, 1.2)
    assert 0 <= lo <= hi <= 1.75
    assert brain.kelly_size_mult(0.30, 1.0) == 0.0     # p*b < (1-p): no edge
    assert brain.kelly_size_mult(0.9, 2.0) <= 1.75     # hard cap holds


def test_cost_gate_and_pwin_gate_block():
    brain = TradingBrain(base_threshold=0.3, threshold_adapt=False, cost_multiple=1.5, min_p_win=0.55)
    row = {"atr_pct": 0.02, "close": 100.0}
    # good edge & prob but thin market blocked by cost only when atr tiny
    ok, why = brain.entry_ok(0.9, 0.6, {"atr_pct": 0.0002, "close": 100}, 0.001, 1.0, 1.0)
    assert not ok and "cost" in why
    ok2, _ = brain.entry_ok(0.9, 0.6, row, 0.001, 1.0, 1.0)
    assert ok2
    ok3, why3 = brain.entry_ok(0.9, 0.50, row, 0.001, 1.0, 1.0)  # p_win below min
    assert not ok3 and "win" in why3.lower()
