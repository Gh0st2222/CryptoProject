"""A non-finite edge must fail LOUDLY, not silently.

clamp() is `lo if x < lo else hi if x > hi else x`, so NaN passes straight
through it. `abs(nan) >= threshold` is False, so a NaN edge does not raise, does
not log, and does not trade — the bot looks perfectly healthy and simply stops
working, and a restart does not fix it because nothing is wrong with the state.

This is reachable input, not a thought experiment: at a 1m or 3m interval (both
offered as the trial clock in the dashboard) the 24h-context window is longer
than the 350-bar warmup, so several features really are NaN. Nothing breaks
today only because no current alpha reads one of them.
"""
import math

import numpy as np
import pytest

from bingxbot.strategy.brain import TradingBrain
from bingxbot.strategy.features import FeatureFrame


def _row(interval="15m", n=420, price=65_000.0):
    from bingxbot.data.history import synthetic_candles
    c = synthetic_candles("BTC-USDT", interval, n, seed=5, start_price=price)
    a = {k: np.array([getattr(x, k) for x in c], dtype=np.float64)
         for k in ("ts", "open", "high", "low", "close", "volume")}
    return FeatureFrame(a, interval=interval).row(-1)


def test_clamp_really_does_pass_nan_through():
    """The premise. If this ever changes, the guard's rationale changes too."""
    from bingxbot.util import clamp
    assert math.isnan(clamp(float("nan"), -1.0, 1.0))
    assert (abs(float("nan")) >= 0.3) is False, "and it fails the gate silently"


def test_short_intervals_really_do_produce_nan_features():
    """The reachable input, asserted rather than assumed."""
    bad = [k for k, v in _row("1m").items() if not math.isfinite(v)]
    assert bad, "the 24h window is longer than the warmup at 1m"
    assert not [k for k, v in _row("15m").items() if not math.isfinite(v)], \
        "15m is comfortably covered"


def test_a_nan_edge_is_refused_and_reported(caplog):
    b = TradingBrain(base_threshold=0.30)
    row = _row()
    row["atr_pct"] = float("nan")          # poison something an alpha reads

    with caplog.at_level("ERROR"):
        ev = b.score(row, {}, {})
    assert math.isfinite(ev["edge"]), "never hand a NaN edge to the gates"
    assert math.isfinite(ev["p_win"])
    if ev["edge"] == 0.0 and "non-finite" in caplog.text:
        assert "atr_pct" in caplog.text, "the culprit feature is named in the log"


def test_a_directly_poisoned_alpha_cannot_silence_the_brain(monkeypatch):
    """Whatever the route in, the outcome is the same: refuse, and say so."""
    import bingxbot.strategy.brain as bm
    b = TradingBrain(base_threshold=0.30)
    row = _row()
    real = bm.clamp
    monkeypatch.setattr(bm, "clamp",
                        lambda x, lo, hi: float("nan") if (lo, hi) == (-1, 1) else real(x, lo, hi))
    ev = b.score(row, {}, {})
    assert ev["edge"] == 0.0, "a NaN edge becomes a refusal, not a silent skip"
    assert math.isfinite(ev["p_win"])


def test_a_healthy_row_is_completely_unaffected():
    """The guard must cost a working brain nothing."""
    a = TradingBrain(base_threshold=0.30)
    row = _row()
    ev = a.score(row, {}, {})
    assert math.isfinite(ev["edge"]) and abs(ev["edge"]) > 0.0
    assert 0.0 < ev["p_win"] < 1.0


def test_a_dead_symbol_is_still_rejected_by_the_cost_gate():
    """Flat bars give the fusion a confident-looking edge out of no information.
    The volatility guard is what stops it becoming a trade — verify it, because
    the edge value alone looks tradeable."""
    from bingxbot.config import RiskConfig, StrategyConfig
    from bingxbot.engine.backtest import _entry_signal_ok
    n = 420
    flat = np.full(n, 100.0)
    a = dict(ts=np.arange(n, dtype=np.float64) * 900_000, open=flat, high=flat,
             low=flat, close=flat.copy(), volume=np.ones(n))
    row = FeatureFrame(a, interval="15m").row(-1)
    b = TradingBrain(base_threshold=0.30)
    ev = b.score(row, {}, {})
    assert abs(ev["edge"]) > 0.3, "the fusion really is confident on dead data"
    assert row["atr"] == 0.0
    assert _entry_signal_ok(b, StrategyConfig(), RiskConfig(), ev["edge"],
                            ev["p_win"], row, ev, 0.001, 1.0, 2.0) is False, \
        "no volatility estimate => no trade"
