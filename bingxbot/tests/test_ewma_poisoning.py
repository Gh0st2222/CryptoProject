"""One bad sample must not disable a live feature forever.

An EWMA is recursive, so a NaN does not pass through it — it STICKS:
`nan + alpha*(x - nan)` is nan for every sample after it, for the life of the
process. These carry the live microstructure (book imbalance, spread, CVD slope,
tick rate) into the alphas and the meta model, so absorbing one bad frame would
quietly disable the microstructure desk until a restart.
"""
import math

import pytest

from bingxbot.util import Ewma


def test_a_nan_does_not_stick():
    e = Ewma(0.2)
    for _ in range(20):
        e.update(1.0)
    good = e.get()
    assert e.update(float("nan")) == pytest.approx(good), "ignored, not absorbed"
    assert math.isfinite(e.get())
    e.update(1.0)
    assert math.isfinite(e.get()), "and the series carries on"


def test_infinity_does_not_stick_either():
    e = Ewma(0.2)
    e.update(2.0)
    e.update(float("inf"))
    e.update(float("-inf"))
    assert e.get() == pytest.approx(2.0)


def test_a_nan_before_any_good_sample_leaves_it_unset():
    e = Ewma(0.3)
    assert e.update(float("nan")) == 0.0
    assert e.get(default=7.0) == 7.0, "still lazily uninitialized"
    e.update(5.0)
    assert e.get() == pytest.approx(5.0), "first real sample seeds it"


def test_normal_behaviour_is_untouched():
    e = Ewma(0.5)
    assert e.update(10.0) == pytest.approx(10.0)      # lazy init
    assert e.update(20.0) == pytest.approx(15.0)
    assert e.update(20.0) == pytest.approx(17.5)


def test_the_live_micro_features_survive_a_bad_frame():
    """End to end in the shape the feed uses: poison every micro EWMA, then
    confirm the snapshot is still usable by the alphas."""
    from bingxbot.data.feed import MarketState
    st = MarketState("BTC-USDT")
    for ew in (st.obi, st.spread_bps, st.cvd_slope, st.ticks_per_s):
        ew.update(0.3)
        ew.update(float("nan"))
    snap = st.micro_snapshot()
    assert all(math.isfinite(v) for v in snap.values()), snap

    from bingxbot.strategy.alphas import ALPHAS
    assert all(math.isfinite(fn({}, snap, {})) for fn in ALPHAS.values())
