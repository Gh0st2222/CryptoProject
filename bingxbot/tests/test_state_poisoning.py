"""A saved brain state must never be able to silently kill the brain.

json.dumps writes NaN and Infinity as bare literals and json.loads reads them
straight back, so one non-finite number reaching the snapshot survives every
restart. A NaN alpha weight makes the fused edge NaN; `abs(nan) >= threshold`
is False, so the brain sits there looking perfectly healthy while never taking
another trade — and restarting, the one thing anybody would try, reloads the
very same NaN.

The rule: persistence may only install values the LIVE code could itself have
produced.
"""
import json
import math

import pytest

from bingxbot.strategy.brain import TradingBrain


def _brain():
    return TradingBrain(base_threshold=0.30)


def _state(**over):
    b = _brain()
    b.graded = 200
    d = b.state_dict()
    d.update(over)
    return d


def test_nan_round_trips_through_json_which_is_why_this_matters():
    """The premise, verified rather than assumed."""
    back = json.loads(json.dumps({"x": float("nan")}))
    assert math.isnan(back["x"]), "a NaN really does survive the snapshot"
    assert (abs(float("nan")) >= 0.3) is False, "and it silently fails the gate"


def test_a_poisoned_alpha_weight_is_refused():
    b = _brain()
    st = _state()
    name = next(iter(st["alpha_w"]))
    good = b.alpha_w[name]
    st["alpha_w"][name] = float("nan")
    assert b.load_state(json.loads(json.dumps(st))) is True
    assert math.isfinite(b.alpha_w[name]), "the NaN never lands"
    assert b.alpha_w[name] == good, "the alpha keeps its current weight"


def test_a_poisoned_beta_or_threshold_falls_back_inside_live_bounds():
    b = _brain()
    st = _state(beta=float("inf"), threshold=float("nan"))
    assert b.load_state(json.loads(json.dumps(st))) is True
    assert 0.3 <= b.beta <= 3.0, "beta stays in the range _grade enforces"
    assert b.base_threshold <= b.threshold <= 0.92, \
        "threshold stays in the range _adapt_threshold enforces"


def test_an_out_of_range_threshold_cannot_prise_the_gate_open():
    """The adaptive threshold may only ever TIGHTEN above base_threshold. A
    snapshot must not be a back door around that."""
    b = _brain()
    assert b.load_state(json.loads(json.dumps(_state(threshold=0.0)))) is True
    assert b.threshold >= b.base_threshold


def test_a_poisoned_calibrator_is_dropped_whole_not_in_part():
    """Its weights are a coupled vector: one bad entry makes every p_win it
    produces meaningless, so a partial restore is not a safe outcome."""
    b = _brain()
    st = _state()
    if not st.get("cal", {}).get("w"):
        pytest.skip("no calibrator weights in this build")
    before = list(b.calibrator.w)
    st["cal"]["w"] = [float("nan")] + list(st["cal"]["w"][1:])
    assert b.load_state(json.loads(json.dumps(st))) is True
    assert all(math.isfinite(x) for x in b.calibrator.w)
    assert list(b.calibrator.w) == before, "left untouched rather than half-restored"


def test_poisoned_score_history_is_filtered():
    """_adapt_threshold takes an order statistic of this history; a NaN in it
    corrupts the quantile and therefore the gate."""
    b = _brain()
    st = _state(score_hist=[0.1, float("nan"), 0.3, float("inf"), 0.5])
    assert b.load_state(json.loads(json.dumps(st))) is True
    assert all(math.isfinite(x) for x in b._score_hist)
    assert len(b._score_hist) == 3


def test_a_clean_state_still_round_trips_unchanged():
    """The guard must not cost a healthy restart its learning."""
    a = _brain()
    a.graded = 321
    a.beta = 1.7
    name = next(iter(a.alpha_w))
    a.alpha_w[name] = 0.4242
    a._score_hist.extend([0.11, 0.22, 0.33])

    b = _brain()
    assert b.load_state(json.loads(json.dumps(a.state_dict()))) is True
    assert b.graded == 321
    assert b.beta == pytest.approx(1.7)
    assert b.alpha_w[name] == pytest.approx(0.4242)
    assert list(b._score_hist)[-3:] == [0.11, 0.22, 0.33]
