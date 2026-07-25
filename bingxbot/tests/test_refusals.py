"""The refusal ledger: measuring what the gates throw away.

Losing trades are the job; refusing PROFITABLE ones silently is the thing that
never shows up in any statistic the system used to keep."""
import pytest

from bingxbot.engine.refusals import RefusalLedger, gate_label


def test_grades_refusals_on_the_horizon_in_signal_direction():
    """A long signal refused at 100 that reaches 102 with ATR 1 scored +2 ATR;
    the identical move refused as a SHORT signal scored -2. Direction is the
    whole point — a gate that blocks correct shorts is as costly as one that
    blocks correct longs."""
    led = RefusalLedger(horizon=3)
    led.record("BTC", "EV floor", +1, 100.0, 1.0, idx=1)
    led.record("ETH", "EV floor", -1, 100.0, 1.0, idx=1)
    led.mature("BTC", 102.0, idx=2)          # not yet at the horizon
    assert led.snapshot(min_n=1)["gates"] == []
    led.mature("BTC", 102.0, idx=4)
    led.mature("ETH", 102.0, idx=4)
    rows = {r["gate"]: r for r in led.snapshot(min_n=1)["gates"]}
    assert rows["EV floor"]["refused"] == 2
    assert rows["EV floor"]["mean_move_atr"] == pytest.approx(0.0)  # +2 and -2
    assert rows["EV floor"]["win_rate"] == pytest.approx(0.5)


def test_separates_a_gate_that_saves_from_one_that_starves():
    """The whole reason the ledger exists: two gates, identical refusal counts,
    opposite verdicts."""
    led = RefusalLedger(horizon=1)
    for i in range(10):
        led.record("BTC", "saves", +1, 100.0, 1.0, idx=i * 10)
        led.mature("BTC", 99.0, idx=i * 10 + 1)      # refused signals went AGAINST
    for i in range(10):
        led.record("ETH", "starves", +1, 100.0, 1.0, idx=i * 10)
        led.mature("ETH", 101.5, idx=i * 10 + 1)     # refused signals were RIGHT
    rows = {r["gate"]: r for r in led.snapshot(min_n=5)["gates"]}
    assert rows["saves"]["mean_move_atr"] < 0 and rows["saves"]["win_rate"] == 0.0
    assert rows["starves"]["mean_move_atr"] > 1.0 and rows["starves"]["win_rate"] == 1.0
    # worst offender (most edge discarded) sorts first, so it can't be missed
    assert led.snapshot(min_n=5)["gates"][0]["gate"] == "starves"


def test_thin_gates_are_not_reported():
    led = RefusalLedger(horizon=1)
    led.record("BTC", "rare", 1, 100.0, 1.0, idx=0)
    led.mature("BTC", 105.0, idx=1)
    assert led.snapshot(min_n=8)["gates"] == [], "one sample is not a verdict"
    assert led.snapshot(min_n=1)["gates"], "...but it is still counted"


def test_ignores_junk_and_never_grows_without_bound():
    led = RefusalLedger(horizon=1, max_pending=5)
    led.record("BTC", "", 1, 100.0, 1.0, idx=0)          # no gate
    led.record("BTC", "g", 0, 100.0, 1.0, idx=0)         # no direction
    led.record("BTC", "g", 1, 0.0, 1.0, idx=0)           # no price
    led.record("BTC", "g", 1, 100.0, 0.0, idx=0)         # no volatility
    led.record("BTC", "g", 1, float("nan"), 1.0, idx=0)  # not finite
    assert led.snapshot(min_n=1)["pending"] == 0
    for i in range(50):
        led.record("BTC", "g", 1, 100.0, 1.0, idx=1000 + i)
    assert led.snapshot(min_n=1)["pending"] == 5, "bounded ring, never a leak"


def test_extreme_gaps_cannot_swamp_the_mean():
    led = RefusalLedger(horizon=1)
    led.record("BTC", "g", 1, 100.0, 0.01, idx=0)   # a gap worth 1000 ATR
    led.mature("BTC", 110.0, idx=1)
    assert led.snapshot(min_n=1)["gates"][0]["mean_move_atr"] == 8.0


def test_gate_labels_collapse_live_messages_onto_the_gate():
    """Block reasons carry live numbers ('P(win) 33% < 52%'), so raw strings
    would make one bucket per unique number and never accumulate a sample."""
    assert gate_label("P(win) 33% < min 52%") == "min P(win)"
    assert gate_label("P 14% < EV floor 92% (b 1.10)") == "EV floor"
    assert gate_label("edge +0.13 < threshold 0.14") == "edge threshold"
    assert gate_label("15m/1h bias -1.00 vetoes edge +0.13") == "MTF veto"
    assert gate_label("trend ER 0.05 < 0.27") == "trend quality"
    assert gate_label("spread 9.8bps > 6.0bps") == "spread"
    assert gate_label("") == ""
    # the same gate with different numbers must aggregate, not fragment
    assert gate_label("P(win) 33% < min 52%") == gate_label("P(win) 41% < min 55%")
