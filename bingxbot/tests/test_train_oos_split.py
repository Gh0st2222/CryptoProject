"""No judged bar may be a bar the search already fitted.

This project has fixed this leak once. The primary clock's OOS folds used to
start at 60% of the series while the DE trained to 75%, so fold 0 sat entirely
inside the training window and fold 1 half-way in — half of every
"out-of-sample" verdict was scored on data the search had seen. The fix was to
DERIVE the training cut from the fold geometry rather than write the number down
twice, plus a purge gap so a training bar's outcome (which matures horizon_bars
later) cannot mature inside the judge's window either.

Two places kept writing it down twice anyway: the vault re-validation window and
the entire trial clock, both hard-coded at 0.75 against a tail of 0.40. The
trial clock was therefore leaking exactly as the primary clock once did — on the
lane whose only job is to answer "which bar size earns more out-of-sample",
whose champions are kept for the day the user switches interval.

These tests pin the derivation itself, so the constants cannot drift apart
again for any value of OOS_TAIL_FRAC.
"""
import pytest

from bingxbot.engine import autotuner as at
from bingxbot.engine.autotuner import (MIN_FOLD_BARS, OOS_TAIL_FRAC, PURGE_BARS,
                                       TRAIN_FRAC, _train_split)
from bingxbot.engine.search import portfolio_folds

WARMUP = 300


def test_the_training_fraction_is_derived_from_the_fold_geometry():
    assert TRAIN_FRAC == pytest.approx(1.0 - OOS_TAIL_FRAC)


def test_training_stops_before_the_first_judged_bar():
    """The core invariant, checked on the real slicing rather than on the
    constants: the last bar the DE may fit must precede the first bar the judge
    trades, by at least the purge gap."""
    n = 17_280                       # 180 days of 15m bars
    candles = list(range(n))
    train = _train_split(candles)
    first_judged = int(n * TRAIN_FRAC)        # portfolio_folds' tail starts here
    assert train[-1] < first_judged - PURGE_BARS + 1, (
        f"training ends at bar {train[-1]}, judging starts at {first_judged}")


def test_the_purge_gap_covers_the_longest_horizon():
    """A training bar's outcome matures horizon_bars later; if that lands inside
    the judge's window the label itself leaks. The horizon box tops out at 16."""
    from bingxbot.engine.backtest import TUNABLES
    assert PURGE_BARS >= TUNABLES["horizon_bars"][1]


@pytest.mark.parametrize("tail", [0.20, 0.30, 0.40, 0.55])
def test_the_split_stays_honest_for_any_tail(tail, monkeypatch):
    """The old hard-coded 0.75 was safe only because the tail happened to be
    0.40. Tighten the tail below 0.25 and it would have started scoring on
    fitted bars, with nothing to notice."""
    monkeypatch.setattr(at, "OOS_TAIL_FRAC", tail)
    monkeypatch.setattr(at, "TRAIN_FRAC", 1.0 - tail)
    n = 17_280
    candles = list(range(n))
    train = at._train_split(candles)
    first_judged = int(n * (1.0 - tail))
    assert train[-1] < first_judged, f"tail {tail}: training overlaps judging"
    # ...and the vault re-validation window, which is judged on too
    valid = at.AutoTuner._valid_window(None, candles)
    assert valid[0] >= first_judged - WARMUP, (
        f"tail {tail}: the re-validation lead-in reaches back past the warmup "
        f"into fitted data")
    assert valid[0] + WARMUP <= first_judged + 1


def test_the_revalidation_window_starts_trading_where_training_stops():
    """A 300-bar lead-in is warmup, not evidence: OOS trading has to begin at
    the cut, not 300 bars before it and not a third of the tail after it."""
    n = 17_280
    valid = at.AutoTuner._valid_window(None, list(range(n)))
    assert valid[0] == int(n * TRAIN_FRAC) - WARMUP
    assert valid[-1] == n - 1, "the window must run to the freshest bar"


def test_the_trial_clock_uses_the_same_split_as_the_primary_clock():
    """The trial lane judges with portfolio_folds at OOS_TAIL_FRAC, so it must
    train with _train_split — not with a second copy of the number."""
    import inspect
    code = [ln.split("#")[0].strip()
            for ln in inspect.getsource(at.AutoTuner._trial_cycle).splitlines()]
    assert "train = _train_split(candles)" in code, (
        "the trial clock must derive its training window, not hard-code a cut")
    assert not [ln for ln in code if ln.startswith("val_cut")], (
        "a second, independent split fraction is back in the trial clock")


def test_the_trial_clocks_training_bars_are_not_in_its_judged_folds():
    """End to end on the real slicers: no bar the trial DE fits may appear as a
    TRADED bar in any fold the trial judge scores."""
    n = 17_280
    candles = list(range(n))
    train = set(_train_split(candles))
    folds = portfolio_folds({"X": candles}, k=4, tail_frac=OOS_TAIL_FRAC, warmup=WARMUP)
    assert folds, "no folds built — the test would prove nothing"
    for i, f in enumerate(folds):
        traded = f["X"][WARMUP:]     # the lead-in is warmup, never traded
        overlap = train.intersection(traded)
        assert not overlap, (
            f"fold {i}: {len(overlap)} traded bars were in the training window")


def test_a_short_series_still_yields_a_usable_training_split():
    """_train_split floors at MIN_FOLD_BARS so a thin symbol cannot produce an
    empty training set — which would score every candidate identically and turn
    DE selection into a coin flip."""
    for n in (400, 1_000, 3_000):
        assert len(_train_split(list(range(n)))) >= min(MIN_FOLD_BARS, n)
