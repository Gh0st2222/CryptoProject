"""The radar publishes measurements. It must not publish guesses as measurements.

The 4h trend read costs one klines call per symbol, so it only ever covered the
top names by funding and by volume. Every other row on the board fell back to
er 0.0 / dir 0 / atr 0.0 — and those were written out as though someone had
looked. On a live board that put LINK, LTC, AVAX, INJ and BEAT on screen at
"ER 0.00, ATR 0.00, score 0.00", which reads as five dead markets rather than
five symbols nobody measured.

Two things then read those zeros as facts:

  * `plan_adoption`'s keep test. An ADOPTED symbol that drifted out of the probe
    set stopped being measured, its absent trend read as a dead trend, and the
    seat was vacated two scans later — for a change in probe coverage, not a
    change in the market. Eviction is expensive: the symbol's brain (hedge
    weights, calibration, graded history) dies with its ctx and restarts cold.
  * the adopt test, which needs a real ER to hand out a seat and so silently
    restricted candidates to the probed few while the board advertised 24 rows.

These tests pin the distinction between "no trend" and "not measured", and pin
the rule that follows from it: an unmeasured incumbent keeps its seat, an
unmeasured newcomer never gets one.
"""
import numpy as np

from bingxbot.engine.scanner import (ADOPT_ER, KEEP_ER, TREND_MIN_QVOL,
                                     plan_adoption, rank_universe,
                                     trend_read_4h)

QV = TREND_MIN_QVOL * 4


def _tickers(*syms):
    return [{"symbol": s, "quote_volume": QV, "last": 100.0, "change_pct": 0.0}
            for s in syms]


def _premium(*syms):
    return [{"symbol": s, "mark": 100.0, "funding_rate": 0.00005,
             "next_funding_time": 0} for s in syms]


def _rising(n=120):
    return np.linspace(100.0, 140.0, n)


# ------------------------------------------------------ measured vs unmeasured

def test_a_real_read_is_marked_as_measured():
    tr = trend_read_4h(_rising())
    assert tr["probed"] is True and tr["er"] > 0.0


def test_too_few_bars_is_not_a_flat_market():
    tr = trend_read_4h(np.linspace(100.0, 140.0, 10))
    assert tr["probed"] is False


def test_an_unprobed_row_publishes_null_not_zero():
    """0.0 is a number somebody measured. None is the truth."""
    rows = rank_universe(_premium("AAA-USDT"), _tickers("AAA-USDT"),
                         trend={}, allowed={"AAA"})
    assert len(rows) == 1
    r = rows[0]
    assert r["probed"] is False
    assert r["er_4h"] is None and r["dir_4h"] is None and r["atr_pct_4h"] is None


def test_a_probed_row_still_publishes_its_numbers():
    rows = rank_universe(_premium("AAA-USDT"), _tickers("AAA-USDT"),
                         trend={"AAA-USDT": trend_read_4h(_rising())}, allowed={"AAA"})
    r = rows[0]
    assert r["probed"] is True and r["er_4h"] > 0.0 and r["dir_4h"] == 1


def test_a_probed_but_genuinely_flat_row_is_distinguishable_from_an_unprobed_one():
    """The bug in one assertion: these two used to be byte-identical on the
    board and to every consumer of it."""
    flat = trend_read_4h(np.full(120, 100.0))
    probed = rank_universe(_premium("AAA-USDT"), _tickers("AAA-USDT"),
                           trend={"AAA-USDT": flat}, allowed={"AAA"})[0]
    unprobed = rank_universe(_premium("AAA-USDT"), _tickers("AAA-USDT"),
                             trend={}, allowed={"AAA"})[0]
    assert probed["probed"] and not unprobed["probed"]
    assert probed["er_4h"] is not None and unprobed["er_4h"] is None


# ------------------------------------------------------------------- the seat

def _row(sym, er=None, d=None, probed=False, kind="watch", qv=QV):
    return {"symbol": sym, "er_4h": er, "dir_4h": d, "probed": probed,
            "kind": kind, "quote_volume": qv, "score": 0.0, "funding_apr": 0.0}


def test_an_unmeasured_incumbent_keeps_its_seat():
    """The seat-churn bug. Two scans outside the probe set used to vacate a
    perfectly healthy seat and kill its brain with it."""
    miss: dict = {}
    for _ in range(6):
        drops, _ = plan_adoption([_row("KAS-USDT", probed=False)], {"KAS-USDT"},
                                 set(), lambda s: False, 3, miss)
        assert drops == [], "an unmeasured trend is not a dead trend"


def test_a_measured_degradation_still_takes_the_seat_after_two_scans():
    """The hysteresis must survive the fix — an incumbent that really has gone
    flat still loses its seat, just not on the first bad scan."""
    miss: dict = {}
    row = [_row("KAS-USDT", er=KEEP_ER - 0.1, d=1, probed=True)]
    assert plan_adoption(row, {"KAS-USDT"}, set(), lambda s: False, 3, miss)[0] == []
    assert plan_adoption(row, {"KAS-USDT"}, set(), lambda s: False, 3, miss)[0] == ["KAS-USDT"]


def test_liquidity_is_checked_even_when_the_trend_is_unmeasured():
    """Volume comes off the ticker and is always known, so a seat whose market
    dried up must still go — being unprobed is not a shield."""
    miss: dict = {}
    row = [_row("KAS-USDT", probed=False, qv=TREND_MIN_QVOL // 10)]
    plan_adoption(row, {"KAS-USDT"}, set(), lambda s: False, 3, miss)
    assert plan_adoption(row, {"KAS-USDT"}, set(), lambda s: False, 3, miss)[0] == ["KAS-USDT"]


def test_an_unmeasured_newcomer_is_never_given_a_seat():
    """Benefit of the doubt is for incumbents only: a seat is GIVEN on evidence
    and only KEPT without it."""
    _, adds = plan_adoption([_row("NEW-USDT", kind="trend", probed=False)],
                            set(), set(), lambda s: False, 3, {})
    assert adds == []


def test_a_measured_trend_is_adopted():
    """...and the happy path still works: the live board's top row was a clean,
    liquid, adoptable trend."""
    _, adds = plan_adoption([_row("KAS-USDT", er=ADOPT_ER + 0.05, d=1,
                                  probed=True, kind="trend")],
                            set(), set(), lambda s: False, 3, {})
    assert adds == ["KAS-USDT"]


def test_a_held_position_pins_the_seat_regardless():
    miss: dict = {}
    row = [_row("KAS-USDT", er=0.0, d=0, probed=True, qv=1)]
    for _ in range(5):
        drops, _ = plan_adoption(row, {"KAS-USDT"}, set(), lambda s: True, 3, miss)
        assert drops == []


def test_a_symbol_that_vanished_from_the_board_still_loses_its_seat():
    """Absent from the board is not the same as present-but-unprobed: there is
    no ticker, so there is no liquidity check to pass."""
    miss: dict = {}
    plan_adoption([], {"GONE-USDT"}, set(), lambda s: False, 3, miss)
    assert plan_adoption([], {"GONE-USDT"}, set(), lambda s: False, 3, miss)[0] == ["GONE-USDT"]
