"""Cost per unit of risk — the denominator nothing else in the system touches.

`gate_ev` already enforces this identity per trade; these functions expose it
as a readout so a symbol's demands are visible before the P&L discovers them.
The tests pin the identity itself and the two levers that move it, because the
whole point of surfacing the number is that it is trusted.
"""
import math

import pytest

from bingxbot.config import RiskConfig
from bingxbot.engine.backtest import gate_ev
from bingxbot.engine.tradability import breakeven_win_rate, cost_in_r, symbol_economics


def test_the_identity_matches_the_gate_that_enforces_it():
    """The readout and the live EV gate must not be able to disagree —
    gate_ev computes the same cost_r and the same breakeven, plus a margin."""
    rc = RiskConfig()
    atr_pct, fees, spread, slip = 0.0009, 0.0007, 1.0, 1.0
    cr = cost_in_r(atr_pct, spread, fees, slip, rc.sl_atr_min)
    be = breakeven_win_rate(cr, rc.expected_rr)
    # the gate refuses just below the breakeven+margin and accepts just above
    row = {"atr_pct": atr_pct}
    ok_lo, _ = gate_ev(rc, rc.expected_rr, be + 0.019, row, fees, spread, slip)
    ok_hi, _ = gate_ev(rc, rc.expected_rr, be + 0.021, row, fees, spread, slip)
    assert ok_lo is False and ok_hi is True


def test_cost_falls_with_the_stop_width():
    """Fees are charged on notional and notional is risk/stop, so a wider stop
    is a smaller position for the same risk — and therefore cheaper per R.
    This is the lever that looks like it should cost something and does not."""
    a = cost_in_r(0.0009, 1.0, 0.0007, 1.0, 1.5)
    b = cost_in_r(0.0009, 1.0, 0.0007, 1.0, 2.2)
    assert b < a
    assert b == pytest.approx(a * 1.5 / 2.2, rel=1e-9)


def test_cost_falls_with_volatility():
    """A 4x ATR is a 4x stop is a quarter of the cost in R. This is why the bar
    clock, which nobody thinks of as a cost parameter, is the biggest one."""
    lo = cost_in_r(0.0009, 1.0, 0.0007, 1.0, 1.5)
    hi = cost_in_r(0.0036, 1.0, 0.0007, 1.0, 1.5)
    assert hi == pytest.approx(lo / 4.0, rel=1e-9)


def test_a_wide_book_is_charged_as_cost_not_ignored():
    tight = cost_in_r(0.0036, 0.4, 0.0007, 1.0, 1.5)
    wide = cost_in_r(0.0036, 30.3, 0.0007, 1.0, 1.5)
    assert wide > tight * 3, "a 30bp spread must dominate a 7bp fee"


def test_breakeven_rises_with_cost_and_falls_with_payoff():
    assert breakeven_win_rate(0.0, 2.2) == pytest.approx(1 / 3.2)
    assert breakeven_win_rate(0.68, 2.2) > breakeven_win_rate(0.14, 2.2)
    assert breakeven_win_rate(0.3, 3.0) < breakeven_win_rate(0.3, 1.5)


def test_the_real_numbers_from_a_live_resume():
    """BTC-USDT as observed: ATR 57.69 at 64,439.55, spread 0.144bp, maker
    entry + taker exit, 1.5 ATR stop. It needs to win more than half its trades
    at 2.2:1 simply to stand still."""
    atr_pct = 57.69241031 / 64439.55
    cr = cost_in_r(atr_pct, 0.1436, 0.0007, 1.0, 1.5)
    assert cr == pytest.approx(0.681, abs=0.005)
    assert breakeven_win_rate(cr, 2.2) == pytest.approx(0.525, abs=0.005)


def test_unknown_inputs_return_none_rather_than_a_confident_number():
    """A readout that invents a cost when it cannot know one is worse than no
    readout — this is the number an operator would act on."""
    for bad in (float("nan"), float("inf"), 0.0, -0.001):
        assert cost_in_r(bad, 1.0, 0.0007, 1.0, 1.5) is None
    assert cost_in_r(0.0009, float("nan"), 0.0007, 1.0, 1.5) is None
    assert cost_in_r(0.0009, 1.0, 0.0007, 1.0, 0.0) is None
    assert breakeven_win_rate(float("nan"), 2.2) is None


def test_symbol_economics_shape_survives_missing_volatility():
    e = symbol_economics(float("nan"), 1.0, 0.0007, 1.0, 1.5, 2.2)
    assert e["cost_r"] is None and e["breakeven_win_rate"] is None
    e2 = symbol_economics(0.0009, 1.0, 0.0007, 1.0, 1.5, 2.2)
    assert e2["cost_r"] > 0 and 0 < e2["breakeven_win_rate"] < 1
    assert math.isclose(e2["stop_pct"], 0.00135, abs_tol=1e-5)
