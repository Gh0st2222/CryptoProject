"""A snapshot must not be a back door into the safety state.

risk.state holds the kill switch, the daily-loss anchor and the loss-streak
cooldown, and it used to be restored by blind setattr straight from the file.
The dangerous value is a non-finite day_start_equity: the daily-loss kill
computes `dd = -day_realized / day_start_equity`, and with a NaN there
`dd >= max_daily_loss_pct` is False forever. The kill switch quietly stops
existing, and nothing in the logs says so.
"""
import json
import math

import pytest

from bingxbot.config import RiskConfig
from bingxbot.engine.persist import restore_into
from bingxbot.engine.portfolio import Portfolio
from bingxbot.exchange.models import TradeRecord
from bingxbot.risk.manager import RiskManager


def _fresh():
    return Portfolio(1000.0, mode="paper"), RiskManager(RiskConfig())


def _snap(**over):
    d = {"cash": 1000.0, "funding_paid": 0.0, "peak_equity": 1000.0, "max_dd": 0.0,
         "trades": [], "equity_curve": [], "positions": [],
         "risk": {"day_key": "2026-07-25", "day_start_equity": 1000.0,
                  "day_realized": 0.0, "consecutive_losses": 0,
                  "cooldown_until": 0.0, "killed": False, "kill_reason": "",
                  "trades_today": 0}}
    d["risk"].update(over.pop("risk", {}))
    d.update(over)
    return json.loads(json.dumps(d))     # through JSON, like the real file


def _loss(pnl):
    return TradeRecord(symbol="BTC-USDT", side="LONG", qty=1.0, entry_price=100.0,
                       exit_price=90.0, entry_ts=0, exit_ts=0, pnl=pnl, fees=0.0,
                       reason_open="t", reason_close="stop loss")


def test_the_daily_loss_kill_still_fires_after_a_poisoned_restore():
    """The whole point. A NaN anchor must not be able to disarm the limit."""
    pf, risk = _fresh()
    restore_into(pf, risk, _snap(risk={"day_start_equity": float("nan")}))
    assert math.isfinite(risk.state.day_start_equity), "anchor is a real number"

    risk.cfg.max_daily_loss_pct = 0.05
    risk.on_trade_closed(_loss(-200.0), equity=800.0)
    assert risk.state.killed is True, "the kill switch survived the bad snapshot"


def test_non_finite_numbers_never_enter_the_portfolio():
    pf, risk = _fresh()
    restore_into(pf, risk, _snap(cash=float("nan"), peak_equity=float("inf"),
                                 max_dd=float("nan")))
    assert math.isfinite(pf.cash) and math.isfinite(pf.peak_equity)
    assert math.isfinite(pf.max_dd)


def test_a_poisoned_equity_curve_point_is_dropped_not_carried():
    """max_drawdown scans this curve; one NaN makes the statistic meaningless."""
    pf, risk = _fresh()
    restore_into(pf, risk, _snap(equity_curve=[[1, 1000.0], [2, float("nan")],
                                               [3, 1010.0]]))
    assert len(pf.equity_curve) == 2
    assert all(math.isfinite(e) for _, e in pf.equity_curve)


def test_unknown_and_unusable_risk_fields_are_ignored():
    pf, risk = _fresh()
    snap = _snap()
    snap["risk"]["not_a_real_field"] = "whatever"
    snap["risk"]["consecutive_losses"] = "seven"
    restore_into(pf, risk, snap)
    assert not hasattr(risk.state, "not_a_real_field")
    assert isinstance(risk.state.consecutive_losses, int)


def test_a_manual_kill_still_survives_the_restore():
    """The guard must not weaken what persistence is FOR."""
    pf, risk = _fresh()
    restore_into(pf, risk, _snap(risk={"killed": True, "kill_reason": "manual stop"}))
    assert risk.state.killed is True and risk.state.kill_reason == "manual stop"
    assert risk.can_enter(1000.0, 0, 1.0)[0] is False


def test_a_zero_anchor_does_not_silently_switch_off_the_daily_loss_limit():
    """Reachable with no corruption at all: restart inside the same UTC day and
    _roll_day does not roll, so whatever anchor came out of the file is kept.
    The kill is written `if day_start_equity > 0: ...`, so a zero there does not
    fail loudly — the limit just stops existing until tomorrow."""
    pf, risk = _fresh()
    import time as _t
    today = _t.strftime("%Y-%m-%d", _t.gmtime())
    restore_into(pf, risk, _snap(risk={"day_key": today, "day_start_equity": 0.0}))
    risk.cfg.max_daily_loss_pct = 0.05

    risk.can_enter(1000.0, 0, 1.0)          # any gate check re-anchors
    assert risk.state.day_start_equity > 0, "the anchor healed itself"

    risk.on_trade_closed(_loss(-200.0), equity=800.0)
    assert risk.state.killed is True


def test_re_anchoring_never_overwrites_a_good_anchor():
    pf, risk = _fresh()
    import time as _t
    today = _t.strftime("%Y-%m-%d", _t.gmtime())
    restore_into(pf, risk, _snap(risk={"day_key": today, "day_start_equity": 5000.0}))
    risk.can_enter(1234.0, 0, 1.0)
    assert risk.state.day_start_equity == pytest.approx(5000.0), \
        "the day's real starting equity is what the limit must measure against"


def test_a_clean_snapshot_restores_exactly():
    """No cost to a healthy restart."""
    pf, risk = _fresh()
    n = restore_into(pf, risk, _snap(
        cash=1234.5, peak_equity=1300.0, max_dd=0.07,
        equity_curve=[[1, 1200.0], [2, 1234.5]],
        risk={"day_realized": -65.5, "consecutive_losses": 2,
              "cooldown_until": 1234.0, "trades_today": 9}))
    assert n == 0
    assert pf.cash == pytest.approx(1234.5)
    assert pf.peak_equity == pytest.approx(1300.0)
    assert pf.max_dd == pytest.approx(0.07)
    assert len(pf.equity_curve) == 2
    assert risk.state.day_realized == pytest.approx(-65.5)
    assert risk.state.consecutive_losses == 2
    assert risk.state.cooldown_until == pytest.approx(1234.0)
    assert risk.state.trades_today == 9
