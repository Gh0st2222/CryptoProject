"""The daily track record must have exactly one row per day.

`_day` is restored from the last written row, and that row is the day that was
CLOSED, not the day in progress. So after rolling Monday and restarting on
Tuesday, `_day` reads back as Monday and the turn fires a second time. The
dashboard's monthly table sums these rows (`g.pnl += r.pnl; g.trades += ...`),
so every restart after midnight inflated that month by a whole day's P&L — in
the one table a user would look at to decide whether the bot works.
"""
import json
import time
from unittest.mock import patch

from bingxbot.engine.portfolio import Portfolio
from bingxbot.engine.record import TrackRecord
from bingxbot.exchange.models import TradeRecord

MON = int(time.mktime(time.strptime("2026-03-02", "%Y-%m-%d")) * 1000)
TUE = MON + 86_400_000
WED = TUE + 86_400_000


def _days(path):
    return [json.loads(ln)["d"] for ln in path.read_text().splitlines() if ln.strip()]


def _pf():
    pf = Portfolio(1000.0, mode="paper")
    pf.trades.append(TradeRecord(
        symbol="BTC-USDT", side="LONG", qty=1.0, entry_price=100.0, exit_price=110.0,
        entry_ts=MON, exit_ts=MON + 3_600_000, pnl=10.0, fees=0.1,
        reason_open="t", reason_close="target", mode="paper"))
    return pf


def _roll(rec, pf, at, mode="paper"):
    with patch("bingxbot.engine.record.now_ms", return_value=at):
        return rec.maybe_roll(pf, mode)


def test_a_restart_does_not_duplicate_yesterdays_row(tmp_path):
    p = tmp_path / "rec.jsonl"
    pf = _pf()
    rec = TrackRecord(p)
    _roll(rec, pf, MON)                       # first call just anchors the day
    assert _roll(rec, pf, TUE) is True        # the turn closes Monday
    assert _days(p) == ["2026-03-02"]

    restarted = TrackRecord(p)                # same file, fresh object
    assert restarted._day == "2026-03-02", "restored from the CLOSED day"
    assert _roll(restarted, pf, TUE) is False, "the same turn must not fire twice"
    assert _days(p) == ["2026-03-02"], "still exactly one Monday"


def test_repeated_restarts_never_accumulate(tmp_path):
    p = tmp_path / "rec.jsonl"
    pf = _pf()
    rec = TrackRecord(p)
    _roll(rec, pf, MON)
    _roll(rec, pf, TUE)
    for _ in range(5):
        _roll(TrackRecord(p), pf, TUE)
    assert _days(p) == ["2026-03-02"]


def test_the_monthly_total_is_no_longer_inflated(tmp_path):
    """State the consequence in the units the user sees: the dashboard sums
    row.pnl per month, so a duplicate is a double-counted trading day."""
    p = tmp_path / "rec.jsonl"
    pf = _pf()
    rec = TrackRecord(p)
    _roll(rec, pf, MON)
    _roll(rec, pf, TUE)
    _roll(TrackRecord(p), pf, TUE)            # the restart
    rows = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
    assert sum(r["pnl"] for r in rows) == 10.0, "one day of P&L, counted once"
    assert sum(r["trades"] for r in rows) == 1


def test_a_genuinely_new_day_still_rolls(tmp_path):
    """The guard must not stop the record from advancing."""
    p = tmp_path / "rec.jsonl"
    pf = _pf()
    rec = TrackRecord(p)
    _roll(rec, pf, MON)
    _roll(rec, pf, TUE)
    assert _roll(rec, pf, WED) is True
    assert _days(p) == ["2026-03-02", "2026-03-03"]


def test_the_same_day_in_a_different_mode_is_still_recorded(tmp_path):
    """Switching paper -> live legitimately gives one day two rows."""
    p = tmp_path / "rec.jsonl"
    pf = _pf()
    rec = TrackRecord(p)
    _roll(rec, pf, MON)
    _roll(rec, pf, TUE, mode="paper")
    rec._day = "2026-03-02"                   # same turn, other account
    assert _roll(rec, pf, TUE, mode="live") is True
    assert _days(p) == ["2026-03-02", "2026-03-02"]
    modes = [json.loads(ln)["mode"] for ln in p.read_text().splitlines()]
    assert modes == ["paper", "live"]
