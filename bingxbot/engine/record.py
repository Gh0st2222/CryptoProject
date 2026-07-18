"""Track record: daily performance snapshots that outlive any session.

One JSONL row per UTC day — closing equity, realized PnL, trades, fees,
funding — appended at day rollover and safe across restarts (the last row's
date seeds the rollover detector; the persisted paper state carries equity
continuity). This is the provable months-long record the whole project is
really building: a system with a live, journaled track record that matches
its backtests is worth more than its early profits.
"""
from __future__ import annotations

import calendar
import json
import logging
import time
from pathlib import Path

from ..config import ROOT
from ..util import now_ms

log = logging.getLogger("record")

RECORD_PATH = ROOT / "data_cache" / "track_record.jsonl"
DAY_MS = 86_400_000


def _day_str(ts_ms: int) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(ts_ms / 1000))


class TrackRecord:
    def __init__(self, path: Path = RECORD_PATH):
        self.path = path
        self.rows: list[dict] = self._load()
        self._day = self.rows[-1]["d"] if self.rows else None

    def _load(self) -> list[dict]:
        rows: list[dict] = []
        try:
            with self.path.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except OSError:
            pass
        return rows[-1000:]

    @staticmethod
    def _summary(portfolio, day: str, mode: str) -> dict:
        day_t0 = calendar.timegm(time.strptime(day, "%Y-%m-%d")) * 1000
        trades = [t for t in portfolio.trades
                  if day_t0 <= t.exit_ts < day_t0 + DAY_MS and t.mode == mode]
        eq = portfolio.equity_curve[-1][1] if portfolio.equity_curve else portfolio.cash
        return {
            "d": day, "mode": mode,
            "equity": round(eq, 4),
            "pnl": round(sum(t.pnl for t in trades), 4),
            "trades": len(trades),
            "wins": sum(1 for t in trades if t.pnl > 0),
            "fees": round(sum(t.fees for t in trades), 4),
            "funding": round(portfolio.funding_paid, 4),   # cumulative to date
        }

    def maybe_roll(self, portfolio, mode: str) -> bool:
        """Called periodically; appends yesterday's row when the UTC day turns."""
        today = _day_str(now_ms())
        if self._day is None:
            self._day = today
            return False
        if today == self._day:
            return False
        prev, self._day = self._day, today
        row = self._summary(portfolio, prev, mode)
        self.rows.append(row)
        self.rows = self.rows[-1000:]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(row) + "\n")
        except OSError as e:
            log.warning("track record write failed: %s", e)
        log.info("track record: %s closed at %.2f (%+.2f, %d trades)",
                 prev, row["equity"], row["pnl"], row["trades"])
        return True

    def snapshot(self, portfolio=None, mode: str = "paper") -> dict:
        d = {"rows": self.rows[-500:]}
        if portfolio is not None:
            d["today"] = self._summary(portfolio, _day_str(now_ms()), mode)
        return d
