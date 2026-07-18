"""Persistent trade journal.

Every closed trade is appended to a JSONL file with the full decision context it
was taken under — regime, the 1m/5m/15m/1h ladder, fused edge, calibrated P(win),
the dominant desk, and the exit reason. It survives restarts (the in-memory list
is one process's worth; the file is the record), and it is what turns "the bot
lost" into "the bot loses when it fades a 1h uptrend at hour 14" — i.e. something
we can actually act on. Backs the Analytics tab and the divergence monitor.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ..config import ROOT

log = logging.getLogger("journal")

JOURNAL_PATH = ROOT / "data_cache" / "journal.jsonl"
MAX_MEM = 4000     # rows kept in memory; the file keeps everything


def _bucket_align(a: float) -> str:
    if a >= 0.35:
        return "with-trend+"
    if a >= 0.1:
        return "with-trend"
    if a <= -0.35:
        return "counter-trend+"
    if a <= -0.1:
        return "counter-trend"
    return "neutral"


class TradeJournal:
    def __init__(self, path: Path = JOURNAL_PATH):
        self.path = path
        self.rows: list[dict] = self._load()

    def _load(self) -> list[dict]:
        rows: list[dict] = []
        try:
            with self.path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except OSError:
            pass
        return rows[-MAX_MEM:]

    def record(self, row: dict) -> None:
        self.rows.append(row)
        self.rows = self.rows[-MAX_MEM:]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(row) + "\n")
        except OSError as e:  # never let journaling break trading
            log.warning("journal write failed: %s", e)

    def recent(self, n: int = 400) -> list[dict]:
        return self.rows[-n:]

    # ----------------------------------------------------------- analytics

    def summary(self, mode: str | None = None) -> dict:
        rows = [r for r in self.rows if (mode is None or r.get("mode") == mode)]
        if not rows:
            return {"trades": 0}

        def agg(key_fn):
            groups: dict[str, dict] = {}
            for r in rows:
                k = key_fn(r)
                if k is None:
                    continue
                g = groups.setdefault(str(k), {"n": 0, "wins": 0, "pnl": 0.0})
                g["n"] += 1
                g["wins"] += 1 if r.get("pnl", 0) > 0 else 0
                g["pnl"] += float(r.get("pnl", 0.0))
            return {k: {"n": g["n"], "win_rate": round(g["wins"] / g["n"], 3) if g["n"] else 0.0,
                        "pnl": round(g["pnl"], 4)} for k, g in groups.items()}

        wins = sum(1 for r in rows if r.get("pnl", 0) > 0)
        gross_w = sum(r["pnl"] for r in rows if r.get("pnl", 0) > 0)
        gross_l = -sum(r["pnl"] for r in rows if r.get("pnl", 0) <= 0)
        def signed_align(r) -> float:
            """Alignment RELATIVE TO THE TRADE: a SHORT taken with the higher-TF
            bias pointing down is WITH-trend — bucketing raw bias sign alone
            mislabeled every with-trend short as counter-trend."""
            d = 1.0 if r.get("side") == "LONG" else -1.0
            return r.get("mtf_bias", 0.0) * d

        # excursion analytics: how much heat trades take (MAE), how much they
        # ever showed (MFE), and what fraction of the shown profit exits capture.
        exc = [r for r in rows if r.get("mfe_r") is not None]
        avg_mae = sum(r.get("mae_r", 0.0) for r in exc) / len(exc) if exc else 0.0
        avg_mfe = sum(r.get("mfe_r", 0.0) for r in exc) / len(exc) if exc else 0.0
        rs = sum(r.get("r", 0.0) for r in exc)
        mfes = sum(r.get("mfe_r", 0.0) for r in exc)
        return {
            "trades": len(rows),
            "win_rate": round(wins / len(rows), 4),
            "pnl": round(sum(r.get("pnl", 0.0) for r in rows), 4),
            "avg_mae_r": round(avg_mae, 3),
            "avg_mfe_r": round(avg_mfe, 3),
            "mfe_capture": round(rs / mfes, 3) if mfes > 0 else 0.0,
            "profit_factor": round(gross_w / gross_l, 3) if gross_l > 0 else (999.0 if gross_w > 0 else 0.0),
            "by_regime": agg(lambda r: r.get("regime")),
            "by_alignment": agg(lambda r: _bucket_align(signed_align(r))),
            "by_hour": agg(lambda r: r.get("hour")),
            "by_desk": agg(lambda r: r.get("desk")),
            "by_side": agg(lambda r: r.get("side")),
            "by_exit": agg(lambda r: r.get("reason_close")),
        }
