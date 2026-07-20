"""Diagnostic resume: one plain-text dump of everything needed to analyze the
system's state remotely — portfolio, risk, brains, gates, tuner, vault, radar,
carry, journal analytics and the effective config. Built for sharing (a support
snapshot): human-readable sections with JSON bodies, hard-capped list sizes,
and NO secrets (API keys never appear; only the has_keys flag does).

Every section builds independently — a failing section reports its error
instead of killing the report. A diagnostic tool that can crash is useless
exactly when it's needed.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict as dc_asdict

from ..config import config_public_dict
from ..util import now_ms

TRADES_N = 60          # recent closed trades included
JOURNAL_RAW_N = 25     # raw journal rows (with decision context)
VAULT_N = 15           # champions listed
RADAR_ROWS_N = 12
RECORD_DAYS_N = 30


def _dump(obj) -> str:
    return json.dumps(obj, indent=1, default=str)


def build_report(orch) -> str:
    parts: list[str] = []

    def section(title: str, fn) -> None:
        try:
            body = fn()
        except Exception as e:  # noqa: BLE001 — a broken section must not kill the report
            body = f"ERROR building section: {e!r}"
        parts.append(f"\n{'=' * 72}\n## {title}\n{'=' * 72}\n{body}")

    eng = orch.engine

    def header():
        marks = {s: st.mark_price() for s, st in eng.feed.states.items()} if eng else {}
        eq = eng.portfolio.equity(marks) if eng else None
        start = eng.portfolio.starting_balance if eng else None
        return _dump({
            "generated_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "mode": orch.mode,
            "feed": type(eng.feed).__name__ if eng else None,
            "feed_healthy": eng.feed.healthy() if eng else None,
            "interval": orch.cfg.strategy.interval,
            "symbols": list(orch.cfg.symbols),
            "adopted": sorted(eng.adopted) if eng else [],
            "uptime_min": round((now_ms() - eng.started_ts) / 60_000, 1) if eng and eng.started_ts else 0,
            "equity": round(eq, 4) if eq is not None else None,
            "starting_balance": start,
            "session_return_pct": round((eq / start - 1) * 100, 3) if eq and start else None,
            "config_version": orch.cfg.version,
            "pending_entries": eng.pending_entries() if eng else 0,
        })

    def risk_health():
        return _dump(eng.risk.status()) if eng else "engine not running"

    def divergence():
        d = orch._divergence()
        return _dump(d) if d else "engine not running"

    def positions():
        if not eng:
            return "engine not running"
        marks = {s: st.mark_price() for s, st in eng.feed.states.items()}
        out = []
        for s, p in eng.portfolio.positions.items():
            d = dc_asdict(p)
            d["mark"] = marks.get(s, 0.0)
            d["upnl"] = round(p.unrealized(marks.get(s, 0.0)), 6) if marks.get(s) else 0.0
            d["held_min"] = round((now_ms() - p.opened_ts) / 60_000, 1)
            out.append(d)
        return _dump(out) if out else "no open positions"

    def trades():
        if not eng:
            return "engine not running"
        return _dump([dc_asdict(t) for t in eng.portfolio.trades[-TRADES_N:]])

    def journal():
        return _dump({
            "summary_all_modes": orch.journal.summary(),
            "recent_rows_with_context": orch.journal.recent(JOURNAL_RAW_N),
        })

    def brains():
        if not eng:
            return "engine not running"
        return _dump(eng.snapshot()["symbols"])

    def tuner():
        at = orch.autotuner
        if at is None:
            return "auto-tuner not running"
        snap = at.snapshot()
        snap["history_full"] = at.history
        return _dump(snap)

    def vault():
        live = orch.champion_live_stats()
        champs = sorted(orch.champions, key=lambda c: (c.get("id") == orch.active_champion_id,
                                                       c.get("fitness", 0.0)), reverse=True)[:VAULT_N]
        out = []
        for c in champs:
            e = dict(c)
            e["live"] = live.get(c.get("id"), {})
            e["active"] = c.get("id") == orch.active_champion_id
            out.append(e)
        return _dump(out) if out else "vault empty"

    def overlays():
        return _dump(orch.symbol_overlays) if orch.symbol_overlays else "no per-symbol overlays"

    def radar():
        sc = orch.scanner
        if sc is None:
            return "radar not running"
        snap = sc.snapshot()
        snap["rows"] = snap.get("rows", [])[:RADAR_ROWS_N]
        return _dump(snap)

    def carry():
        return _dump(orch.carry.snapshot()) if orch.carry is not None else "carry desk not running"

    def record():
        pf = eng.portfolio if eng else None
        snap = orch.record.snapshot(pf, pf.mode if pf else "paper")
        snap["rows"] = snap.get("rows", [])[-RECORD_DAYS_N:]
        return _dump(snap)

    def config():
        return _dump(config_public_dict(orch.cfg))

    parts.append("PULSE — diagnostic resume (no secrets; safe to share)")
    section("HEADER / SESSION", header)
    section("RISK & HEALTH", risk_health)
    section("DIVERGENCE MONITOR", divergence)
    section("OPEN POSITIONS", positions)
    section(f"RECENT CLOSED TRADES (last {TRADES_N})", trades)
    section("JOURNAL ANALYTICS + RECENT DECISIONS", journal)
    section("PER-SYMBOL BRAINS (edge/gates/desks/alphas/ladder)", brains)
    section("AUTO-TUNER (state + full promotion history)", tuner)
    section(f"CHAMPION VAULT (top {VAULT_N})", vault)
    section("PER-SYMBOL OVERLAYS", overlays)
    section("RADAR (universe + board)", radar)
    section("CARRY DESK", carry)
    section(f"TRACK RECORD (last {RECORD_DAYS_N} days)", record)
    section("EFFECTIVE CONFIG (public — no keys)", config)
    return "\n".join(parts) + "\n"
