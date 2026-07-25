"""Paper-session persistence: the account survives restarts.

The portfolio (cash, open positions, trade tail, equity curve) and the risk
day-state are snapshotted to disk every few seconds while paper trading and on
graceful stop, then restored on the next paper start — so a settings change,
crash or reboot no longer wipes an 8-hour session. Live mode never uses this
(the exchange is the source of truth there).
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
from pathlib import Path

from ..config import ROOT
from ..exchange.models import Position, TradeRecord
from ..util import now_ms

log = logging.getLogger("persist")

STATE_PATH = ROOT / "data_cache" / "paper_state.json"
MAX_AGE_MS = 7 * 86_400_000     # a week-old snapshot is stale — start fresh
TRADE_TAIL = 300
CURVE_TAIL = 4000


def save_paper_state(portfolio, risk_state, path: Path = STATE_PATH,
                     brains: dict | None = None, entry_ctx: dict | None = None,
                     health: dict | None = None) -> None:
    try:
        data = {
            "ts": now_ms(),
            "mode": portfolio.mode,
            "starting_balance": portfolio.starting_balance,
            "cash": portfolio.cash,
            "funding_paid": portfolio.funding_paid,
            "peak_equity": portfolio.peak_equity,
            "max_dd": portfolio.max_dd,
            "positions": [dataclasses.asdict(p) for p in portfolio.positions.values()],
            "trades": [dataclasses.asdict(t) for t in portfolio.trades[-TRADE_TAIL:]],
            "equity_curve": list(portfolio.equity_curve)[-CURVE_TAIL:],
            "risk": dataclasses.asdict(risk_state),
            "health": health or {},       # the throttle's memory (recent R, peak)
            "brains": brains or {},        # per-symbol online learning survives too
            "entry_ctx": entry_ctx or {},  # open positions keep their decision context
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(path)
    except Exception as e:  # noqa: BLE001 — persistence must never break trading
        log.warning("paper state save failed: %s", e)


def _build(cls, d: dict):
    """Reconstruct a dataclass from a dict, ignoring unknown keys (schema drift)."""
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


def load_paper_state(starting_balance: float, path: Path = STATE_PATH) -> dict | None:
    """Return a restorable snapshot, or None if absent/stale/incompatible.
    A changed starting balance means the user wants a fresh account."""
    try:
        d = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if d.get("mode") != "paper":
        return None
    if now_ms() - d.get("ts", 0) > MAX_AGE_MS:
        log.info("paper state is stale — starting fresh")
        return None
    if abs(d.get("starting_balance", 0.0) - starting_balance) > 1e-9:
        log.info("starting balance changed — fresh paper account")
        return None
    return d


def _num(x, default: float) -> float:
    """A snapshot number, or `default` if it is not a real one.

    json.dumps writes NaN/Infinity as bare literals and json.loads reads them
    back, so a single non-finite value that reaches the file survives every
    restart. The dangerous one is day_start_equity: the daily-loss kill
    computes `dd = -day_realized / day_start_equity`, and a NaN there makes
    `dd >= max_daily_loss_pct` False forever — the kill switch quietly stops
    existing, with nothing in the logs to say so."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def restore_into(portfolio, risk, snapshot: dict) -> int:
    """Apply a snapshot onto a fresh Portfolio + RiskManager. Returns the number
    of open positions restored."""
    portfolio.cash = _num(snapshot.get("cash"), portfolio.cash)
    portfolio.funding_paid = _num(snapshot.get("funding_paid"), 0.0)
    portfolio.peak_equity = _num(snapshot.get("peak_equity"), 0.0)
    portfolio.max_dd = _num(snapshot.get("max_dd"), 0.0)
    portfolio.trades = [_build(TradeRecord, t) for t in snapshot.get("trades", [])]
    curve = [(int(ts), float(eq)) for ts, eq in snapshot.get("equity_curve", [])
             if math.isfinite(_num(eq, math.nan))]
    try:
        portfolio.equity_curve.extend(curve)
    except AttributeError:
        portfolio.equity_curve = curve
    n = 0
    for pd in snapshot.get("positions", []):
        try:
            pos = _build(Position, pd)
            portfolio.positions[pos.symbol] = pos
            n += 1
        except (TypeError, ValueError) as e:
            log.warning("could not restore position %s: %s", pd.get("symbol"), e)
    # Restore risk state by FIELD TYPE, not by blind setattr. This is the
    # safety state — the kill switch, the daily-loss anchor, the loss-streak
    # cooldown — and a value the live code could never produce must not be able
    # to enter through a file. A non-finite day_start_equity in particular
    # disables the daily-loss kill silently and permanently.
    rs = snapshot.get("risk", {})
    if isinstance(rs, dict):
        # Coerce by the DECLARED field type, not by whatever the field happens
        # to hold right now — an int sitting in a float field would otherwise
        # make the next restore truncate it. `from __future__ import
        # annotations` means f.type is the annotation's source text.
        types = {f.name: str(f.type) for f in dataclasses.fields(risk.state)}
        for k, v in rs.items():
            kind = types.get(k)
            if kind is None:
                continue
            try:
                if kind == "bool":
                    setattr(risk.state, k, bool(v))
                elif kind == "int":
                    setattr(risk.state, k, int(v))
                elif kind == "float":
                    setattr(risk.state, k, _num(v, getattr(risk.state, k)))
                elif kind == "str":
                    setattr(risk.state, k, str(v))
            except (TypeError, ValueError):
                log.warning("ignored unusable risk field %s=%r in snapshot", k, v)
    # restore the throttle's memory FIRST (recent R window + equity peak),
    # then re-anchor on current cash so drawdown math continues from here.
    risk.health.load_state(snapshot.get("health"))
    eq = portfolio.cash
    risk.health.mark_equity(eq)
    log.info("paper state restored: %d open positions, %d trades, cash %.2f",
             n, len(portfolio.trades), portfolio.cash)
    return n


def clear_paper_state(path: Path = STATE_PATH) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
