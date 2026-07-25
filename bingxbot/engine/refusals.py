"""What the gates turn away — the system's missing feedback loop.

Every gate in the entry chain is measured by what it PREVENTS, and until now
nothing measured what it COSTS. A gate that refuses signals which would have
run against us is earning its keep; a gate refusing signals that would have
paid is quietly starving the account, and both look identical from the
outside: a flat equity curve and a dashboard full of red gate rows.

So each refused signal is recorded and then graded on the same clock the brain
grades itself with: the forward move over `horizon` bars, in the signal's own
direction, normalized by the ATR at the moment of refusal. Per gate we keep the
count, the hit rate and the mean ATR-move.

Read it honestly — this is DIRECTIONAL outcome, not simulated PnL. It does not
run the exit engine, pay fees or model fills, so it cannot say "this gate cost
us $X". What it can say, which is the thing nobody could say before, is
whether the signals a gate rejects were systematically right or systematically
wrong. A mean move near zero means the gate is discarding noise (working). A
consistently positive mean over a real sample means the gate is discarding
edge, and that is the number worth arguing with.
"""
from __future__ import annotations

import math
from collections import deque

MAX_PENDING = 400        # per symbol; a bounded ring, never a leak
MIN_REPORT = 8           # a gate needs this many graded refusals to be listed


class RefusalLedger:
    """Bounded, per-gate accounting of refused entry signals."""

    def __init__(self, horizon: int = 8, max_pending: int = MAX_PENDING):
        self.horizon = max(1, int(horizon))
        self.max_pending = max(1, int(max_pending))
        self._pending: dict[str, deque] = {}
        self.stats: dict[str, dict] = {}      # gate -> {n, hits, sum_move}

    def record(self, symbol: str, gate: str, direction: int, close: float,
               atr: float, idx: int) -> None:
        """A bar closed, the brain wanted this direction, and `gate` said no."""
        if not gate or direction == 0 or close <= 0 or atr <= 0:
            return
        if not (math.isfinite(close) and math.isfinite(atr)):
            return
        q = self._pending.setdefault(symbol, deque(maxlen=self.max_pending))
        q.append((idx, gate, direction, close, atr))

    def mature(self, symbol: str, close: float, idx: int) -> None:
        """Grade every refusal on this symbol that has reached the horizon."""
        q = self._pending.get(symbol)
        if not q or close <= 0 or not math.isfinite(close):
            return
        while q and idx - q[0][0] >= self.horizon:
            _i, gate, d, c0, atr = q.popleft()
            move = (close - c0) / atr * d          # ATR units, signal-relative
            if not math.isfinite(move):
                continue
            move = max(-8.0, min(8.0, move))       # a gap must not swamp the mean
            s = self.stats.setdefault(gate, {"n": 0, "hits": 0, "sum_move": 0.0})
            s["n"] += 1
            s["sum_move"] += move
            if move > 0:
                s["hits"] += 1

    def snapshot(self, min_n: int = MIN_REPORT) -> dict:
        """Per-gate verdict, worst offender first (most edge discarded)."""
        rows = []
        for gate, s in self.stats.items():
            if s["n"] < min_n:
                continue
            rows.append({
                "gate": gate,
                "refused": s["n"],
                "win_rate": round(s["hits"] / s["n"], 4),
                "mean_move_atr": round(s["sum_move"] / s["n"], 4),
            })
        rows.sort(key=lambda r: r["mean_move_atr"], reverse=True)
        return {"horizon_bars": self.horizon,
                "pending": sum(len(q) for q in self._pending.values()),
                "graded": sum(s["n"] for s in self.stats.values()),
                "gates": rows}


def gate_label(reason: str) -> str:
    """Collapse a live block message ('P(win) 33% < 52%', 'trend ER 0.14 <
    0.27') onto the GATE that produced it, so the ledger aggregates instead of
    accumulating one bucket per unique number."""
    r = (reason or "").strip()
    if not r:
        return ""
    low = r.lower()
    for needle, label in (
        ("threshold", "edge threshold"),
        ("p(win)", "min P(win)"),
        ("ev floor", "EV floor"),
        ("cost", "cost multiple"),
        ("vetoes edge", "MTF veto"),
        ("bias", "MTF veto"),
        ("align", "regime align"),
        ("er ", "trend quality"),
        ("%b", "range band"),
        ("range scalps off", "range disabled"),
        ("volatile", "volatile sit-out"),
        ("funding", "funding"),
        ("order-flow", "order flow"),
        ("spread", "spread"),
        ("cooldown", "risk cooldown"),
        ("kill switch", "kill switch"),
        ("max open positions", "position cap"),
        ("net exposure", "exposure cap"),
        ("volatility", "no volatility"),
        ("minimum", "size minimum"),
    ):
        if needle in low:
            return label
    return r[:28]
