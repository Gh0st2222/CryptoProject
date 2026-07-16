"""MetaAllocator — the firm's CIO.

Each desk (trend / meanrev / micro / vol / carry) reports a directional call
every bar; those calls are graded against realized returns just like the
alphas. The allocator maintains an online risk-adjusted performance estimate
per desk and turns it into a capital weight via multiplicative weights (Hedge
over desks), with a floor so no desk is ever permanently exiled and can earn
its way back when its market returns.

This is what "evaluate many strategies at once and back the winners" means:
desks that print get more sway; desks that bleed get muted automatically —
auto-correction without anyone touching a setting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..util import clamp


@dataclass
class DeskPerf:
    weight: float = 0.0
    ew_payoff: float = 0.0      # exp-weighted mean graded payoff
    ew_win: float = 0.0         # exp-weighted directional hit rate
    ew_var: float = 1.0         # exp-weighted payoff variance (for risk adj)
    graded: int = 0
    disabled: bool = False


class MetaAllocator:
    def __init__(self, desks: list[str], eta: float = 0.6, floor: float = 0.05,
                 ew_alpha: float = 0.05, disable_after: int = 120, max_weight: float = 0.40):
        self.desks = list(desks)
        self.eta = eta
        self.floor = floor
        self.ew_alpha = ew_alpha            # faster forgetting -> adapts to regime shifts
        self.disable_after = disable_after
        self.max_weight = max_weight        # no single desk may dominate the book
        n = len(self.desks)
        self.perf: dict[str, DeskPerf] = {d: DeskPerf(weight=1.0 / n) for d in self.desks}
        self._log_w: dict[str, float] = {d: 0.0 for d in self.desks}  # unnormalized log-weights

    def weights(self) -> dict[str, float]:
        return {d: p.weight for d, p in self.perf.items()}

    def update(self, desk: str, payoff: float, hit: bool) -> None:
        """Grade one desk's matured directional call. `payoff` is the
        ATR-normalized realized return in the desk's called direction."""
        p = self.perf.get(desk)
        if p is None:
            return
        a = self.ew_alpha
        p.ew_payoff = (1 - a) * p.ew_payoff + a * payoff
        p.ew_win = (1 - a) * p.ew_win + a * (1.0 if hit else 0.0)
        p.ew_var = (1 - a) * p.ew_var + a * (payoff - p.ew_payoff) ** 2
        p.graded += 1
        # Risk-adjusted score drives multiplicative weights.
        risk_adj = p.ew_payoff / math.sqrt(max(p.ew_var, 1e-6))
        self._log_w[desk] += self.eta * clamp(risk_adj, -3.0, 3.0) * 0.02
        # Auto-disable a desk that is durably negative; keep probing rarely.
        p.disabled = (p.graded > self.disable_after and p.ew_payoff < -0.05 and p.ew_win < 0.42)
        self._recompute()

    def _recompute(self) -> None:
        # softmax over log-weights, decayed toward uniform, with an exact floor
        m = max(self._log_w.values())
        raw = {d: math.exp(self._log_w[d] - m) for d in self.desks}
        for d in self.desks:
            if self.perf[d].disabled:
                raw[d] *= 0.15                       # probe weight, not zero
        tot = sum(raw.values()) or 1.0
        n = len(self.desks)
        free = 1.0 - n * self.floor
        for d in self.desks:
            self.perf[d].weight = self.floor + free * raw[d] / tot
        # ceiling: cap any runaway desk and redistribute the overflow to the rest,
        # so the CIO can't over-commit the whole book to one desk (it over-weighted
        # mean-reversion into an uptrend before). A few passes handle cascades.
        ceil = max(self.max_weight, 1.0 / n)
        for _ in range(3):
            over = {d: self.perf[d].weight - ceil for d in self.desks if self.perf[d].weight > ceil + 1e-9}
            if not over:
                break
            spill = sum(over.values())
            under = [d for d in self.desks if self.perf[d].weight < ceil - 1e-9]
            base = sum(self.perf[d].weight for d in under) or 1.0
            for d in over:
                self.perf[d].weight = ceil
            for d in under:
                self.perf[d].weight += spill * self.perf[d].weight / base
        # gentle decay of log-weights toward 0 keeps adaptation reversible
        for d in self.desks:
            self._log_w[d] *= 0.995

    def snapshot(self) -> dict:
        return {
            d: {
                "weight": round(p.weight, 4),
                "ew_payoff": round(p.ew_payoff, 4),
                "win": round(p.ew_win, 4),
                "sharpe": round(p.ew_payoff / math.sqrt(max(p.ew_var, 1e-6)), 3),
                "graded": p.graded,
                "disabled": p.disabled,
            }
            for d, p in self.perf.items()
        }
