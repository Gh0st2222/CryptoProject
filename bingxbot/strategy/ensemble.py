"""AlphaEnsemble: online-adaptive combination of the alpha signals.

Three adaptive mechanisms, all self-contained and observable from the UI:

1. Hedge (multiplicative weights): every alpha's directional call is graded
   against the ATR-normalized realized return `horizon_bars` later; weights
   multiply by exp(eta * payoff) and renormalize. Alphas that keep being
   right accumulate weight; wrong ones decay to a floor, never to zero, so
   they can win their weight back when the market changes.

2. Regime gating: a detected regime (trend/range/volatile) multiplies each
   alpha's contribution by a suitability factor before mixing.

3. Cost-aware adaptive threshold: the entry threshold tracks the recent
   |score| distribution to hit a target trade rate, and a trade is only
   taken when the calibrated expected move (beta * |score| * ATR) clears
   round-trip costs with margin.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from ..util import clamp
from .alphas import ALPHAS, MICRO_ALPHAS
from .regime import REGIME_ALPHA_MULT, detect_regime


@dataclass
class AlphaStat:
    calls: int = 0
    hits: int = 0
    payoff_sum: float = 0.0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.calls if self.calls else 0.0


@dataclass
class PendingGrade:
    idx: int
    close: float
    atr: float
    scores: dict[str, float]
    ensemble_score: float = 0.0


class AlphaEnsemble:
    def __init__(
        self,
        eta: float = 0.35,
        weight_floor: float = 0.04,
        horizon_bars: int = 5,
        base_threshold: float = 0.30,
        threshold_adapt: bool = True,
        target_trades_per_hour: float = 2.0,
        bars_per_hour: float = 60.0,
        cost_multiple: float = 1.5,
    ):
        self.eta = eta
        self.floor = weight_floor
        self.horizon = max(1, horizon_bars)
        self.base_threshold = base_threshold
        self.threshold_adapt = threshold_adapt
        self.target_rate = max(0.1, target_trades_per_hour)
        self.bars_per_hour = bars_per_hour
        self.cost_multiple = cost_multiple

        self.names = list(ALPHAS.keys())
        n = len(self.names)
        self.weights: dict[str, float] = {k: 1.0 / n for k in self.names}
        self.stats: dict[str, AlphaStat] = {k: AlphaStat() for k in self.names}
        self._pending: deque[PendingGrade] = deque()
        self._score_hist: deque[float] = deque(maxlen=720)
        self._bar_idx = 0
        self.beta = 1.0            # realized-move / predicted-move calibration
        self.threshold = base_threshold
        self.last_regime = "RANGE"
        self.last_conf = 0.0
        self.last_scores: dict[str, float] = {k: 0.0 for k in self.names}
        self.last_score = 0.0
        self.graded = 0

    # ------------------------------------------------------------- scoring

    def evaluate(self, row: dict, micro: dict) -> dict:
        """Full per-bar evaluation. Returns a snapshot dict (also kept on self)."""
        regime, conf = detect_regime(row)
        mults = REGIME_ALPHA_MULT[regime]
        scores: dict[str, float] = {}
        num = 0.0
        den = 0.0
        for name in self.names:
            s = float(ALPHAS[name](row, micro))
            scores[name] = s
            w = self.weights[name] * mults.get(name, 1.0)
            num += w * s
            den += w
        score = clamp(num / den if den > 0 else 0.0, -1.0, 1.0)

        self._grade_pending(row)
        self._pending.append(PendingGrade(
            idx=self._bar_idx, close=row.get("close", 0.0),
            atr=max(row.get("atr", 0.0), 1e-12), scores=dict(scores), ensemble_score=score,
        ))
        self._score_hist.append(abs(score))
        self._bar_idx += 1
        self._adapt_threshold()

        self.last_regime, self.last_conf = regime, conf
        self.last_scores, self.last_score = scores, score
        return {
            "score": score, "scores": scores, "regime": regime, "regime_conf": conf,
            "threshold": self.threshold, "weights": dict(self.weights), "beta": self.beta,
        }

    # ------------------------------------------------------------- learning

    def _grade_pending(self, row: dict) -> None:
        c = row.get("close", 0.0)
        if c <= 0:
            return
        while self._pending and self._bar_idx - self._pending[0].idx >= self.horizon:
            p = self._pending.popleft()
            if p.close <= 0:
                continue
            ret = (c - p.close) / p.close
            norm = clamp(ret / max(p.atr / p.close, 1e-9), -2.5, 2.5)
            for name, s in p.scores.items():
                if abs(s) < 0.10:
                    continue
                payoff = math.copysign(min(abs(s), 1.0), s) * norm
                self.weights[name] *= math.exp(self.eta * clamp(payoff, -2.0, 2.0) * 0.25 / self.horizon)
                st = self.stats[name]
                st.calls += 1
                st.payoff_sum += payoff
                if s * ret > 0:
                    st.hits += 1
            self._renormalize()
            # calibrate beta: realized |move| vs predicted |score| * atr_pct * sqrt(H)
            pred = abs(p.ensemble_score) * (p.atr / p.close) * math.sqrt(self.horizon)
            if pred > 1e-9 and abs(p.ensemble_score) > 0.15:
                ratio = clamp(abs(ret) / pred, 0.1, 4.0)
                self.beta += 0.03 * (ratio - self.beta)
                self.beta = clamp(self.beta, 0.3, 3.0)
            self.graded += 1

    def _renormalize(self) -> None:
        total = sum(self.weights.values())
        if total <= 0:
            eq = 1.0 / len(self.names)
            self.weights = {k: eq for k in self.names}
            return
        # Shrink slightly toward uniform each grade: prevents permanent
        # lock-in on one alpha and keeps re-adaptation fast in new regimes.
        lam = 0.004
        n = len(self.names)
        uni = 1.0 / n
        for k in self.weights:
            self.weights[k] = (1 - lam) * (self.weights[k] / total) + lam * uni
        # Project onto the simplex with an exact per-weight floor: clamp, then
        # rescale only the mass above the floor so the floor is never violated.
        excess = {k: max(self.weights[k] - self.floor, 0.0) for k in self.weights}
        tot_excess = sum(excess.values())
        free = 1.0 - n * self.floor
        if tot_excess <= 1e-12 or free <= 0:
            self.weights = {k: uni for k in self.weights}
            return
        for k in self.weights:
            self.weights[k] = self.floor + excess[k] * (free / tot_excess)

    def _adapt_threshold(self) -> None:
        if not self.threshold_adapt or len(self._score_hist) < 120:
            self.threshold = self.base_threshold
            return
        # Choose the |score| quantile whose exceedance rate matches the target
        # trade rate; keep it tethered to the configured base.
        opportunities = self.bars_per_hour
        p = clamp(1.0 - self.target_rate / opportunities, 0.5, 0.995)
        hist = sorted(self._score_hist)
        q = hist[min(int(p * len(hist)), len(hist) - 1)]
        self.threshold = clamp(0.5 * q + 0.5 * self.base_threshold,
                               0.55 * self.base_threshold, 0.92)

    # ------------------------------------------------------------- gating

    def entry_ok(self, score: float, row: dict, fees_roundtrip: float,
                 spread_bps: float, slippage_bps: float) -> tuple[bool, str]:
        if abs(score) < self.threshold:
            return False, f"score {score:+.2f} below threshold {self.threshold:.2f}"
        atr_pct = row.get("atr_pct", 0.0)
        if not math.isfinite(atr_pct) or atr_pct <= 0:
            return False, "no volatility estimate"
        predicted_move = self.beta * abs(score) * atr_pct * math.sqrt(self.horizon)
        cost = fees_roundtrip + (spread_bps + 2 * slippage_bps) / 10_000.0
        if predicted_move < cost * self.cost_multiple:
            return False, (f"edge {predicted_move*100:.3f}% < cost x{self.cost_multiple:.1f} "
                           f"({cost*100:.3f}%)")
        return True, "ok"

    def micro_confirms(self, score: float, micro: dict) -> bool:
        """Order-flow veto: block entries that lean hard against live flow."""
        lean = 0.6 * micro.get("flow", 0.0) + 0.4 * micro.get("obi", 0.0)
        return not (abs(lean) > 0.35 and lean * score < 0)

    # ------------------------------------------------------------- exposure

    def snapshot(self) -> dict:
        return {
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "scores": {k: round(v, 3) for k, v in self.last_scores.items()},
            "score": round(self.last_score, 4),
            "regime": self.last_regime,
            "regime_conf": round(self.last_conf, 3),
            "threshold": round(self.threshold, 4),
            "beta": round(self.beta, 3),
            "graded": self.graded,
            "alpha_stats": {
                k: {"calls": st.calls, "hit_rate": round(st.hit_rate, 4),
                    "payoff": round(st.payoff_sum, 3),
                    "is_micro": k in MICRO_ALPHAS}
                for k, st in self.stats.items()
            },
        }
