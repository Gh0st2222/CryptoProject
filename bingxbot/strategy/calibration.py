"""Online probability calibration.

A raw ensemble edge in [-1, 1] is not a probability. This maps |edge| (plus
regime and volatility context) to a calibrated P(the trade's direction is
correct over the horizon) using online logistic regression trained on realized
outcomes. Early on it leans on a mild monotone prior; as real outcomes arrive
it hands over to the learned model. Brier score is tracked so the UI can show
how well-calibrated the brain currently is.

Calibrated probability is what makes Kelly sizing meaningful, and it is its own
auto-correction: if the edge stops predicting, P(win) falls below 0.5 and the
gate simply refuses to trade.
"""
from __future__ import annotations

import math
from collections import deque

from ..util import clamp

REGIME_TREND = {"TREND_UP", "TREND_DOWN"}


def _sigmoid(z: float) -> float:
    if z >= 0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


class ProbabilityCalibrator:
    def __init__(self, lr: float = 0.03, l2: float = 1e-4):
        self.lr = lr
        self.l2 = l2
        # features: [|edge|, edge^2, is_trend, atr_pctile]
        self.w = [1.2, 0.0, 0.15, 0.0]
        self.b = 0.0
        self.n = 0
        self._brier = deque(maxlen=500)

    def _features(self, edge: float, regime: str, atr_pctile: float) -> list[float]:
        a = abs(clamp(edge, -1, 1))
        return [a, a * a, 1.0 if regime in REGIME_TREND else 0.0, clamp(atr_pctile, 0, 1)]

    def predict(self, edge: float, regime: str, atr_pctile: float) -> float:
        x = self._features(edge, regime, atr_pctile)
        z = self.b + sum(wi * xi for wi, xi in zip(self.w, x))
        p_model = _sigmoid(z)
        # Blend a mild monotone prior in until we have real outcomes.
        prior = 0.5 + 0.22 * abs(clamp(edge, -1, 1))
        blend = clamp(self.n / 200.0, 0.0, 1.0)
        return clamp(blend * p_model + (1 - blend) * prior, 0.05, 0.95)

    def update(self, edge: float, regime: str, atr_pctile: float, won: bool) -> None:
        x = self._features(edge, regime, atr_pctile)
        z = self.b + sum(wi * xi for wi, xi in zip(self.w, x))
        p = _sigmoid(z)
        y = 1.0 if won else 0.0
        g = p - y
        self.b -= self.lr * g
        for i in range(len(self.w)):
            self.w[i] -= self.lr * (g * x[i] + self.l2 * self.w[i])
        self._brier.append((p - y) ** 2)
        self.n += 1

    @property
    def brier(self) -> float:
        return sum(self._brier) / len(self._brier) if self._brier else 0.25

    def snapshot(self) -> dict:
        return {"n": self.n, "brier": round(self.brier, 4),
                "skill": round(clamp((0.25 - self.brier) / 0.25, -1, 1), 3)}
