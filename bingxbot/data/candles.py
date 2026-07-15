"""Fixed-capacity numpy ring buffer of OHLCV bars.

Indicators consume contiguous numpy views of the CLOSED bars; the in-progress
bar is tracked separately so partial data never contaminates the series.
"""
from __future__ import annotations

import numpy as np

from ..exchange.models import Candle


class CandleSeries:
    def __init__(self, capacity: int = 3000):
        self.capacity = capacity
        self._ts = np.zeros(capacity, dtype=np.int64)
        self._o = np.zeros(capacity, dtype=np.float64)
        self._h = np.zeros(capacity, dtype=np.float64)
        self._l = np.zeros(capacity, dtype=np.float64)
        self._c = np.zeros(capacity, dtype=np.float64)
        self._v = np.zeros(capacity, dtype=np.float64)
        self._n = 0            # bars stored (<= capacity)
        self._head = 0         # next write index
        self.partial: Candle | None = None

    def __len__(self) -> int:
        return self._n

    @property
    def last_ts(self) -> int:
        return int(self._ts[(self._head - 1) % self.capacity]) if self._n else 0

    @property
    def last_close(self) -> float:
        return float(self._c[(self._head - 1) % self.capacity]) if self._n else 0.0

    def append(self, c: Candle) -> bool:
        """Append a closed bar. Ignores duplicates/out-of-order. True if stored."""
        if self._n and c.ts <= self.last_ts:
            return False
        i = self._head
        self._ts[i], self._o[i], self._h[i] = c.ts, c.open, c.high
        self._l[i], self._c[i], self._v[i] = c.low, c.close, c.volume
        self._head = (i + 1) % self.capacity
        self._n = min(self._n + 1, self.capacity)
        if self.partial is not None and self.partial.ts <= c.ts:
            self.partial = None
        return True

    def update_partial(self, c: Candle) -> None:
        if self._n == 0 or c.ts > self.last_ts:
            self.partial = c

    def seed(self, candles: list[Candle]) -> int:
        added = 0
        for c in sorted(candles, key=lambda x: x.ts):
            added += 1 if self.append(c) else 0
        return added

    def _view(self, arr: np.ndarray, n: int) -> np.ndarray:
        n = min(n, self._n)
        if n == 0:
            return arr[:0].copy()
        start = (self._head - n) % self.capacity
        if start + n <= self.capacity:
            return arr[start:start + n].copy()
        k = self.capacity - start
        return np.concatenate((arr[start:], arr[:n - k]))

    def arrays(self, n: int | None = None) -> dict[str, np.ndarray]:
        """Last `n` closed bars, oldest first."""
        n = self._n if n is None else n
        return {
            "ts": self._view(self._ts, n),
            "open": self._view(self._o, n),
            "high": self._view(self._h, n),
            "low": self._view(self._l, n),
            "close": self._view(self._c, n),
            "volume": self._view(self._v, n),
        }

    def tail(self, n: int) -> list[Candle]:
        a = self.arrays(n)
        return [
            Candle(int(a["ts"][i]), float(a["open"][i]), float(a["high"][i]),
                   float(a["low"][i]), float(a["close"][i]), float(a["volume"][i]))
            for i in range(len(a["ts"]))
        ]
