"""Small shared helpers used across the bot."""
from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque


def now_ms() -> int:
    return int(time.time() * 1000)


def round_step(value: float, precision: int) -> float:
    """Round down to `precision` decimals (exchange quantity/price rules)."""
    if precision <= 0:
        return float(math.floor(value))
    factor = 10 ** precision
    return math.floor(value * factor + 1e-9) / factor


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def interval_ms(interval: str) -> int:
    unit = interval[-1]
    n = int(interval[:-1])
    return n * {"s": 1_000, "m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}[unit]


class Ewma:
    """Exponentially weighted moving average with lazy init."""

    __slots__ = ("alpha", "value")

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value: float | None = None

    def update(self, x: float) -> float:
        self.value = x if self.value is None else self.value + self.alpha * (x - self.value)
        return self.value

    def get(self, default: float = 0.0) -> float:
        return default if self.value is None else self.value


class RateLimiter:
    """Async token bucket. `rate` tokens per second, burst up to `burst`."""

    def __init__(self, rate: float, burst: int = 1):
        self.rate = rate
        self.burst = burst
        self._tokens = float(burst)
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(self.burst, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                await asyncio.sleep((1.0 - self._tokens) / self.rate)


class RollingStat:
    """Fixed-window rolling mean/std over a deque (small windows only)."""

    def __init__(self, size: int):
        self.buf: deque[float] = deque(maxlen=size)

    def add(self, x: float) -> None:
        self.buf.append(x)

    def mean(self) -> float:
        return sum(self.buf) / len(self.buf) if self.buf else 0.0

    def std(self) -> float:
        n = len(self.buf)
        if n < 2:
            return 0.0
        m = self.mean()
        return math.sqrt(sum((x - m) ** 2 for x in self.buf) / (n - 1))

    def __len__(self) -> int:
        return len(self.buf)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
