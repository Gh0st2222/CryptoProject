"""Historical kline store: paginated BingX download with a gzip CSV disk
cache, plus a regime-switching synthetic generator for offline work.
"""
from __future__ import annotations

import csv
import gzip
import io
import logging
import math
import random
from pathlib import Path

from ..exchange.models import Candle
from ..exchange.rest import BingXRest
from ..util import interval_ms, now_ms

log = logging.getLogger("history")

MAX_PAGE = 1440  # BingX v3 klines hard limit per request


class HistoryStore:
    def __init__(self, rest: BingXRest | None, data_dir: str | Path = "data_cache"):
        self.rest = rest
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, interval: str) -> Path:
        return self.dir / f"{symbol}_{interval}.csv.gz"

    # ------------------------------------------------------------- cache io

    def _load_cache(self, symbol: str, interval: str) -> list[Candle]:
        p = self._path(symbol, interval)
        if not p.exists():
            return []
        try:
            with gzip.open(p, "rt", newline="") as f:
                return [
                    Candle(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]))
                    for r in csv.reader(f)
                ]
        except (OSError, ValueError, IndexError):
            log.warning("cache unreadable, discarding: %s", p)
            return []

    def _save_cache(self, symbol: str, interval: str, candles: list[Candle]) -> None:
        buf = io.StringIO()
        w = csv.writer(buf)
        for c in candles:
            w.writerow([c.ts, c.open, c.high, c.low, c.close, c.volume])
        with gzip.open(self._path(symbol, interval), "wt", newline="") as f:
            f.write(buf.getvalue())

    @staticmethod
    def _merge(a: list[Candle], b: list[Candle]) -> list[Candle]:
        by_ts = {c.ts: c for c in a}
        by_ts.update({c.ts: c for c in b})
        return [by_ts[t] for t in sorted(by_ts)]

    # ------------------------------------------------------------- fetching

    async def _download(self, symbol: str, interval: str, start_ms: int, end_ms: int,
                        progress=None) -> list[Candle]:
        assert self.rest is not None
        step = interval_ms(interval)
        out: list[Candle] = []
        cursor = start_ms
        total_span = max(end_ms - start_ms, 1)
        while cursor < end_ms:
            page = await self.rest.klines(
                symbol, interval,
                start_ms=cursor,
                end_ms=min(cursor + step * MAX_PAGE, end_ms),
                limit=MAX_PAGE,
            )
            if not page:
                cursor += step * MAX_PAGE  # gap (delisting/maintenance): skip window
                continue
            out.extend(page)
            new_cursor = page[-1].ts + step
            cursor = new_cursor if new_cursor > cursor else cursor + step * MAX_PAGE
            if progress:
                progress(min((cursor - start_ms) / total_span, 1.0))
        return out

    async def get_range(self, symbol: str, interval: str, start_ms: int, end_ms: int,
                        progress=None) -> list[Candle]:
        """Return candles covering [start_ms, end_ms], using/extending the cache."""
        end_ms = min(end_ms, now_ms())
        cached = self._load_cache(symbol, interval)
        if self.rest is None:
            return [c for c in cached if start_ms <= c.ts <= end_ms]

        need_head = not cached or start_ms < cached[0].ts
        need_tail = not cached or end_ms > cached[-1].ts + interval_ms(interval)
        if need_head or need_tail:
            fetch_start = start_ms if not cached or need_head else cached[-1].ts
            fetch_end = end_ms if not cached or need_tail else cached[0].ts
            if need_head and need_tail:
                fetch_start, fetch_end = start_ms, end_ms
            fresh = await self._download(symbol, interval, fetch_start, fetch_end, progress)
            cached = self._merge(cached, fresh)
            self._save_cache(symbol, interval, cached)
            log.info("%s %s cache now %d bars", symbol, interval, len(cached))
        return [c for c in cached if start_ms <= c.ts <= end_ms]


# ---------------------------------------------------------------- synthetic

def synthetic_candles(symbol: str, interval: str, bars: int, seed: int | None = None,
                      start_price: float | None = None) -> list[Candle]:
    """Regime-switching price process with real crypto microstructure:

    - TREND segments: drift + positively autocorrelated returns (momentum)
    - RANGE segments: Ornstein-Uhlenbeck pull toward an anchor (mean reversion)
    - CHOP segments: high vol, negative autocorrelation
    - volatility clustering across all segments

    This is what intraday crypto actually looks like statistically, so the
    adaptive ensemble has genuine, regime-dependent edges to discover in the
    demo and in tests. Deterministic per seed.
    """
    rng = random.Random(seed if seed is not None else hash(symbol) & 0xFFFF)
    base = {"BTC-USDT": 65_000.0, "ETH-USDT": 3_400.0}
    px = start_price or base.get(symbol, 250.0)
    step = interval_ms(interval)
    t0 = (now_ms() // step) * step - bars * step

    out: list[Candle] = []
    drift, vol, phi, kappa, left = 0.0, 0.0009, 0.0, 0.0, 0
    anchor = px
    vol_mult = 1.0
    prev_ret = 0.0
    for i in range(bars):
        if left <= 0:
            r = rng.random()
            if r < 0.30:      # trend up: drift + momentum persistence
                drift, vol, phi, kappa = 0.00015, 0.0009, 0.25, 0.0
            elif r < 0.60:    # trend down
                drift, vol, phi, kappa = -0.00015, 0.0009, 0.25, 0.0
            elif r < 0.90:    # range: OU mean reversion around anchor
                drift, vol, phi, kappa = 0.0, 0.0006, -0.08, 0.08
                anchor = px
            else:             # chop: loud and spiteful
                drift, vol, phi, kappa = 0.0, 0.0022, -0.12, 0.0
            left = rng.randint(60, 240)
        left -= 1
        vol_mult = max(0.5, min(2.0, vol_mult + rng.gauss(0, 0.03)))  # clustering

        o = px
        hi = lo = px
        n_sub = 6
        bar_ret_accum = 0.0
        for _ in range(n_sub):
            pull = -kappa * (px / anchor - 1.0) if kappa > 0 else 0.0
            ret = (drift / n_sub + pull / n_sub + phi * prev_ret / n_sub
                   + vol * vol_mult * rng.gauss(0, 1) / math.sqrt(n_sub))
            px = max(px * (1 + ret), 1e-9)
            bar_ret_accum += ret
            hi, lo = max(hi, px), min(lo, px)
        prev_ret = bar_ret_accum
        body = abs(px - o) / max(o, 1e-9)
        volume = (40.0 + 4000.0 * body + abs(rng.gauss(0, 12))) * (1.5 if vol > 0.001 else 1.0)
        out.append(Candle(ts=t0 + i * step, open=o, high=hi, low=lo, close=px, volume=volume))
    return out
