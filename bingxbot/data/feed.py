"""Market state per symbol + feed implementations.

`MarketState` is the single source of truth the strategy reads: closed bars,
best bid/ask, order-book imbalance, and short-horizon trade-flow statistics
(aggressor volume delta, CVD slope). Feeds (live BingX or synthetic) mutate it
and emit compact events into an asyncio queue for the trader loop:

    ("bar", symbol)    - a bar just closed, run the full decision pass
    ("tick", symbol)   - trade tick(s) arrived, manage exits / micro state
"""
from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from collections import deque

from ..exchange.models import BookTop, Candle, DepthSnapshot, Tick
from ..exchange.rest import BingXRest
from ..exchange.ws import BingXMarketWS
from ..util import Ewma, interval_ms, now_ms
from .candles import CandleSeries

log = logging.getLogger("feed")

FLOW_WINDOW_S = 30.0
# seed enough base bars that the full 1m/5m/15m/1h ladder is populated from the
# first tick (the 1h rung on a 1m base needs ~8h of history to be meaningful).
MTF_SEED_MIN = 1200


class MarketState:
    def __init__(self, symbol: str, capacity: int = 3000):
        self.symbol = symbol
        self.candles = CandleSeries(capacity)
        self.book: BookTop | None = None
        self.depth: DepthSnapshot | None = None
        self.last_price = 0.0
        self.last_tick_ts = 0
        self.spread_bps = Ewma(0.2)
        self.obi = Ewma(0.15)              # order-book imbalance [-1, 1]
        self._flow: deque[tuple[float, float]] = deque()   # (mono_ts, signed qty)
        self.cvd = 0.0
        self.cvd_slope = Ewma(0.08)        # signed-volume EWMA per tick batch
        self.ticks_per_s = Ewma(0.1)
        self._last_tick_mono = 0.0
        # derivatives context (funding / open interest / mark), live only
        self.funding_rate: float | None = None
        self.mark: float | None = None
        self.open_interest: float | None = None
        self._funding_hist: deque[float] = deque(maxlen=200)
        self._oi_hist: deque[float] = deque(maxlen=200)

    # -- mutation ----------------------------------------------------------

    def on_context(self, funding: float | None, mark: float | None, oi: float | None) -> None:
        if funding is not None and math.isfinite(funding):
            self.funding_rate = funding
            self._funding_hist.append(funding)
        if mark is not None and mark > 0:
            self.mark = mark
        if oi is not None and oi > 0:
            self.open_interest = oi
            self._oi_hist.append(oi)

    def context_snapshot(self) -> dict:
        """Derivatives context for the carry desk. All None/0 offline so those
        alphas stay dormant rather than firing on absent data."""
        fz = None
        if len(self._funding_hist) >= 20:
            arr = list(self._funding_hist)
            m = sum(arr) / len(arr)
            var = sum((x - m) ** 2 for x in arr) / len(arr)
            sd = math.sqrt(var) if var > 0 else 0.0
            fz = (self.funding_rate - m) / sd if sd > 1e-12 and self.funding_rate is not None else None
        oi_chg = None
        if len(self._oi_hist) >= 2 and self._oi_hist[-2] > 0:
            oi_chg = (self._oi_hist[-1] - self._oi_hist[-2]) / self._oi_hist[-2]
        return {
            "funding_rate": self.funding_rate,
            "funding_z": fz,
            "oi_change_pct": oi_chg,
            "mark": self.mark,
        }

    def on_tick(self, t: Tick) -> None:
        self.last_price = t.price
        self.last_tick_ts = t.ts
        signed = -t.qty if t.is_buyer_maker else t.qty
        self.cvd += signed
        mono = time.monotonic()
        self._flow.append((mono, signed))
        cutoff = mono - FLOW_WINDOW_S
        while self._flow and self._flow[0][0] < cutoff:
            self._flow.popleft()
        self.cvd_slope.update(signed)
        if self._last_tick_mono:
            dt = mono - self._last_tick_mono
            if dt > 0:
                self.ticks_per_s.update(min(1.0 / dt, 50.0))
        self._last_tick_mono = mono

    def on_book(self, b: BookTop) -> None:
        self.book = b
        self.spread_bps.update(b.spread_bps)
        if self.last_price <= 0:
            self.last_price = b.mid

    def on_depth(self, d: DepthSnapshot) -> None:
        self.depth = d
        self.obi.update(d.imbalance(10))

    # -- reads -------------------------------------------------------------

    def flow_imbalance(self) -> float:
        """Aggressor volume imbalance over the last FLOW_WINDOW_S, in [-1, 1]."""
        if not self._flow:
            return 0.0
        pos = sum(q for _, q in self._flow if q > 0)
        neg = -sum(q for _, q in self._flow if q < 0)
        tot = pos + neg
        return (pos - neg) / tot if tot > 0 else 0.0

    def mark_price(self) -> float:
        if self.book is not None:
            return self.book.mid
        return self.last_price

    def micro_snapshot(self) -> dict:
        return {
            "obi": self.obi.get(),
            "flow": self.flow_imbalance(),
            "spread_bps": self.spread_bps.get(),
            "cvd_slope": self.cvd_slope.get(),
            "ticks_per_s": self.ticks_per_s.get(),
        }


class BaseFeed:
    def __init__(self, symbols: list[str], interval: str):
        self.symbols = symbols
        self.interval = interval
        self.states: dict[str, MarketState] = {s: MarketState(s) for s in symbols}
        self.events: asyncio.Queue[tuple[str, str]] = asyncio.Queue(maxsize=20_000)
        self.started = False

    def _emit(self, kind: str, symbol: str) -> None:
        try:
            self.events.put_nowait((kind, symbol))
        except asyncio.QueueFull:
            try:  # drop the oldest event; state is already up to date anyway
                self.events.get_nowait()
                self.events.put_nowait((kind, symbol))
            except asyncio.QueueEmpty:
                pass

    async def start(self) -> None:
        raise NotImplementedError

    async def stop(self) -> None:
        raise NotImplementedError

    def healthy(self) -> bool:
        return self.started


class LiveFeed(BaseFeed):
    """Real BingX market data: REST seed + WebSocket streaming."""

    def __init__(self, rest: BingXRest, ws_url: str, symbols: list[str], interval: str, warmup_bars: int):
        super().__init__(symbols, interval)
        self.rest = rest
        self.warmup_bars = warmup_bars
        self.ws = BingXMarketWS(
            ws_url,
            on_kline=self._on_kline,
            on_tick=self._on_tick,
            on_book=self._on_book,
            on_depth=self._on_depth,
        )
        for s in symbols:
            self.ws.subscribe_symbol(s, interval)
        self._ctx_task: asyncio.Task | None = None

    async def start(self) -> None:
        for s in self.symbols:
            seed_n = min(max(self.warmup_bars + 60, MTF_SEED_MIN), 1440)
            candles = await self.rest.klines(s, self.interval, limit=seed_n)
            if candles:
                candles = candles[:-1]  # last row is the still-open bar
            n = self.states[s].candles.seed(candles)
            log.info("%s seeded %d bars (%s)", s, n, self.interval)
        await self.ws.start()
        self._ctx_task = asyncio.create_task(self._poll_context(), name="ctx-poller")
        self.started = True

    async def stop(self) -> None:
        if self._ctx_task:
            self._ctx_task.cancel()
            try:
                await self._ctx_task
            except (asyncio.CancelledError, Exception):
                pass
            self._ctx_task = None
        await self.ws.stop()
        self.started = False

    async def _poll_context(self) -> None:
        """Periodically pull funding rate, mark price and open interest so the
        carry desk has data. Best-effort; failures never disturb trading."""
        while True:
            for s in self.symbols:
                try:
                    prem = await self.rest.premium_index(s)
                    oi = await self.rest.open_interest(s)
                    self.states[s].on_context(prem.get("funding_rate"), prem.get("mark"), oi)
                except Exception as e:  # noqa: BLE001
                    log.debug("context poll %s: %s", s, e)
            await asyncio.sleep(45)

    def healthy(self) -> bool:
        return self.started and self.ws.connected

    async def _on_kline(self, symbol: str, c: Candle) -> None:
        st = self.states.get(symbol)
        if st is None:
            return
        if c.closed:
            if st.candles.append(c):
                self._emit("bar", symbol)
        else:
            st.candles.update_partial(c)

    async def _on_tick(self, symbol: str, t: Tick) -> None:
        st = self.states.get(symbol)
        if st is None:
            return
        st.on_tick(t)
        self._emit("tick", symbol)

    async def _on_book(self, symbol: str, b: BookTop) -> None:
        st = self.states.get(symbol)
        if st is not None:
            st.on_book(b)

    async def _on_depth(self, symbol: str, d: DepthSnapshot) -> None:
        st = self.states.get(symbol)
        if st is not None:
            st.on_depth(d)


class SyntheticFeed(BaseFeed):
    """Offline demo feed: regime-switching random walk rendered as live ticks.

    Lets the whole stack (paper trading, dashboard) run with zero network.
    `speed` > 1 compresses time (a 1m bar arrives every 60/speed seconds).
    """

    def __init__(self, symbols: list[str], interval: str, warmup_bars: int = 600,
                 speed: float = 1.0, seed: int | None = None):
        super().__init__(symbols, interval)
        self.speed = max(0.1, speed)
        self.warmup_bars = warmup_bars
        self._rng = random.Random(seed)
        self._tasks: list[asyncio.Task] = []
        self._px: dict[str, float] = {}
        self._regime_left: dict[str, int] = {}
        self._drift: dict[str, float] = {}
        self._vol: dict[str, float] = {}
        base = {"BTC-USDT": 65_000.0, "ETH-USDT": 3_400.0}
        for i, s in enumerate(symbols):
            self._px[s] = base.get(s, 100.0 * (i + 1))
            self._roll_regime(s)

    def _roll_regime(self, s: str) -> None:
        r = self._rng.random()
        if r < 0.30:
            self._drift[s], self._vol[s] = 0.00004, 0.0006      # trend up
        elif r < 0.60:
            self._drift[s], self._vol[s] = -0.00004, 0.0006     # trend down
        elif r < 0.90:
            self._drift[s], self._vol[s] = 0.0, 0.0004          # range
        else:
            self._drift[s], self._vol[s] = 0.0, 0.0014          # chop / vol spike
        self._regime_left[s] = self._rng.randint(60, 240)       # bars

    async def start(self) -> None:
        from .history import synthetic_candles
        seed_n = max(self.warmup_bars, MTF_SEED_MIN)
        for s in self.symbols:
            candles = synthetic_candles(s, self.interval, seed_n,
                                        seed=self._rng.randint(0, 10**9),
                                        start_price=self._px[s])
            self.states[s].candles.seed(candles)
            self._px[s] = candles[-1].close
            self._tasks.append(asyncio.create_task(self._run_symbol(s), name=f"synth-{s}"))
        self.started = True
        log.info("synthetic feed started (speed x%.1f)", self.speed)

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks.clear()
        self.started = False

    async def _run_symbol(self, s: str) -> None:
        st = self.states[s]
        bar_ms = interval_ms(self.interval)
        bar_wall_s = (bar_ms / 1000.0) / self.speed
        ts = (now_ms() // bar_ms) * bar_ms
        while True:
            self._regime_left[s] -= 1
            if self._regime_left[s] <= 0:
                self._roll_regime(s)
            o = self._px[s]
            hi, lo, px = o, o, o
            n_ticks = self._rng.randint(20, 45)
            vol_bar = 0.0
            for i in range(n_ticks):
                ret = self._drift[s] + self._vol[s] * self._rng.gauss(0, 1) / math.sqrt(n_ticks)
                px = max(px * (1 + ret), 1e-9)
                hi, lo = max(hi, px), min(lo, px)
                qty = abs(self._rng.gauss(0, 1)) * 0.6 + 0.02
                vol_bar += qty
                is_seller = self._rng.random() < (0.5 - self._drift[s] * 3000)
                st.on_tick(Tick(ts=now_ms(), price=px, qty=qty, is_buyer_maker=is_seller))
                half_spread = px * 0.00002 + abs(self._rng.gauss(0, px * 0.00001))
                st.on_book(BookTop(ts=now_ms(), bid=px - half_spread, bid_qty=self._rng.uniform(1, 8),
                                   ask=px + half_spread, ask_qty=self._rng.uniform(1, 8)))
                if i % 6 == 0:
                    lean = 0.5 + self._drift[s] * 2500 + self._rng.gauss(0, 0.08)
                    bids = [(px - half_spread * (k + 1), self._rng.uniform(0.5, 6) * lean) for k in range(10)]
                    asks = [(px + half_spread * (k + 1), self._rng.uniform(0.5, 6) * (1 - lean)) for k in range(10)]
                    st.on_depth(DepthSnapshot(ts=now_ms(), bids=bids, asks=asks))
                self._emit("tick", s)
                st.candles.update_partial(Candle(ts=ts, open=o, high=hi, low=lo, close=px, volume=vol_bar, closed=False))
                await asyncio.sleep(bar_wall_s / n_ticks)
            self._px[s] = px
            closed = Candle(ts=ts, open=o, high=hi, low=lo, close=px, volume=vol_bar, closed=True)
            if st.candles.append(closed):
                self._emit("bar", s)
            ts += bar_ms
