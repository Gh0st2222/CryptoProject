"""Bar-pipeline starvation guard: the failure where prices keep streaming but
no bar ever closes — the brain starves and the terminal freezes silently."""
import asyncio

from bingxbot.data.feed import LiveFeed, bars_overdue
from bingxbot.exchange.models import Candle
from bingxbot.exchange.ws import BingXMarketWS
from bingxbot.util import now_ms


def test_bars_overdue_thresholds():
    iv = 60_000
    assert not bars_overdue(0, 10 * iv, iv), "an unseeded series is not 'stale'"
    assert not bars_overdue(100 * iv, 102 * iv, iv), "2 intervals old is healthy (open-stamped bars)"
    assert not bars_overdue(100 * iv, 103 * iv, iv), "3x is the tolerance edge"
    assert bars_overdue(100 * iv, 103 * iv + 1, iv), "beyond 3x a close was missed"


async def test_kline_ts_zero_never_poisons_rollover():
    """A payload row without a resolvable bar time must be dropped BEFORE the
    rollover tracker: one ts=0 row used to pin the tracker at 0 so `ts > prev`
    never fired again — no bar would ever close while every other channel
    streamed happily. Good rows after a bad one must still roll over."""
    seen = []

    async def rec(symbol, c):
        seen.append((c.ts, c.closed))

    ws = BingXMarketWS("wss://example", on_kline=rec)
    bad = {"o": "1", "h": "1", "l": "1", "c": "1", "v": "0"}          # no T/t at all
    live1 = {"T": 60_000, "o": "1", "h": "2", "l": "1", "c": "2", "v": "3"}
    live2 = {"T": 120_000, "o": "2", "h": "3", "l": "2", "c": "3", "v": "4"}
    await ws._handle_kline("BTC-USDT", [bad])
    await ws._handle_kline("BTC-USDT", [live1])
    await ws._handle_kline("BTC-USDT", [bad])       # mid-stream garbage, same story
    await ws._handle_kline("BTC-USDT", [live2])
    closed = [ts for ts, c in seen if c]
    assert closed == [60_000], "the first bar must close exactly once on rollover"
    assert all(ts > 0 for ts, _ in seen), "ts=0 rows must never reach the engine"


class _FakeRest:
    def __init__(self, candles):
        self.candles = candles
        self.calls = 0

    async def klines(self, symbol, interval, limit=500):
        self.calls += 1
        return self.candles[-int(limit):]


async def test_resync_backfills_missed_bars():
    """When the stream starves, the REST truth path heals the series: every
    missed CLOSED bar is appended (still-open last row dropped) and a 'bar'
    event fires per bar so the engine evaluates them in order."""
    iv = 60_000
    t0 = (now_ms() // iv) * iv - 10 * iv
    hist = [Candle(ts=t0 + i * iv, open=1, high=2, low=1, close=1.5, volume=3)
            for i in range(8)]                       # last row plays the still-open bar
    feed = LiveFeed(_FakeRest(hist), "wss://example", ["BTC-USDT"], "1m", warmup_bars=5)
    st = feed.states["BTC-USDT"]
    st.candles.append(hist[0])
    st.candles.append(hist[1])                       # ...then the stream died
    added = await feed._resync_symbol("BTC-USDT")
    assert added == 5, "bars 2..6 backfilled; the still-open row is dropped"
    assert st.candles.last_ts == hist[6].ts
    events = []
    while not feed.events.empty():
        events.append(feed.events.get_nowait())
    assert events == [("bar", "BTC-USDT")] * 5


async def test_resync_is_idempotent_when_fresh():
    """A resync on an already-current series stores nothing and emits nothing —
    the watchdog can misfire without consequences."""
    iv = 60_000
    t0 = (now_ms() // iv) * iv - 3 * iv
    hist = [Candle(ts=t0 + i * iv, open=1, high=2, low=1, close=1.5, volume=3)
            for i in range(3)]
    feed = LiveFeed(_FakeRest(hist), "wss://example", ["BTC-USDT"], "1m", warmup_bars=5)
    st = feed.states["BTC-USDT"]
    for c in hist[:-1]:
        st.candles.append(c)
    added = await feed._resync_symbol("BTC-USDT")
    assert added == 0
    assert feed.events.empty()
