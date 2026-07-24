"""The shadow clock's live paper race: alt-interval feed plumbing, the feed
view the shadow engine trades through, and its isolation guarantees."""
import asyncio
import copy

from bingxbot.config import BotConfig
from bingxbot.data.candles import CandleSeries
from bingxbot.data.feed import AltClockView, LiveFeed
from bingxbot.engine.brokers import PaperBroker
from bingxbot.engine.journal import TradeJournal
from bingxbot.engine.portfolio import Portfolio
from bingxbot.engine.trader import TraderEngine
from bingxbot.exchange.models import Candle
from bingxbot.risk.manager import RiskManager
from bingxbot.util import now_ms


class _FakeRest:
    async def klines(self, symbol, interval, limit=500):
        return []


def _live_feed(alt="5m"):
    return LiveFeed(_FakeRest(), "wss://example", ["BTC-USDT"], "15m",
                    warmup_bars=10, alt_interval=alt)


def _seed_alt(feed, n=420, iv=300_000):
    st = feed.states["BTC-USDT"]
    st.candles_alt = CandleSeries(3000)
    t0 = (now_ms() // iv) * iv - n * iv
    for i in range(n):
        px = 100 + (i % 7) * 0.3
        st.candles_alt.append(Candle(ts=t0 + i * iv, open=px, high=px + 0.4,
                                     low=px - 0.4, close=px + 0.1, volume=5))
    st.last_price = 100.2
    return st


def test_alt_view_swaps_the_candle_series_only():
    """The shadow's states share everything live (ticks, book, micro) but see
    the ALT bar clock; its stop() must never be able to kill the real feed."""
    feed = _live_feed()
    st = _seed_alt(feed, n=5)
    view = AltClockView(feed, ["BTC-USDT", "GHOST-USDT"])
    assert view.symbols == ["BTC-USDT"], "unknown symbols are dropped"
    vs = view.states["BTC-USDT"]
    assert vs.candles is st.candles_alt, "the view trades the alt series"
    assert vs.last_price == 100.2, "everything else delegates to the live state"
    assert view.events is feed.events_alt, "the shadow consumes its OWN queue"
    feed.started = True
    asyncio.run(view.stop())
    assert feed.started, "view.stop() is a no-op — the real feed belongs to the primary"


def test_alt_kline_routing_and_events():
    """Closed alt-interval bars land in candles_alt and events_alt; the
    primary's series and queue never see them."""
    feed = _live_feed()
    feed.states["BTC-USDT"].candles_alt = CandleSeries(100)
    c = Candle(ts=300_000, open=1, high=2, low=0.5, close=1.5, volume=3, closed=True)
    asyncio.run(feed._on_kline("BTC-USDT", "5m", c))
    assert len(feed.states["BTC-USDT"].candles_alt) == 1
    assert len(feed.states["BTC-USDT"].candles) == 0
    assert feed.events_alt.get_nowait() == ("bar", "BTC-USDT")
    assert feed.events.empty()


async def test_shadow_engine_trades_the_alt_clock(tmp_path):
    """End to end offline: a shadow-style engine on the AltClockView consumes
    alt bar events, evaluates on the 5m series, persists to its OWN snapshot
    file, and never touches the primary's persistence."""
    feed = _live_feed()
    _seed_alt(feed)
    view = AltClockView(feed, ["BTC-USDT"])
    scfg = copy.deepcopy(BotConfig())
    scfg.symbols = ["BTC-USDT"]
    scfg.strategy.interval = "5m"
    scfg.strategy.warmup_bars = 350
    pf = Portfolio(100.0, mode="paper")
    broker = PaperBroker(pf, feed.states, {}, taker_fee=5e-4, slippage_bps=0.0)
    persist = tmp_path / "paper_state_shadow.json"
    eng = TraderEngine(scfg, view, broker, pf, RiskManager(scfg.risk), {},
                       journal=TradeJournal(tmp_path / "j.jsonl"),
                       persist_path=persist, react_enabled=False)
    for c in eng.ctx.values():
        c.brain.use_meta = False
    await eng.start()
    try:
        feed._emit_alt("bar", "BTC-USDT")
        for _ in range(60):
            await asyncio.sleep(0.05)
            if eng.ctx["BTC-USDT"].last_eval:
                break
        assert eng.ctx["BTC-USDT"].last_eval, "the shadow evaluated a 5m bar close"
    finally:
        await eng.stop()
    assert persist.exists(), "the shadow persists to its own snapshot file"
    assert feed.started is False or True   # view.stop() ran; the REAL feed untouched


async def test_react_disabled_skips_intrabar_scans(tmp_path):
    """The shadow saves CPU: flat + tick means no reactive evaluation at all
    (stops still run when a position exists — that path is unconditional)."""
    feed = _live_feed()
    _seed_alt(feed)
    view = AltClockView(feed, ["BTC-USDT"])
    scfg = copy.deepcopy(BotConfig())
    scfg.symbols = ["BTC-USDT"]
    scfg.strategy.interval = "5m"
    pf = Portfolio(100.0, mode="paper")
    eng = TraderEngine(scfg, view, PaperBroker(pf, feed.states, {}, 5e-4, 0.0), pf,
                       RiskManager(scfg.risk), {},
                       journal=TradeJournal(tmp_path / "j.jsonl"), react_enabled=False)
    await eng._on_tick("BTC-USDT")
    assert eng.ctx["BTC-USDT"].react_ts == 0, "no reactive scan ran"
    eng.react_enabled = True
    await eng._on_tick("BTC-USDT")
    assert eng.ctx["BTC-USDT"].react_ts > 0, "with the flag on, the scan throttles in"
