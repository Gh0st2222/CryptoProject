import asyncio

import pytest

from bingxbot.config import BotConfig
from bingxbot.data.feed import SyntheticFeed
from bingxbot.engine.brokers import PaperBroker
from bingxbot.engine.portfolio import Portfolio
from bingxbot.engine.trader import TraderEngine
from bingxbot.exchange.models import LONG, SHORT, BookTop, ContractSpec
from bingxbot.risk.manager import RiskManager, SizedOrder


class _FakeState:
    def __init__(self, bid=100.0, ask=100.1):
        self.book = BookTop(ts=0, bid=bid, bid_qty=1, ask=ask, ask_qty=1)
        self.last_price = (bid + ask) / 2

    class candles:
        last_close = 100.05


@pytest.mark.asyncio
async def test_paper_broker_roundtrip_accounting():
    pf = Portfolio(10_000.0, mode="paper")
    states = {"BTC-USDT": _FakeState(bid=100.0, ask=100.1)}
    broker = PaperBroker(pf, states, {"BTC-USDT": ContractSpec("BTC-USDT")},
                         taker_fee=0.0005, slippage_bps=0.0)
    sized = SizedOrder(qty=2.0, notional=200.2, leverage=1,
                       stop_price=99.0, take_profit=102.0, risk_amount=2.2)
    res = await broker.open_position("BTC-USDT", LONG, sized, "test", bar_ts=0)
    assert res.ok and res.filled_price == pytest.approx(100.1)      # buys lift the ask
    states["BTC-USDT"].book = BookTop(ts=1, bid=101.0, bid_qty=1, ask=101.1, ask_qty=1)
    res2 = await broker.close_position("BTC-USDT", "test exit")
    assert res2.ok and res2.filled_price == pytest.approx(101.0)    # sells hit the bid
    assert len(pf.trades) == 1
    t = pf.trades[0]
    gross = (101.0 - 100.1) * 2.0
    fees = 2.0 * 100.1 * 0.0005 + 2.0 * 101.0 * 0.0005
    assert t.pnl == pytest.approx(gross - fees, abs=1e-9)
    assert pf.equity() == pytest.approx(10_000.0 + t.pnl, abs=1e-9)
    assert not pf.positions


@pytest.mark.asyncio
async def test_paper_broker_short_side():
    pf = Portfolio(10_000.0, mode="paper")
    states = {"ETH-USDT": _FakeState(bid=2000.0, ask=2000.2)}
    broker = PaperBroker(pf, states, {"ETH-USDT": ContractSpec("ETH-USDT")},
                         taker_fee=0.0, slippage_bps=0.0)
    sized = SizedOrder(qty=1.0, notional=2000, leverage=1,
                       stop_price=2020.0, take_profit=1980.0, risk_amount=20)
    await broker.open_position("ETH-USDT", SHORT, sized, "t", 0)
    states["ETH-USDT"].book = BookTop(ts=1, bid=1990.0, bid_qty=1, ask=1990.2, ask_qty=1)
    await broker.close_position("ETH-USDT", "t")
    assert pf.trades[0].pnl == pytest.approx(2000.0 - 1990.2)


@pytest.mark.asyncio
async def test_trader_engine_end_to_end_on_synthetic_feed():
    """Boot the full realtime stack on the synthetic feed at high speed and
    prove bars flow through ensemble evaluation without errors."""
    cfg = BotConfig()
    cfg.symbols = ["BTC-USDT"]
    cfg.mode = "paper"
    cfg.strategy.interval = "1m"
    cfg.strategy.warmup_bars = 320
    feed = SyntheticFeed(cfg.symbols, "1m", warmup_bars=340, speed=1200.0, seed=3)
    pf = Portfolio(cfg.paper.starting_balance, mode="paper")
    broker = PaperBroker(pf, feed.states, {s: ContractSpec(s) for s in cfg.symbols},
                         taker_fee=cfg.exchange.taker_fee, slippage_bps=cfg.paper.slippage_bps)
    risk = RiskManager(cfg.risk)
    engine = TraderEngine(cfg, feed, broker, pf, risk,
                          {s: ContractSpec(s) for s in cfg.symbols})
    await engine.start()
    try:
        await asyncio.sleep(12)   # ~240 synthetic minutes at speed 1200
    finally:
        await engine.stop(flatten=True)
    ctx = engine.ctx["BTC-USDT"]
    assert len(feed.states["BTC-USDT"].candles) > 345, "bars did not accumulate"
    assert ctx.ensemble.graded > 50, "ensemble never graded alpha calls"
    assert ctx.last_eval, "no ensemble evaluation happened"
    snap = engine.snapshot()
    assert snap["portfolio"]["equity"] > 0
    assert snap["symbols"]["BTC-USDT"]["ensemble"]["weights"]
