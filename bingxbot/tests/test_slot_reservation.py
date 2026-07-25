"""max_open_positions must count RESERVED slots, not just filled ones.

A resting limit entry has not opened a position yet, so counting only
`pf.positions` at decision time lets several of them fill and blow through the
cap. That was already fixed for the signal engine's own pendings — but the
carry desk trades symbols deliberately kept OUT of `eng.ctx`, so its in-flight
entries stayed invisible, and it rests an entry for up to 120 seconds.
"""
import asyncio

import pytest

from bingxbot.config import BotConfig
from bingxbot.engine.portfolio import Portfolio
from bingxbot.exchange.models import LONG, Position
from bingxbot.risk.manager import RiskManager


def _engine(tmp_path, symbols=("BTC-USDT",)):
    from bingxbot.data.feed import SyntheticFeed
    from bingxbot.engine.brokers import PaperBroker
    from bingxbot.engine.trader import TraderEngine
    cfg = BotConfig()
    cfg.symbols = list(symbols)
    feed = SyntheticFeed(cfg.symbols, "15m", warmup_bars=10, seed=1)
    pf = Portfolio(1000.0, mode="paper")
    br = PaperBroker(pf, feed.states, {}, taker_fee=5e-4, slippage_bps=0.0)
    return TraderEngine(cfg, feed, br, pf, RiskManager(cfg.risk), {}), pf


def test_a_carry_reservation_consumes_a_slot(tmp_path):
    eng, pf = _engine(tmp_path)
    assert eng.pending_entries() == 0
    eng.carry_pending = 1
    assert eng.pending_entries() == 1, "the desk's in-flight entry holds a slot"


def test_the_reservation_is_what_stops_the_cap_being_exceeded(tmp_path):
    """Concretely: with max_open_positions=3, two filled plus one carry order
    resting must leave NO room for the signal engine."""
    eng, pf = _engine(tmp_path)
    eng.cfg.risk.max_open_positions = 3
    for s in ("ETH-USDT", "SOL-USDT"):
        pf.positions[s] = Position(symbol=s, side=LONG, qty=1.0,
                                   entry_price=100.0, opened_ts=0)
    used = len(pf.positions) + eng.pending_entries()
    assert eng.risk.can_enter(1000.0, used, 1.0)[0] is True, "3rd slot is free"

    eng.carry_pending = 1
    used = len(pf.positions) + eng.pending_entries()
    ok, why = eng.risk.can_enter(1000.0, used, 1.0)
    assert ok is False and "max open positions" in why, \
        "the carry desk already reserved the last slot"


async def test_a_failed_carry_entry_never_leaks_the_reservation(tmp_path):
    """A leaked counter would permanently shrink the account's capacity, and
    nothing would ever put it back."""
    from bingxbot.engine.carry import CarryDesk
    eng, pf = _engine(tmp_path)

    class _Boom:
        async def open_position(self, *a, **kw):
            raise RuntimeError("exchange said no")

    eng.broker = _Boom()

    class _Orch:
        cfg = eng.cfg
        engine = eng
        specs: dict = {}
        _notify = None

    desk = CarryDesk(_Orch())
    row = {"symbol": "DOGE-USDT", "mark": 0.1, "atr_pct_4h": 0.01,
           "funding_rate": 0.001, "funding_apr": 1.0}
    with pytest.raises(RuntimeError):
        await desk._open(row, equity=10_000.0)
    assert eng.carry_pending == 0, "the slot was released even though the entry failed"


async def test_the_reservation_is_actually_held_during_the_wait(tmp_path):
    """The counter has to be up WHILE the limit rests — releasing it only after
    the await would leave exactly the window this fixes."""
    from bingxbot.engine.carry import CarryDesk
    eng, pf = _engine(tmp_path)
    seen = []

    class _Slow:
        async def open_position(self, *a, **kw):
            seen.append(eng.carry_pending)
            await asyncio.sleep(0)
            from bingxbot.exchange.models import OrderResult
            return OrderResult(ok=False, error="unfilled")

    eng.broker = _Slow()

    class _Orch:
        cfg = eng.cfg
        engine = eng
        specs: dict = {}
        _notify = None

    desk = CarryDesk(_Orch())
    await desk._open({"symbol": "DOGE-USDT", "mark": 0.1, "atr_pct_4h": 0.01,
                      "funding_rate": 0.001, "funding_apr": 1.0}, equity=10_000.0)
    assert seen == [1], "held during the order, not merely around it"
    assert eng.carry_pending == 0
