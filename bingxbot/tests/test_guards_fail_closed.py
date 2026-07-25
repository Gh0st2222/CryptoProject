"""Remaining fail-open guards, found by an AST sweep rather than by sampling.

The last three rounds all hit the same class: a bail-out guard written as a bare
COMPARISON. Every comparison with NaN is False, so the guard does not fire on
exactly the input it exists to reject. Fuzzing finds these one at a time; a
static scan over every function that takes a numeric parameter finds the set.

These are the ones that survived rounds seven to nine.
"""
import asyncio
import math

import pytest

from bingxbot.config import BotConfig, RiskConfig
from bingxbot.engine.carry import carry_entry_ok
from bingxbot.engine.portfolio import Portfolio
from bingxbot.exchange.models import LONG, ContractSpec, Position
from bingxbot.risk.manager import RiskManager

NAN, INF = float("nan"), float("inf")


# --------------------------------------------------------------- the risk gate

def test_can_enter_refuses_an_unreadable_book_or_equity():
    """`spread_bps > max` and `equity <= 0` are both False for NaN. The spread
    limit exists to stop us trading into a broken or one-sided book — which is
    exactly the situation that produces an unreadable spread."""
    rm = RiskManager(RiskConfig())
    for bad in ({"spread_bps": NAN}, {"spread_bps": INF},
                {"equity": NAN}, {"equity": INF}):
        kw = {"equity": 1000.0, "open_positions": 0, "spread_bps": 1.0}
        kw.update(bad)
        ok, why = rm.can_enter(**kw)
        assert ok is False, f"risk gate passed on {bad}: {why}"


def test_can_enter_still_passes_a_healthy_book():
    rm = RiskManager(RiskConfig())
    assert rm.can_enter(1000.0, 0, 1.0)[0] is True


def test_the_kill_switch_and_position_cap_still_win():
    """The new check must run alongside the existing gates, not replace them."""
    rm = RiskManager(RiskConfig())
    rm.manual_kill("test")
    assert rm.can_enter(1000.0, 0, 1.0)[0] is False
    rm.reset_kill()
    assert rm.can_enter(1000.0, 99, 1.0)[0] is False, "max open positions"


# ------------------------------------------------------------ the carry desk

def test_carry_entry_refuses_a_non_finite_funding_or_trend_read():
    """`abs(nan) < min_apr` and `er_4h >= trend_veto_er` are both False, so this
    waved through a carry entry it could not evaluate. The desk opens real
    positions on this verdict."""
    cfg = BotConfig().carry
    for bad in ({"apr": NAN}, {"apr": INF}, {"er_4h": NAN}, {"er_4h": INF}):
        kw = {"apr": 1.0, "er_4h": 0.1, "dir_4h": 0}
        kw.update(bad)
        ok, why = carry_entry_ok(cfg=cfg, **kw)
        assert ok is False, f"carry gate passed on {bad}: {why}"


def test_carry_entry_still_takes_a_good_setup():
    cfg = BotConfig().carry
    assert carry_entry_ok(apr=1.0, er_4h=0.1, dir_4h=0, cfg=cfg)[0] is True


# ---------------------------------------------------------- live order paths

def _live_broker():
    from bingxbot.engine.brokers import LiveBroker

    class _Rest:
        def __init__(self):
            self.orders = []

        async def place_order(self, **kw):
            self.orders.append(kw)
            return {"orderId": "o1"}

        async def cancel_order(self, symbol, order_id):
            return {}

    rest = _Rest()
    return LiveBroker(rest, Portfolio(1000.0, mode="live"),
                      {"BTC-USDT": ContractSpec("BTC-USDT")}, BotConfig()), rest


def test_arming_a_maker_exit_declines_instead_of_raising():
    """round_step floors, and math.floor RAISES on NaN — so this refusal path
    used to throw. The caller catches it, so no bad order ever reached the
    exchange, but 'returns False' is the documented contract and an order path
    should decline cleanly rather than rely on someone else's except."""
    async def go():
        b, rest = _live_broker()
        for qty, price in ((1.0, NAN), (NAN, 100.0), (1.0, INF), (INF, 100.0)):
            assert await b.arm_maker_exit("BTC-USDT", LONG, qty, price) is False
        assert rest.orders == [], "nothing may be sent to the exchange"
        assert await b.arm_maker_exit("BTC-USDT", LONG, 1.0, 110.0) is True
        assert len(rest.orders) == 1
    asyncio.run(go())


def test_the_paper_broker_declines_the_same_way():
    from bingxbot.engine.brokers import PaperBroker

    async def go():
        br = PaperBroker(Portfolio(1000.0, mode="paper"), {},
                         {"BTC-USDT": ContractSpec("BTC-USDT")},
                         taker_fee=5e-4, slippage_bps=0.0)
        for qty, price in ((1.0, NAN), (NAN, 100.0), (1.0, INF)):
            assert await br.arm_maker_exit("BTC-USDT", LONG, qty, price) is False
    asyncio.run(go())


# ------------------------------------------------- already correct, pinned

def test_scale_out_already_fails_closed():
    """`not 0.0 < frac < 1.0` inverts the comparison, so NaN lands in the
    REJECT branch. Pinned so a future refactor to `if frac <= 0` does not
    quietly turn it into a fail-open."""
    pf = Portfolio(1000.0, mode="paper")
    pf.positions["X"] = Position(symbol="X", side=LONG, qty=1.0,
                                 entry_price=100.0, opened_ts=0)
    assert pf.scale_out("X", NAN, 110.0, 0, 0.1, "t") is None
    assert pf.positions["X"].qty == pytest.approx(1.0), "untouched"
