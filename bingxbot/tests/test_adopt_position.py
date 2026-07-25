"""Adopting a live position must recover its protective levels.

Live deliberately never restores local state — "the exchange is the truth" — so
EVERY live restart with an open position goes through adoption. Adopting with
stop_price=0 silently disabled the entire adaptive exit stack for that position:
`exits.manage` falls back to `risk = |entry - stop|`, which with stop=0 is the
entry price itself, so rr sits near 0.0001 forever and the breakeven move, the
chandelier trail, the give-back lock and the scale-out are all rr-gated and can
never fire again. The position was left to the exchange stop and the time stop,
and booked r_multiple=0 into the live stats the demotion rule reads.
"""
import pytest

from bingxbot.config import BotConfig
from bingxbot.engine.portfolio import Portfolio
from bingxbot.exchange.models import LONG, SHORT, ContractSpec


class _Rest:
    def __init__(self, positions, orders=None, fail_orders=False):
        self._positions, self._orders = positions, orders or []
        self.fail_orders = fail_orders

    async def positions(self):
        return self._positions

    async def balance(self):
        return {"equity": 1000.0, "balance": 1000.0}

    async def open_orders(self, symbol=None):
        if self.fail_orders:
            from bingxbot.exchange.errors import BingXError
            raise BingXError("boom")
        return self._orders


def _broker(positions, orders=None, **kw):
    from bingxbot.engine.brokers import LiveBroker
    return LiveBroker(_Rest(positions, orders, **kw), Portfolio(1000.0, mode="live"),
                      {"BTC-USDT": ContractSpec("BTC-USDT")}, BotConfig())


def _pos(side=LONG, amt=1.0):
    return [{"symbol": "BTC-USDT", "positionAmt": amt if side == LONG else -amt,
             "positionSide": side, "avgPrice": 100.0, "leverage": 3}]


async def test_adopted_position_recovers_its_stop_and_target():
    b = _broker(_pos(), orders=[
        {"side": "SELL", "positionSide": LONG, "type": "STOP_MARKET", "stopPrice": 95.0},
        {"side": "SELL", "positionSide": LONG, "type": "TAKE_PROFIT_MARKET", "stopPrice": 112.0},
    ])
    await b.reconcile(["BTC-USDT"])
    p = b.portfolio.positions["BTC-USDT"]
    assert p.stop_price == 95.0 and p.take_profit == 112.0
    assert p.init_risk == pytest.approx(5.0), "1R must be the real stop distance"


async def test_recovered_risk_is_what_makes_the_trail_able_to_arm():
    """The consequence, stated directly: with init_risk=0 the exit manager's rr
    is ~0 forever and every rr-gated exit is dead."""
    from bingxbot.config import RiskConfig
    from bingxbot.strategy.exits import AdaptiveExitManager

    b = _broker(_pos(), orders=[
        {"side": "SELL", "positionSide": LONG, "type": "STOP_MARKET", "stopPrice": 95.0}])
    await b.reconcile(["BTC-USDT"])
    p = b.portfolio.positions["BTC-USDT"]
    ex = AdaptiveExitManager(RiskConfig())
    row = {"atr": 1.0, "mtf_bias": 0.0, "dc_hi": 0.0, "dc_lo": 0.0}
    # price up 2R (100 -> 110 with a 5-wide stop): breakeven must engage
    ex.manage(p, 110.0, 110.0, 100.0, 1.0, row, 0.4, 0.3, "TREND_UP", 5)
    assert p.breakeven_moved is True, "with real 1R the exit stack works"

    # the old behaviour, for contrast
    p.init_risk, p.stop_price, p.breakeven_moved = 0.0, 0.0, False
    ex.manage(p, 110.0, 110.0, 100.0, 1.0, row, 0.4, 0.3, "TREND_UP", 5)
    assert p.breakeven_moved is False, "stop=0 makes 1R the entry price: rr never arrives"


async def test_a_short_reads_the_orders_on_its_own_closing_side():
    b = _broker(_pos(side=SHORT), orders=[
        {"side": "BUY", "positionSide": SHORT, "type": "STOP_MARKET", "stopPrice": 105.0},
        {"side": "SELL", "positionSide": LONG, "type": "STOP_MARKET", "stopPrice": 1.0},
    ])
    await b.reconcile(["BTC-USDT"])
    p = b.portfolio.positions["BTC-USDT"]
    assert p.side == SHORT and p.stop_price == 105.0, "the other side's order is ignored"


async def test_a_resting_maker_target_is_re_adopted_not_orphaned():
    """A post-only reduce-only limit is the maker exit. After a restart it is
    still working on the book, so the engine has to know about it — otherwise it
    would market-close on touch and pay taker on an order already resting."""
    b = _broker(_pos(), orders=[
        {"side": "SELL", "positionSide": LONG, "type": "STOP_MARKET", "stopPrice": 95.0},
        {"side": "SELL", "positionSide": LONG, "type": "LIMIT", "price": 111.0,
         "orderId": "abc123", "reduceOnly": "true"},
    ])
    await b.reconcile(["BTC-USDT"])
    assert b.portfolio.positions["BTC-USDT"].take_profit == 111.0
    assert b.maker_exit_price("BTC-USDT") == 111.0, "re-armed, not orphaned"


async def test_a_manual_limit_order_is_never_mistaken_for_our_target():
    """Only a REDUCE-ONLY limit is ours. Adopting a user's own resting order
    would mean cancel_maker_exit() later pulls an order we never placed."""
    b = _broker(_pos(), orders=[
        {"side": "SELL", "positionSide": LONG, "type": "STOP_MARKET", "stopPrice": 95.0},
        {"side": "SELL", "positionSide": LONG, "type": "LIMIT", "price": 130.0,
         "orderId": "users-own", "reduceOnly": "false"},
    ])
    await b.reconcile(["BTC-USDT"])
    assert b.maker_exit_price("BTC-USDT") == 0.0, "not ours, not touched"
    assert b.portfolio.positions["BTC-USDT"].take_profit == 0.0


async def test_a_position_with_no_stop_is_adopted_but_never_invents_one():
    """Inventing a stop would invent an init_risk, and every R statistic
    downstream would be fiction. Adopt honestly and say so loudly instead."""
    b = _broker(_pos(), orders=[])
    await b.reconcile(["BTC-USDT"])
    p = b.portfolio.positions["BTC-USDT"]
    assert p.stop_price == 0.0 and p.init_risk == 0.0


async def test_unreadable_orders_still_adopt_the_position():
    """Losing the levels is bad; losing track of the POSITION is worse."""
    b = _broker(_pos(), fail_orders=True)
    await b.reconcile(["BTC-USDT"])
    assert "BTC-USDT" in b.portfolio.positions
