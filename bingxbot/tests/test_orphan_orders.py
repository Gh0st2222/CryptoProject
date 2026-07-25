"""An abandoned entry must not leave a live order on the exchange.

When a resting limit does not fill, the engine reports "unfilled — entry
abandoned" and forgets it. If the order is still on the book it can fill HOURS
later on a signal that has long since gone stale, and the first the bot hears of
it is reconcile adopting a position nobody decided to take — while it silently
holds margin in the meantime.

The same applies to task cancellation: drop_symbol() cancels a pending entry
task, and cancelling an asyncio task does nothing at all to an order that is
already resting on the exchange.
"""
import asyncio

import pytest

from bingxbot.config import BotConfig
from bingxbot.engine.portfolio import Portfolio
from bingxbot.exchange.models import ContractSpec


class _Rest:
    """Records cancels; the order never fills unless told otherwise."""

    def __init__(self, status="NEW", exec_qty=0.0, raise_on_get=False):
        self.status, self.exec_qty = status, exec_qty
        self.raise_on_get = raise_on_get
        self.cancels: list[tuple[str, str]] = []

    async def get_order(self, symbol, order_id):
        if self.raise_on_get:
            from bingxbot.exchange.errors import BingXError
            raise BingXError("gateway hiccup")
        return {"status": self.status, "executedQty": self.exec_qty,
                "avgPrice": 100.0, "commission": 0.01}

    async def cancel_order(self, symbol, order_id):
        self.cancels.append((symbol, order_id))
        return {}


def _broker(rest):
    from bingxbot.engine.brokers import LiveBroker
    return LiveBroker(rest, Portfolio(1000.0, mode="live"),
                      {"BTC-USDT": ContractSpec("BTC-USDT")}, BotConfig())


async def test_an_unfilled_limit_is_pulled_off_the_book():
    """The core case: the window elapses with nothing filled."""
    rest = _Rest(status="NEW", exec_qty=0.0)
    b = _broker(rest)
    ap, qty, fee = await b._await_limit_fill("BTC-USDT", "o1", window_s=3.0)
    assert (ap, qty, fee) == (0.0, 0.0, 0.0), "reported as unfilled"
    assert rest.cancels == [("BTC-USDT", "o1")], "and actually cancelled"


async def test_a_filled_limit_is_not_cancelled():
    rest = _Rest(status="FILLED", exec_qty=0.5)
    b = _broker(rest)
    ap, qty, _ = await b._await_limit_fill("BTC-USDT", "o1", window_s=3.0)
    assert qty == 0.5 and ap == 100.0
    assert rest.cancels == [], "nothing to pull — it filled"


async def test_an_already_dead_order_is_not_cancelled_again():
    rest = _Rest(status="CANCELED", exec_qty=0.0)
    b = _broker(rest)
    await b._await_limit_fill("BTC-USDT", "o1", window_s=3.0)
    assert rest.cancels == [], "the exchange already removed it"


async def test_an_unreadable_order_is_cancelled_rather_than_assumed_gone():
    """If we cannot tell what happened, the safe assumption is that it is still
    resting — leaving a live order behind is the costlier mistake."""
    rest = _Rest(raise_on_get=True)
    b = _broker(rest)
    await b._await_limit_fill("BTC-USDT", "o1", window_s=3.0)
    assert rest.cancels == [("BTC-USDT", "o1")]


async def test_cancelling_the_task_still_pulls_the_order():
    """drop_symbol() cancels the pending entry TASK. That does nothing to an
    order already on the book unless this path pulls it."""
    rest = _Rest(status="NEW", exec_qty=0.0)
    b = _broker(rest)
    task = asyncio.get_running_loop().create_task(
        b._await_limit_fill("BTC-USDT", "o1", window_s=600.0))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.05)      # let the shielded cancel land
    assert rest.cancels == [("BTC-USDT", "o1")], "the exchange order was pulled too"


async def test_a_partial_fill_keeps_what_filled_and_pulls_the_rest():
    rest = _Rest(status="NEW", exec_qty=0.3)
    b = _broker(rest)
    ap, qty, _ = await b._await_limit_fill("BTC-USDT", "o1", window_s=3.0)
    assert qty == 0.3, "the part that filled is real and must be kept"
    assert rest.cancels == [("BTC-USDT", "o1")], "the remainder is pulled"
    assert len(rest.cancels) == 1, "and only once"
