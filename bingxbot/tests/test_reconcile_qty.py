"""Reconcile must agree with the exchange about SIZE, not just existence.

Presence was always checked; size never was. A TradeRecord books
`qty * (exit - entry)`, so a stale qty does not merely misstate exposure — it
manufactures P&L that never happened, and that number feeds the journal, the
live-evidence demotion rule, probation stats and the report. The system learns
from those numbers, so this is a correctness bug with compounding consequences.
"""
import pytest

from bingxbot.config import BotConfig
from bingxbot.engine.portfolio import Portfolio
from bingxbot.exchange.models import LONG, ContractSpec, Position


class _Rest:
    def __init__(self, rows):
        self.rows = rows

    async def positions(self):
        return self.rows

    async def balance(self):
        return {"equity": 1000.0, "balance": 1000.0}

    async def place_order(self, **kw):
        return {"orderId": "o1"}

    async def cancel_order(self, symbol, order_id):
        return {}

    async def open_orders(self, symbol=None):
        return []


def _broker(rows, qty_precision=3, min_qty=0.001):
    from bingxbot.engine.brokers import LiveBroker
    spec = ContractSpec("BTC-USDT")
    spec.qty_precision, spec.min_qty = qty_precision, min_qty
    return LiveBroker(_Rest(rows), Portfolio(1000.0, mode="live"),
                      {"BTC-USDT": spec}, BotConfig())


def _row(amt, mark=110.0):
    return {"symbol": "BTC-USDT", "positionAmt": amt, "positionSide": LONG,
            "avgPrice": 100.0, "markPrice": mark}


def _open(b, qty=1.0):
    b.portfolio.positions["BTC-USDT"] = Position(
        symbol="BTC-USDT", side=LONG, qty=qty, entry_price=100.0, opened_ts=0,
        stop_price=95.0, take_profit=110.0, init_risk=5.0)
    return b.portfolio.positions["BTC-USDT"]


async def test_a_partially_filled_exit_is_booked_not_ignored():
    """The ordinary case: a resting reduce-only limit is a limit order, so price
    trading through it fills PART of the size and leaves. The symbol is still on
    both sides, so the presence check sees nothing wrong."""
    b = _broker([_row(0.4)])
    _open(b, qty=1.0)
    await b.arm_maker_exit("BTC-USDT", LONG, 1.0, 110.0)
    await b.reconcile(["BTC-USDT"])
    pos = b.portfolio.positions["BTC-USDT"]
    assert pos.qty == pytest.approx(0.4), "local size now matches the exchange"
    t = b.portfolio.trades[-1]
    assert t.qty == pytest.approx(0.6), "the filled 60% is booked as a real trade"
    assert t.exit_price == 110.0, "at the resting price we actually know"
    assert t.pnl > 0 and "maker" in t.reason_close


async def test_stale_qty_would_otherwise_invent_pnl():
    """The reason this matters, stated as arithmetic: booking a 1.0 close when
    only 0.4 is held overstates the trade by 150%."""
    b = _broker([_row(0.4)])
    _open(b, qty=1.0)
    await b.reconcile(["BTC-USDT"])
    pos = b.portfolio.positions["BTC-USDT"]
    banked = sum(t.pnl for t in b.portfolio.trades)
    # what the old code would have booked on the eventual close vs the truth
    fabricated = (110.0 - 100.0) * 1.0
    honest = banked + (110.0 - 100.0) * pos.qty
    assert honest < fabricated, "the corrected book is smaller than the fabricated one"
    assert pos.qty == pytest.approx(0.4)


async def test_growth_adopts_exchange_truth_without_inventing_a_trade():
    b = _broker([_row(1.5)])
    _open(b, qty=1.0)
    await b.reconcile(["BTC-USDT"])
    assert b.portfolio.positions["BTC-USDT"].qty == pytest.approx(1.5)
    assert not b.portfolio.trades, "nothing was closed, so nothing may be booked"


async def test_dust_and_rounding_never_trigger_a_phantom_partial():
    """Exchanges round to the symbol's step. If that tripped a partial close the
    log would fill with fake trades on every poll."""
    b = _broker([_row(0.9995)])
    _open(b, qty=1.0)
    await b.reconcile(["BTC-USDT"])
    assert b.portfolio.positions["BTC-USDT"].qty == pytest.approx(1.0), "untouched"
    assert not b.portfolio.trades


async def test_a_reconcile_partial_does_not_cancel_the_strategys_scale_out():
    """scale_out() marks the position as having taken its scale-out. A partial
    FILL is an execution artifact, not the strategy's decision — if it set that
    flag, the scale-out the strategy still intends would silently never fire."""
    b = _broker([_row(0.5)])
    pos = _open(b, qty=1.0)
    assert pos.scaled_out is False
    await b.reconcile(["BTC-USDT"])
    assert b.portfolio.positions["BTC-USDT"].scaled_out is False


async def test_presence_checks_still_work():
    """The behaviour that already existed must survive the change."""
    b = _broker([])
    _open(b, qty=1.0)
    await b.reconcile(["BTC-USDT"])
    assert "BTC-USDT" not in b.portfolio.positions, "gone on the exchange = closed"

    b2 = _broker([_row(2.0)])
    await b2.reconcile(["BTC-USDT"])
    assert b2.portfolio.positions["BTC-USDT"].qty == pytest.approx(2.0), "adopted"
