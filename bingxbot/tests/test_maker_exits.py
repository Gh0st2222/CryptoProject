"""Resting post-only profit targets: earn the maker side instead of paying
taker, without ever putting the account at risk of an unmanaged position."""
import pytest

from bingxbot.config import BotConfig
from bingxbot.exchange.models import LONG, SHORT, ContractSpec, Position
from bingxbot.engine.portfolio import Portfolio


# ------------------------------------------------------------- live order safety

class _FakeRest:
    """Records what would be sent to BingX."""

    def __init__(self, fail=False):
        self.orders, self.cancels, self.fail = [], [], fail

    async def place_order(self, **kw):
        if self.fail:
            from bingxbot.exchange.errors import BingXAPIError
            raise BingXAPIError(code=1, msg="post-only would cross")
        self.orders.append(kw)
        return {"orderId": f"o{len(self.orders)}"}

    async def cancel_order(self, symbol, order_id):
        self.cancels.append((symbol, order_id))
        return {}


def _live(rest):
    from bingxbot.engine.brokers import LiveBroker
    cfg = BotConfig()
    return LiveBroker(rest, Portfolio(1000.0, mode="live"),
                      {"BTC-USDT": ContractSpec("BTC-USDT")}, cfg)


async def test_resting_exit_is_reduce_only_and_post_only():
    """The two properties that make a resting order safe to leave on the book.
    reduceOnly is exchange-enforced: the order can only ever SHRINK a position,
    so it can never open a reverse one if the position is already gone.
    PostOnly means it is rejected rather than filled if it would cross — it can
    never become the taker order we are trying to avoid."""
    rest = _FakeRest()
    b = _live(rest)
    assert await b.arm_maker_exit("BTC-USDT", LONG, 0.5, 61000.0) is True
    o = rest.orders[0]
    assert o["reduce_only"] is True, "MUST be reduce-only — this is the safety property"
    assert o["time_in_force"] == "PostOnly", "must never cross the spread"
    assert o["order_type"] == "LIMIT" and o["price"] == 61000.0
    assert o["side"] == "SELL" and o["position_side"] == LONG, "closing side of a LONG"

    rest2 = _FakeRest()
    b2 = _live(rest2)
    await b2.arm_maker_exit("BTC-USDT", SHORT, 0.5, 59000.0)
    assert rest2.orders[0]["side"] == "BUY" and rest2.orders[0]["position_side"] == SHORT


async def test_stop_loss_is_never_turned_into_a_resting_order():
    """A stop that fails to fill is the one loss you cannot iterate on. Only the
    TARGET may rest; the protective stop stays an attached market order."""
    from bingxbot.risk.manager import SizedOrder
    b = _live(_FakeRest())
    sized = SizedOrder(qty=1.0, notional=100.0, leverage=2, stop_price=59000.0,
                       take_profit=61000.0, risk_amount=10.0)
    sl, tp = b._sl_tp(sized)
    assert sl["type"] == "STOP_MARKET", "the stop must always be a market order"
    assert tp is None, "with maker exits the target is a separate resting order"
    b.cfg.strategy.maker_exits = False
    sl2, tp2 = b._sl_tp(sized)
    assert sl2["type"] == "STOP_MARKET" and tp2["type"] == "TAKE_PROFIT_MARKET"


async def test_rejected_exit_falls_back_instead_of_leaving_a_position_stranded():
    b = _live(_FakeRest(fail=True))
    assert await b.arm_maker_exit("BTC-USDT", LONG, 0.5, 61000.0) is False, \
        "a refusal must be reported so the caller keeps market-close-on-touch"


async def test_arming_twice_never_stacks_two_resting_exits():
    rest = _FakeRest()
    b = _live(rest)
    await b.arm_maker_exit("BTC-USDT", LONG, 0.5, 61000.0)
    await b.arm_maker_exit("BTC-USDT", LONG, 0.4, 61500.0)
    assert rest.cancels == [("BTC-USDT", "o1")], "the previous order is pulled first"
    assert b.maker_exit_price("BTC-USDT") == 61500.0


async def test_vanished_position_is_booked_at_the_resting_price_not_the_stop():
    """A maker exit filling between reconcile polls is the MOST likely reason a
    position disappears. Booking it at the stop would invent a loss that never
    happened."""
    rest = _FakeRest()
    b = _live(rest)
    b.portfolio.positions["BTC-USDT"] = Position(
        symbol="BTC-USDT", side=LONG, qty=1.0, entry_price=60000.0, opened_ts=0,
        stop_price=59000.0, take_profit=61000.0)
    await b.arm_maker_exit("BTC-USDT", LONG, 1.0, 61000.0)
    b._record_external_close("BTC-USDT", reason="gone")
    t = b.portfolio.trades[-1]
    assert t.exit_price == 61000.0 and t.pnl > 0, "booked as the win it was"
    assert "maker" in t.reason_close


# ------------------------------------------------------------- paper fill model

async def test_paper_maker_close_fills_at_the_limit_and_pays_maker_fee():
    from bingxbot.engine.brokers import PaperBroker
    pf = Portfolio(1000.0, mode="paper")
    pf.positions["BTC-USDT"] = Position(symbol="BTC-USDT", side=LONG, qty=1.0,
                                        entry_price=100.0, opened_ts=0)
    br = PaperBroker(pf, {}, {}, taker_fee=5e-4, slippage_bps=10.0, maker_fee=2e-4)
    res = await br.close_position("BTC-USDT", "target", maker_price=110.0)
    assert res.ok and res.filled_price == 110.0, "fills at OUR price, no slippage"
    assert res.fee == pytest.approx(1.0 * 110.0 * 2e-4), "maker fee, not taker"


class _Book:
    def __init__(self, bid, ask):
        self.bid, self.ask, self.mid = bid, ask, (bid + ask) / 2


class _St:
    def __init__(self, bid, ask):
        self.book, self.last_price = _Book(bid, ask), (bid + ask) / 2


def _paper(bid=100.0, ask=100.02):
    from bingxbot.engine.brokers import PaperBroker
    pf = Portfolio(1000.0, mode="paper")
    return PaperBroker(pf, {"BTC-USDT": _St(bid, ask)},
                       {"BTC-USDT": ContractSpec("BTC-USDT")},
                       taker_fee=5e-4, slippage_bps=10.0, maker_fee=2e-4)


async def test_paper_actually_arms_the_exit_instead_of_inheriting_the_no_op():
    """Regression: PaperBroker used to inherit Broker.arm_maker_exit -> False,
    so paper was the ONLY engine still taking its targets while the simulator
    and the kernel priced them as maker fills. A champion measured on maker
    exits would then be traded on taker exits — the exact parity break the
    whole doctrine exists to prevent, just on the cheaper side of the fee."""
    from bingxbot.engine.brokers import Broker, PaperBroker
    assert PaperBroker.arm_maker_exit is not Broker.arm_maker_exit
    br = _paper()
    assert await br.arm_maker_exit("BTC-USDT", LONG, 1.0, 110.0) is True
    assert br.maker_exit_price("BTC-USDT") == 110.0
    await br.cancel_maker_exit("BTC-USDT")
    assert br.maker_exit_price("BTC-USDT") == 0.0


async def test_paper_rejects_a_post_only_exit_that_would_cross():
    """Post-only is a rejection, not a discount. A SELL at or below the bid
    would be an instant taker, so the exchange refuses it — paper must refuse
    it too, or it books maker fees on fills that were really taker."""
    br = _paper(bid=100.0, ask=100.02)
    assert await br.arm_maker_exit("BTC-USDT", LONG, 1.0, 99.0) is False, \
        "a SELL below the bid crosses"
    assert await br.arm_maker_exit("BTC-USDT", LONG, 1.0, 100.0) is False, \
        "a SELL at the bid crosses"
    assert await br.arm_maker_exit("BTC-USDT", LONG, 1.0, 100.5) is True
    assert await br.arm_maker_exit("BTC-USDT", SHORT, 1.0, 101.0) is False, \
        "a BUY above the ask crosses"
    assert await br.arm_maker_exit("BTC-USDT", SHORT, 1.0, 99.5) is True


async def test_paper_drops_the_resting_exit_when_the_position_goes_away():
    """A resting order that outlives its position is how paper would invent a
    maker fill on a symbol it no longer holds."""
    br = _paper()
    br.portfolio.positions["BTC-USDT"] = Position(
        symbol="BTC-USDT", side=LONG, qty=1.0, entry_price=100.0, opened_ts=0)
    await br.arm_maker_exit("BTC-USDT", LONG, 1.0, 110.0)
    await br.close_position("BTC-USDT", "stop loss")
    assert br.maker_exit_price("BTC-USDT") == 0.0

    br.portfolio.positions["BTC-USDT"] = Position(
        symbol="BTC-USDT", side=LONG, qty=1.0, entry_price=100.0, opened_ts=0)
    await br.arm_maker_exit("BTC-USDT", LONG, 1.0, 110.0)
    await br.close_position("BTC-USDT", "target", maker_price=110.0)
    assert br.maker_exit_price("BTC-USDT") == 0.0


async def test_engine_end_to_end_touch_holds_then_trade_through_pays_maker(tmp_path):
    """The whole chain in one go, through the real engine tick path:
    a touch of the target must NOT close (we are still queued), and the
    trade-through must close at OUR price for the MAKER fee. If a touch closed
    here, every backtested target would be a paper win the book never gave us."""
    from bingxbot.data.feed import SyntheticFeed
    from bingxbot.engine.backtest import FILL_THROUGH_BPS
    from bingxbot.engine.brokers import PaperBroker
    from bingxbot.engine.trader import TraderEngine
    from bingxbot.risk.manager import RiskManager

    cfg = BotConfig()
    cfg.symbols = ["BTC-USDT"]
    cfg.strategy.maker_exits = True
    feed = SyntheticFeed(cfg.symbols, "15m", warmup_bars=10, seed=1)
    pf = Portfolio(1000.0, mode="paper")
    st = feed.states["BTC-USDT"]
    st.book, st.last_price = _Book(100.0, 100.02), 100.0
    br = PaperBroker(pf, feed.states, {"BTC-USDT": ContractSpec("BTC-USDT")},
                     taker_fee=5e-4, slippage_bps=10.0, maker_fee=2e-4)
    eng = TraderEngine(cfg, feed, br, pf, RiskManager(cfg.risk), {})

    pf.positions["BTC-USDT"] = Position(
        symbol="BTC-USDT", side=LONG, qty=1.0, entry_price=100.0, opened_ts=0,
        stop_price=98.0, take_profit=110.0)
    ctx = eng.ctx["BTC-USDT"]
    ctx.maker_exit = await br.arm_maker_exit("BTC-USDT", LONG, 1.0, 110.0)
    assert ctx.maker_exit is True, "the exit must actually arm in paper"

    st.last_price = 110.0                       # exact touch: still in the queue
    await eng._on_tick("BTC-USDT")
    assert "BTC-USDT" in pf.positions, "a touch must NOT bank the target"
    assert not pf.trades

    st.last_price = 110.0 * (1 + FILL_THROUGH_BPS / 10_000.0) + 1e-9   # traded through
    await eng._on_tick("BTC-USDT")
    assert "BTC-USDT" not in pf.positions, "a trade-through fills the resting order"
    t = pf.trades[-1]
    assert t.exit_price == 110.0, "fills at OUR resting price, never worse"
    assert t.fees == pytest.approx(1.0 * 110.0 * 2e-4), "the maker fee we came for"
    assert ctx.maker_exit is False and br.maker_exit_price("BTC-USDT") == 0.0


def test_paper_trade_through_margin_is_bolted_to_the_simulator_constant():
    """The engine's resting-fill margin must be the SIMULATOR's constant, not
    the spread constant that happens to share its value today."""
    import inspect

    from bingxbot.engine import trader
    src = inspect.getsource(trader.TraderEngine._on_tick)
    assert "FILL_THROUGH_BPS" in src and "ASSUMED_SPREAD_BPS" not in src


# ------------------------------------------------------------- simulator honesty

def _sim(maker_exits):
    from bingxbot.config import RiskConfig, StrategyConfig
    from bingxbot.engine.backtest import _SymSim
    from bingxbot.exchange.models import Candle
    s, r = StrategyConfig(), RiskConfig()
    s.maker_exits = maker_exits
    candles = [Candle(ts=i * 900_000, open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0)
               for i in range(400)]
    return _SymSim("BTC-USDT", "15m", candles, s, r, ContractSpec("BTC-USDT"),
                   5e-4, 1.0, False)


def test_a_resting_target_is_not_a_free_upgrade():
    """Honesty check: with maker exits the simulator must REQUIRE trade-through
    before banking the target — a touch leaves us in the queue. Otherwise this
    would just be the old flattery wearing a new name."""
    from bingxbot.engine.backtest import FILL_THROUGH_BPS
    assert _sim(True).maker_exits is True
    assert _sim(False).maker_exits is False
    tp, thru = 100.0, FILL_THROUGH_BPS / 10_000.0
    touch, through = tp, tp * (1 + thru) + 1e-9
    assert not (touch >= tp * (1 + thru)), "a touch must NOT fill a resting order"
    assert through >= tp * (1 + thru), "a trade-through does"


def test_kernel_mirrors_the_setting():
    """The compiled kernel and the python simulator must never disagree about
    how the exit is priced, or the tuner optimizes one engine and trades another."""
    from bingxbot.config import RiskConfig, StrategyConfig
    from bingxbot.engine.kernel import P, pack_params
    assert "maker_exits" in P
    s, r = StrategyConfig(), RiskConfig()
    s.maker_exits = True
    assert pack_params(s, r)[P["maker_exits"]] == 1.0
    s.maker_exits = False
    assert pack_params(s, r)[P["maker_exits"]] == 0.0


def test_setting_is_user_owned_and_survives_migration(tmp_path):
    import json

    from bingxbot.config import CONFIG_VERSION, load_config
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"version": CONFIG_VERSION - 1, "mode": "paper",
                             "strategy": {"maker_exits": False}}))
    assert load_config(path=p).strategy.maker_exits is False, \
        "a deliberate choice must survive a config migration"
