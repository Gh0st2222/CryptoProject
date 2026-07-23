"""Owner round three: execution honesty, funding parity, risk-memory
continuity, kill-switch persistence, and adoption seat hysteresis."""
import asyncio
import types

import pytest

from bingxbot.config import BotConfig
from bingxbot.exchange.models import LONG, SHORT, Position
from bingxbot.risk.manager import RiskManager


def _paper_broker(pf, states):
    from bingxbot.engine.brokers import PaperBroker
    return PaperBroker(pf, states, {}, taker_fee=5e-4, slippage_bps=0.0)


def _sized(limit: float):
    from bingxbot.risk.manager import SizedOrder
    return SizedOrder(qty=1.0, notional=limit, leverage=2, stop_price=limit * 0.98,
                      take_profit=0.0, risk_amount=limit * 0.02,
                      entry_limit=limit, entry_wait_s=6.0)


async def test_paper_resting_fill_requires_trade_through():
    """A resting paper limit must NOT fill on a mere touch — on the real book a
    kiss of the level fills the queue ahead of us and leaves. Only a tape print
    THROUGH the limit (by the backtest's FILL_THROUGH_BPS margin) fills, and it
    fills AT the limit price — a limit can never fill worse than where it rested."""
    from bingxbot.engine.backtest import FILL_THROUGH_BPS
    from bingxbot.engine.portfolio import Portfolio
    pf = Portfolio(1000.0, mode="paper")
    st = types.SimpleNamespace(last_price=101.0, book=None)
    br = _paper_broker(pf, {"BTC-USDT": st})
    task = asyncio.get_running_loop().create_task(
        br.open_position("BTC-USDT", LONG, _sized(100.0), "test", bar_ts=0))
    st.last_price = 100.0            # exact touch — queue-position honesty says no
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(task), timeout=1.4)
    st.last_price = 100.0 * (1 - FILL_THROUGH_BPS / 10_000.0) - 1e-9   # traded through
    res = await asyncio.wait_for(task, timeout=5.0)
    assert res.ok, res.error
    assert res.filled_price == pytest.approx(100.0), \
        "resting fill must be AT the limit — no adverse markup on an honest wait"
    assert pf.positions["BTC-USDT"].entry_price == pytest.approx(100.0)


def test_manual_kill_and_cooldown_survive_midnight():
    """The UTC day roll resets DAILY counters only. A manual stop must never
    un-stop itself at midnight, an active loss-streak cooldown keeps braking,
    and only the daily-loss kill clears with its new day."""
    cfg = BotConfig()
    t = [1_700_000_000.0]
    rm = RiskManager(cfg.risk, clock=lambda: t[0])
    ok, _ = rm.can_enter(1000.0, 0, 1.0)
    assert ok
    rm.manual_kill("manual stop")
    t[0] += 86_400
    ok, why = rm.can_enter(1000.0, 0, 1.0)
    assert not ok and "manual stop" in why, "manual kill must survive the day roll"

    rm2 = RiskManager(cfg.risk, clock=lambda: t[0])
    rm2.can_enter(1000.0, 0, 1.0)                       # pin today's day_key
    rm2.state.cooldown_until = t[0] + 86_400 + 3_600    # brake extends past midnight
    t[0] += 86_400                                      # cross the day roll, still cooling
    ok, why = rm2.can_enter(1000.0, 0, 1.0)
    assert not ok and "cooldown" in why, "streak cooldown must survive the day roll"

    rm3 = RiskManager(cfg.risk, clock=lambda: t[0])
    rm3.can_enter(1000.0, 0, 1.0)
    rm3.state.killed = True
    rm3.state.kill_reason = "daily loss limit hit (5.0%)"
    t[0] += 86_400
    ok, _ = rm3.can_enter(1000.0, 0, 1.0)
    assert ok, "the DAILY loss kill clears with its new day — that's its semantics"


def test_max_drawdown_survives_curve_truncation():
    """The equity deque holds ~8h; a dip that scrolls out of it must not scroll
    out of the max-drawdown statistic."""
    from bingxbot.engine.portfolio import Portfolio
    pf = Portfolio(1000.0, mode="paper")
    pf.record_equity(10_000)                 # 1000 -> peak
    pf.cash = 800.0
    pf.record_equity(20_000)                 # 20% dip recorded
    pf.cash = 1000.0
    pf.equity_curve.clear()                  # simulate the dip aging out
    pf.record_equity(30_000)
    assert pf.stats()["max_drawdown"] >= 0.2 - 1e-9


def test_health_governor_and_drawdown_roundtrip(tmp_path):
    """A restart during a cold streak must resume at the throttled size, not
    silently restore full risk; the running peak/max-dd ride along too."""
    from bingxbot.engine.persist import (load_paper_state, restore_into,
                                         save_paper_state)
    from bingxbot.engine.portfolio import Portfolio
    cfg = BotConfig()
    pf = Portfolio(1000.0, mode="paper")
    rm = RiskManager(cfg.risk)
    for _ in range(12):
        rm.health.on_trade(-1.0, 950.0)      # cold streak -> scalar well below 1
    pf.peak_equity, pf.max_dd = 1050.0, 0.17
    assert rm.health.scalar < 0.9
    p = tmp_path / "paper_state.json"
    save_paper_state(pf, rm.state, path=p, health=rm.health.state_dict())
    snap = load_paper_state(1000.0, path=p)
    assert snap is not None
    pf2, rm2 = Portfolio(1000.0, mode="paper"), RiskManager(cfg.risk)
    restore_into(pf2, rm2, snap)
    assert rm2.health.scalar == pytest.approx(rm.health.scalar, rel=1e-6)
    assert list(rm2.health.r_hist) == list(rm.health.r_hist)
    assert pf2.peak_equity == pytest.approx(1050.0)
    assert pf2.max_dd == pytest.approx(0.17)


def _engine(tmp_path, symbols=("BTC-USDT",)):
    from bingxbot.data.feed import SyntheticFeed
    from bingxbot.engine.journal import TradeJournal
    from bingxbot.engine.portfolio import Portfolio
    from bingxbot.engine.trader import TraderEngine
    cfg = BotConfig()
    cfg.symbols = list(symbols)
    feed = SyntheticFeed(cfg.symbols, "15m", warmup_bars=10, seed=1)
    pf = Portfolio(1000.0, mode="paper")
    j = TradeJournal(tmp_path / "j.jsonl")
    eng = TraderEngine(cfg, feed, _paper_broker(pf, feed.states), pf,
                       RiskManager(cfg.risk), {}, journal=j)
    return eng, pf


def test_paper_funding_settlement(tmp_path):
    """At each 8h boundary paper transfers REAL signed funding on open
    positions — longs pay a positive rate, shorts receive it — exactly like
    the exchange and the backtest. Carry-desk positions are skipped (that desk
    books its own settlements)."""
    eng, pf = _engine(tmp_path)
    pf.positions["BTC-USDT"] = Position(symbol="BTC-USDT", side=LONG, qty=2.0,
                                        entry_price=100.0, opened_ts=0,
                                        entry_reason="edge +0.4 P60%")
    eng._funding_rates = {"BTC-USDT": 0.0001}
    cash0 = pf.cash
    eng._settle_paper_funding({"BTC-USDT": 100.0})
    assert pf.cash == pytest.approx(cash0 - 2.0 * 100.0 * 0.0001)
    assert pf.funding_paid == pytest.approx(0.02)

    pf.positions["BTC-USDT"].side = SHORT   # receiving side of the same rate
    cash1 = pf.cash
    eng._settle_paper_funding({"BTC-USDT": 100.0})
    assert pf.cash == pytest.approx(cash1 + 0.02)

    pf.positions["BTC-USDT"].entry_reason = "carry +35% APR"
    cash2 = pf.cash
    eng._settle_paper_funding({"BTC-USDT": 100.0})
    assert pf.cash == cash2, "carry positions settle through the carry desk only"

    pf.mode = "live"
    pf.positions["BTC-USDT"].entry_reason = "edge"
    eng._settle_paper_funding({"BTC-USDT": 100.0})
    assert pf.cash == cash2, "live funding is the exchange's ledger, not ours"


def test_adoption_seat_hysteresis():
    """An adopted incumbent is never evicted for being outranked — only for
    failing the KEEP standard twice (or holding a position: never). New
    symbols fill free seats only, over the full adopt bar."""
    from bingxbot.engine.scanner import plan_adoption
    def row(sym, er, qv=30e6, kind="trend", d=1):
        return {"symbol": sym, "er_4h": er, "dir_4h": d, "quote_volume": qv, "kind": kind}
    no_pos = lambda s: False   # noqa: E731

    # outranked but healthy: SEI (er .35, below the .40 adopt bar but above the
    # .30 keep bar) holds its seat against a hotter SOL — no churn.
    miss = {}
    drops, adds = plan_adoption([row("SOL-USDT", 0.9), row("SEI-USDT", 0.35)],
                                {"SEI-USDT"}, {"BTC-USDT"}, no_pos, 1, miss)
    assert drops == [] and adds == [] and miss == {}

    # degradation: below the keep bar it takes TWO consecutive scans to drop;
    # the freed seat then goes to the best full-bar candidate in the same plan.
    miss = {}
    stale = [row("SOL-USDT", 0.9), row("SEI-USDT", 0.2)]
    drops, adds = plan_adoption(stale, {"SEI-USDT"}, set(), no_pos, 1, miss)
    assert drops == [] and miss == {"SEI-USDT": 1}
    drops, adds = plan_adoption(stale, {"SEI-USDT"}, set(), no_pos, 1, miss)
    assert drops == ["SEI-USDT"] and adds == ["SOL-USDT"]

    # an open position pins the seat no matter what the radar says
    miss = {}
    drops, _ = plan_adoption([row("SEI-USDT", 0.0, d=0)], {"SEI-USDT"}, set(),
                             lambda s: True, 1, miss)
    assert drops == [] and miss == {}

    # cap shrank: shed the WEAKEST surplus seat, keep the stronger incumbent
    miss = {}
    drops, adds = plan_adoption([row("SEI-USDT", 0.5), row("TON-USDT", 0.8)],
                                {"SEI-USDT", "TON-USDT"}, set(), no_pos, 1, miss)
    assert drops == ["SEI-USDT"] and adds == []

    # the user's own symbols and position-holding tokens never enter via adds
    drops, adds = plan_adoption([row("BTC-USDT", 0.9), row("SOL-USDT", 0.8)],
                                set(), {"BTC-USDT"}, lambda s: s == "SOL-USDT", 2, {})
    assert adds == []
