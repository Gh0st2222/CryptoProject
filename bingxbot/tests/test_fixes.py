"""Tests for the deep-review fixes: causal (no-lookahead) MTF ladder, the
one-position-per-token guarantee, exactly-once risk settlement, tunable-bounds
clamping, carry/token exclusivity, and pullback (resting) entries."""
import asyncio

import numpy as np
import pytest

from bingxbot.config import RiskConfig, StrategyConfig
from bingxbot.data.history import synthetic_candles
from bingxbot.engine.backtest import apply_tunables_inplace, candles_to_arrays
from bingxbot.engine.portfolio import Portfolio
from bingxbot.exchange.models import LONG, SHORT, BookTop, ContractSpec, Position, TradeRecord
from bingxbot.strategy import indicators as ta
from bingxbot.strategy.features import FeatureFrame


# ------------------------------------------------- MTF ladder causality

def _frames(symbol="BTC-USDT", interval="15m", n=1200, seed=9):
    candles = synthetic_candles(symbol, interval, n, seed=seed)
    return candles


def test_resample_ohlc_epoch_anchored():
    candles = _frames(n=400)
    a = candles_to_arrays(candles)
    bucket_ms = 60 * 60_000  # 1h buckets over a 15m base
    ho, hh, hl, hc, hv, bidx = ta.resample_ohlc(
        a["ts"], a["open"], a["high"], a["low"], a["close"], a["volume"], bucket_ms)
    assert len(bidx) == len(a["ts"])
    assert bidx[0] == 0 and bidx[-1] == len(ho) - 1
    # every bar maps into the epoch bucket its timestamp says
    ids = a["ts"] // bucket_ms
    assert np.all(np.diff(bidx) >= 0)
    # bars sharing an epoch id share a bucket ordinal
    for g in range(len(ho)):
        sel = bidx == g
        assert len(set(ids[sel])) == 1
        assert hh[g] == a["high"][sel].max()
        assert hl[g] == a["low"][sel].min()
        assert hc[g] == a["close"][sel][-1]


def test_mtf_ladder_has_no_lookahead():
    """The decisive test: the higher-timeframe features at bar i must be
    identical whether or not the future exists. The old index-bucketed
    broadcast read the close of a bucket that included FUTURE base bars, so
    truncating history changed the past — textbook lookahead."""
    candles = _frames(interval="15m", n=1000, seed=17)
    full = FeatureFrame(candles_to_arrays(candles), interval="15m")
    for cut in (700, 701, 702, 703):     # cover every offset within a 1h bucket
        part = FeatureFrame(candles_to_arrays(candles[:cut]), interval="15m")
        i = cut - 1
        for key in ("mtf_bias", "mtf_align", "tf_1h_dir", "tf_1h_rsi", "tf_1h_adx",
                    "ema21_slope", "ema55_slope", "macd_rising", "htf_med_dir"):
            if key not in full.f:
                continue
            a, b = float(full.f[key][i]), float(part.f[key][i])
            if np.isnan(a) and np.isnan(b):
                continue
            assert a == pytest.approx(b, abs=1e-12), (
                f"{key} at bar {i} changed when the future was removed "
                f"(full={a}, truncated={b}) — lookahead")


def test_mtf_ladder_stable_under_window_shift():
    """Epoch-anchored buckets: sliding the window by one bar (what the live
    tail does every bar close) must not re-partition the higher-TF buckets."""
    candles = _frames(interval="15m", n=1000, seed=23)
    w1 = FeatureFrame(candles_to_arrays(candles[100:900]), interval="15m")
    w2 = FeatureFrame(candles_to_arrays(candles[101:900]), interval="15m")
    # same final bar, same data tail -> same higher-TF read at the last row
    for key in ("tf_1h_dir", "mtf_bias"):
        if key in w1.f and key in w2.f:
            assert float(w1.f[key][-1]) == pytest.approx(float(w2.f[key][-1]), abs=1e-9)


# ------------------------------------------- one position per token, always

def test_portfolio_refuses_duplicate_open():
    pf = Portfolio(10_000.0, mode="paper")
    p1 = Position(symbol="BTC-USDT", side=LONG, qty=1.0, entry_price=100.0, opened_ts=1)
    p2 = Position(symbol="BTC-USDT", side=SHORT, qty=9.0, entry_price=100.0, opened_ts=2)
    assert pf.open_position(p1, entry_fee=0.1) is True
    cash_after_first = pf.cash
    assert pf.open_position(p2, entry_fee=0.5) is False, "second open on a held token must be refused"
    assert pf.positions["BTC-USDT"] is p1, "the original position must survive"
    assert pf.cash == cash_after_first, "a refused open must not touch cash"


@pytest.mark.asyncio
async def test_paper_broker_refuses_double_open():
    from bingxbot.engine.brokers import PaperBroker
    from bingxbot.risk.manager import SizedOrder

    class _St:
        book = BookTop(ts=0, bid=100.0, bid_qty=1, ask=100.1, ask_qty=1)
        last_price = 100.05

        class candles:
            last_close = 100.05

    pf = Portfolio(10_000.0, mode="paper")
    broker = PaperBroker(pf, {"BTC-USDT": _St()}, {"BTC-USDT": ContractSpec("BTC-USDT")},
                         taker_fee=0.0005, slippage_bps=0.0, entry_mode="taker")
    sized = SizedOrder(qty=1.0, notional=100.1, leverage=1,
                       stop_price=99.0, take_profit=0.0, risk_amount=1.1)
    r1 = await broker.open_position("BTC-USDT", LONG, sized, "t", 0)
    r2 = await broker.open_position("BTC-USDT", SHORT, sized, "t", 0)
    assert r1.ok and not r2.ok
    assert "already open" in r2.error
    assert pf.positions["BTC-USDT"].side == LONG


def test_carry_desk_never_picks_engine_tokens():
    """The carry desk must skip tokens a signal brain is watching, not just
    tokens already held — 'never looking at the same token at the same time'."""
    from bingxbot.config import CarryConfig
    from bingxbot.engine.carry import pick_carry_entry
    now = 1_700_000_000_000
    row = {"symbol": "BTC-USDT", "kind": "carry", "funding_apr": 0.9, "er_4h": 0.1,
           "dir_4h": 0, "next_funding_time": now + 3_600_000, "mark": 1.0,
           "funding_rate": 0.9 / 1095}
    # engine ctx symbols are passed inside `held` by CarryDesk._maybe_enter
    picked, _ = pick_carry_entry([row], {"BTC-USDT"}, CarryConfig(), now)
    assert picked is None


# ------------------------------------------------- exactly-once risk settle

def _mk_trade(sym: str, pnl: float) -> TradeRecord:
    return TradeRecord(symbol=sym, side="LONG", qty=1.0, entry_price=100.0,
                       exit_price=100.0 + pnl, entry_ts=1, exit_ts=2,
                       pnl=pnl, fees=0.0, reason_open="t", reason_close="t")


@pytest.mark.asyncio
async def test_settle_risk_accounts_each_trade_exactly_once():
    from bingxbot.config import BotConfig
    from bingxbot.data.feed import SyntheticFeed
    from bingxbot.engine.brokers import PaperBroker
    from bingxbot.engine.trader import TraderEngine
    from bingxbot.risk.manager import RiskManager

    cfg = BotConfig()
    cfg.symbols = ["BTC-USDT"]
    feed = SyntheticFeed(cfg.symbols, "1m", warmup_bars=10, speed=1000.0, seed=1)
    pf = Portfolio(10_000.0, mode="paper")
    pf.trades.append(_mk_trade("BTC-USDT", -5.0))   # restored pre-existing trade
    risk = RiskManager(cfg.risk)
    engine = TraderEngine(cfg, feed, PaperBroker(pf, feed.states, {}, 0.0005, 0.0),
                          pf, risk, {})
    # the restored trade was already accounted in the restored day-state
    engine.settle_risk()
    assert risk.state.trades_today == 0
    # new closes (e.g. from the carry desk or a reconcile) get picked up once
    pf.trades.append(_mk_trade("ETH-USDT", -3.0))
    pf.trades.append(_mk_trade("SOL-USDT", 4.0))
    engine.settle_risk()
    assert risk.state.trades_today == 2
    assert risk.state.day_realized == pytest.approx(1.0)
    engine.settle_risk()   # idempotent
    assert risk.state.trades_today == 2
    # a paper reset (trades cleared) must not crash or double-count afterwards
    pf.trades.clear()
    engine.settle_risk()
    pf.trades.append(_mk_trade("BTC-USDT", 2.0))
    engine.settle_risk()
    assert risk.state.trades_today == 3


# ------------------------------------------- edge-flip exit trend discipline

def _long(stop=97.0):
    p = Position(symbol="BTC-USDT", side=LONG, qty=1.0, entry_price=100.0,
                 opened_ts=1, stop_price=stop)
    return p


def test_edge_flip_holds_through_pullback_in_supported_trend():
    """A shallow reversed edge while the 15m/1h backdrop still clearly says UP
    is a pullback, not a reversal — the position must be held, however many
    bars the wobble lasts. This is the exit-side mirror of the entry's hard
    MTF veto (the old behavior sold with-trend longs at pullback lows)."""
    from bingxbot.strategy.exits import AdaptiveExitManager
    ex = AdaptiveExitManager(RiskConfig())
    pos = _long()
    ex.attach(pos, atr=1.0, init_risk=3.0)
    row = {"eff_ratio": 0.5, "mtf_bias": 0.6, "funding_rate": 0.0}
    for bars in range(1, 7):
        _, reason = ex.manage(pos, 99.5, 100.2, 99.3, 1.0, row, edge=-0.25,
                              threshold=0.3, regime="TREND_UP", bars_held=bars)
        assert reason is None, f"supported-trend pullback exited at bar {bars}: {reason}"


def test_edge_flip_fires_after_persistence_when_backdrop_decays():
    from bingxbot.strategy.exits import EDGE_FLIP_BARS, AdaptiveExitManager
    ex = AdaptiveExitManager(RiskConfig())
    pos = _long()
    ex.attach(pos, atr=1.0, init_risk=3.0)
    row = {"eff_ratio": 0.5, "mtf_bias": 0.1, "funding_rate": 0.0}   # trend gone
    reasons = []
    for bars in range(1, EDGE_FLIP_BARS + 1):
        _, reason = ex.manage(pos, 99.5, 100.2, 99.3, 1.0, row, edge=-0.25,
                              threshold=0.3, regime="TREND_UP", bars_held=bars)
        reasons.append(reason)
    assert reasons[:-1] == [None] * (EDGE_FLIP_BARS - 1), "one noisy close must not exit"
    assert reasons[-1] == "edge reversed"
    # an intervening non-reversed close resets the persistence counter
    pos2 = _long()
    ex.attach(pos2, atr=1.0, init_risk=3.0)
    ex.manage(pos2, 99.5, 100.2, 99.3, 1.0, row, edge=-0.25, threshold=0.3,
              regime="TREND_UP", bars_held=1)
    ex.manage(pos2, 99.5, 100.2, 99.3, 1.0, row, edge=0.05, threshold=0.3,
              regime="TREND_UP", bars_held=2)
    assert pos2.edge_flip_bars == 0
    _, reason = ex.manage(pos2, 99.5, 100.2, 99.3, 1.0, row, edge=-0.25,
                          threshold=0.3, regime="TREND_UP", bars_held=3)
    assert reason is None


def test_severe_edge_reversal_overrides_trend_and_persistence():
    from bingxbot.strategy.exits import AdaptiveExitManager
    ex = AdaptiveExitManager(RiskConfig())
    pos = _long()
    ex.attach(pos, atr=1.0, init_risk=3.0)
    row = {"eff_ratio": 0.5, "mtf_bias": 0.6, "funding_rate": 0.0}   # backdrop still UP
    _, reason = ex.manage(pos, 99.5, 100.2, 99.3, 1.0, row, edge=-0.60,
                          threshold=0.3, regime="TREND_UP", bars_held=1)
    assert reason == "edge reversed hard", "a violent reversal must exit immediately"


# ------------------------------------------------- pullback (resting) entries

class _MutState:
    """Fake market state with a mutable tape price."""
    def __init__(self, px):
        self.last_price = px
        self.book = None

        class candles:
            last_close = 0.0


@pytest.mark.asyncio
async def test_paper_pullback_limit_fills_on_touch():
    from bingxbot.engine.brokers import PaperBroker
    from bingxbot.risk.manager import SizedOrder
    pf = Portfolio(10_000.0, mode="paper")
    st = _MutState(100.0)
    broker = PaperBroker(pf, {"BTC-USDT": st}, {}, taker_fee=0.0005, slippage_bps=0.0,
                         maker_adverse_bps=0.0)
    sized = SizedOrder(qty=1.0, notional=99.5, leverage=1, stop_price=97.0,
                       take_profit=0.0, risk_amount=2.5,
                       entry_limit=99.5, entry_wait_s=6.0)
    task = asyncio.get_running_loop().create_task(
        broker.open_position("BTC-USDT", LONG, sized, "pullback test", 0))
    await asyncio.sleep(1.2)
    assert not task.done(), "must rest while price stays above the limit"
    st.last_price = 99.4                      # the retrace touches the limit
    res = await asyncio.wait_for(task, timeout=5.0)
    assert res.ok and res.filled_price == pytest.approx(99.5)
    assert pf.positions["BTC-USDT"].entry_price == pytest.approx(99.5)


@pytest.mark.asyncio
async def test_paper_pullback_limit_abandons_when_price_runs():
    from bingxbot.engine.brokers import PaperBroker
    from bingxbot.risk.manager import SizedOrder
    pf = Portfolio(10_000.0, mode="paper")
    st = _MutState(100.0)                     # never retraces
    broker = PaperBroker(pf, {"BTC-USDT": st}, {}, taker_fee=0.0005, slippage_bps=0.0)
    sized = SizedOrder(qty=1.0, notional=99.0, leverage=1, stop_price=97.0,
                       take_profit=0.0, risk_amount=2.0,
                       entry_limit=99.0, entry_wait_s=1.5)
    res = await broker.open_position("BTC-USDT", LONG, sized, "pullback test", 0)
    assert not res.ok and "unfilled" in res.error
    assert not pf.positions, "an abandoned pullback entry must not open anything"


def test_pullback_depth_is_tuner_owned_and_off_by_default():
    from bingxbot.engine.backtest import TUNABLES
    assert StrategyConfig().entry_pullback_atr == 0.0, "ships off; the tuner explores it"
    lo, hi, grp, kind = TUNABLES["entry_pullback_atr"]
    assert grp == "strategy" and kind == "float" and lo == 0.0 and hi <= 2.0


def test_backtest_runs_with_pullback_entries():
    s = StrategyConfig(entry_pullback_atr=0.6)
    from bingxbot.engine.backtest import run_backtest
    candles = synthetic_candles("BTC-USDT", "5m", 8000, seed=11)
    res = run_backtest(candles, "BTC-USDT", "5m", s, RiskConfig(), starting_balance=10_000.0)
    assert "error" not in res
    # deterministic like every other mode
    res2 = run_backtest(candles, "BTC-USDT", "5m", s, RiskConfig(), starting_balance=10_000.0)
    assert res["stats"] == res2["stats"]


# ------------------------------------------------- sizing honors risk budget

def test_min_leverage_never_inflates_risk():
    """The exact failure from live: with a wide ATR stop, risk sizing wants
    LESS than min_leverage's worth of size; flooring size at the band minimum
    doubled/tripled the realized loss at the stop. Size must follow
    risk_per_trade exactly; min_leverage may only floor the exchange margin
    setting."""
    from bingxbot.risk.manager import RiskManager
    cfg = RiskConfig(risk_per_trade=0.008, min_leverage=2, max_leverage=7)
    rm = RiskManager(cfg)
    spec = ContractSpec("BTC-USDT", qty_precision=6, min_qty=0.000001, min_notional_usdt=1.0)
    equity, price = 10_000.0, 60_000.0
    stop_dist = price * 0.015           # 1.5% stop -> implied leverage ~0.53x
    sized = rm.size_entry(equity, price, stop_dist, LONG, spec)
    assert sized is not None
    loss_at_stop = sized.qty * stop_dist
    assert loss_at_stop == pytest.approx(equity * 0.008, rel=0.02), \
        f"loss at stop {loss_at_stop:.2f} must equal the 0.8% risk budget, not 2x it"
    assert sized.notional / equity < 1.0, "size must not be inflated to the band floor"
    assert sized.leverage >= cfg.min_leverage, "margin setting still respects the band floor"
    # a $100 account behaves identically in percentage terms
    small = rm.size_entry(100.0, price, stop_dist, LONG, spec)
    assert small is not None
    assert small.qty * stop_dist == pytest.approx(0.8, rel=0.05)


def test_stop_out_shows_full_mae():
    """An intrabar stop-out closed before any bar-close excursion update must
    still journal ~1R of adverse excursion — the exit price is an extreme."""
    pf = Portfolio(10_000.0, mode="paper")
    pos = Position(symbol="BTC-USDT", side=LONG, qty=1.0, entry_price=100.0,
                   opened_ts=1, stop_price=97.0)
    pos.init_risk = 3.0
    pf.positions["BTC-USDT"] = pos
    tr = pf.close_position("BTC-USDT", 97.0, 2, exit_fee=0.0, reason="stop loss",
                           planned_risk=3.0)
    assert tr is not None
    assert tr.mae_r == pytest.approx(1.0), "a 1R stop-out must show 1R of heat"


# ------------------------------------------------------- tunable clamping

def test_apply_tunables_clamps_to_bounds():
    s, r = StrategyConfig(), RiskConfig()
    apply_tunables_inplace(s, r, {"risk_per_trade": 5.0, "base_threshold": -3.0,
                                  "time_stop_bars": 100000})
    assert r.risk_per_trade <= 0.014, "an out-of-box apply must be clamped"
    assert s.base_threshold >= 0.12
    assert r.time_stop_bars <= 200
