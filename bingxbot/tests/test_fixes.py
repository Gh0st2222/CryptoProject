"""Tests for the deep-review fixes: causal (no-lookahead) MTF ladder, the
one-position-per-token guarantee, exactly-once risk settlement, tunable-bounds
clamping, and carry/token exclusivity."""
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


# ------------------------------------------------------- tunable clamping

def test_apply_tunables_clamps_to_bounds():
    s, r = StrategyConfig(), RiskConfig()
    apply_tunables_inplace(s, r, {"risk_per_trade": 5.0, "base_threshold": -3.0,
                                  "time_stop_bars": 100000})
    assert r.risk_per_trade <= 0.014, "an out-of-box apply must be clamped"
    assert s.base_threshold >= 0.12
    assert r.time_stop_bars <= 200
