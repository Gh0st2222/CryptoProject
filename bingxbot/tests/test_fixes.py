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


# --------------------------------------------- fusion over speaking desks only

def test_dormant_desks_do_not_dilute_the_fused_edge(monkeypatch):
    """Backtests run with the micro + carry desks dormant (no book/tape/funding
    data). A desk with NO speaking alphas must drop out of the fusion — leaving
    it in the denominator shrank every backtest edge ~40% vs live and broke
    threshold comparability between validation and trading."""
    from bingxbot.strategy import alphas as alpha_mod
    from bingxbot.strategy.alphas import DESKS
    from bingxbot.strategy.brain import TradingBrain
    for nm in DESKS["trend"]:
        monkeypatch.setitem(alpha_mod.ALPHAS, nm, lambda row, micro, ctx: 1.0)
    brain = TradingBrain()
    micro0 = {"obi": 0.0, "flow": 0.0, "cvd_slope": 0.0, "spread_bps": 1.0, "ticks_per_s": 0.0}
    ev = brain.score({}, micro0, {})
    # only the trend desk speaks (unanimous +1); every other desk is dormant.
    # The fused edge must be that desk's opinion, not a fifth of it.
    assert ev["edge"] == pytest.approx(1.0, abs=1e-9), \
        f"dormant desks diluted the edge to {ev['edge']}"


@pytest.mark.asyncio
async def test_pending_entries_reserve_position_slots():
    from bingxbot.config import BotConfig
    from bingxbot.data.feed import SyntheticFeed
    from bingxbot.engine.brokers import PaperBroker
    from bingxbot.engine.trader import TraderEngine
    from bingxbot.risk.manager import RiskManager

    cfg = BotConfig()
    cfg.symbols = ["BTC-USDT", "ETH-USDT"]
    cfg.risk.max_open_positions = 1
    feed = SyntheticFeed(cfg.symbols, "1m", warmup_bars=10, speed=1000.0, seed=2)
    pf = Portfolio(10_000.0, mode="paper")
    risk = RiskManager(cfg.risk)
    engine = TraderEngine(cfg, feed, PaperBroker(pf, feed.states, {}, 0.0005, 0.0),
                          pf, risk, {})
    assert engine.pending_entries() == 0
    engine.ctx["BTC-USDT"].pending_task = asyncio.get_running_loop().create_task(asyncio.sleep(5))
    try:
        assert engine.pending_entries() == 1
        ok, why = risk.can_enter(10_000.0, len(pf.positions) + engine.pending_entries(), 1.0)
        assert not ok and "max open positions" in why, \
            "a resting entry must reserve a slot against the position cap"
    finally:
        engine.ctx["BTC-USDT"].pending_task.cancel()


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


# --------------------------------------------- 24h range context features

def test_24h_range_features():
    candles = synthetic_candles("BTC-USDT", "15m", 800, seed=13)
    ff = FeatureFrame(candles_to_arrays(candles), interval="15m")
    w24 = 96                                  # 24h of 15m bars
    import numpy as np_
    h = np_.array([c.high for c in candles])
    l = np_.array([c.low for c in candles])
    assert float(ff.f["hi_24h"][-1]) == pytest.approx(h[-w24:].max())
    assert float(ff.f["lo_24h"][-1]) == pytest.approx(l[-w24:].min())
    rp = float(ff.f["range_pos_24h"][-1])
    assert 0.0 <= rp <= 1.0, "price sits inside its own 24h range by construction"
    assert float(ff.f["dist_hi_24h"][-1]) >= 0.0 and float(ff.f["dist_lo_24h"][-1]) >= 0.0
    # and they feed the meta-model feature vector
    from bingxbot.ml.meta import FEATURE_NAMES
    for k in ("range_pos_24h", "dist_hi_24h", "dist_lo_24h", "vwap24_dev"):
        assert k in FEATURE_NAMES


# --------------------------------------------- brain learning survives restarts

def test_brain_state_roundtrip():
    from bingxbot.strategy.brain import TradingBrain
    ff = FeatureFrame(candles_to_arrays(synthetic_candles("BTC-USDT", "5m", 1200, seed=6)),
                      interval="5m")
    micro0 = {"obi": 0.0, "flow": 0.0, "cvd_slope": 0.0, "spread_bps": 1.0, "ticks_per_s": 0.0}
    a = TradingBrain()
    for i in range(300, ff.n):
        a.evaluate(ff.row(i), micro0, {})
    assert a.graded > 100
    b = TradingBrain()
    assert b.load_state(a.state_dict()) is True
    assert b.graded == a.graded
    assert b.alpha_w == pytest.approx(a.alpha_w)
    assert b.calibrator.n == a.calibrator.n and b.calibrator.w == pytest.approx(a.calibrator.w)
    assert b.allocator.weights() == pytest.approx(a.allocator.weights())
    assert b.beta == pytest.approx(a.beta, abs=1e-6)
    # a changed alpha roster refuses the stale state
    st = a.state_dict()
    st["alphas"] = st["alphas"][:-1]
    assert TradingBrain().load_state(st) is False


def test_de_population_growth_on_load(tmp_path):
    from bingxbot.engine.search import DEOptimizer
    p = tmp_path / "tuner_state.json"
    small = DEOptimizer(pop_size=8, seed=1, state_path=p)
    small.seed_population()
    small.save()
    big = DEOptimizer(pop_size=16, seed=2, state_path=p)
    assert big.load() is True
    assert len(big.pop) == 16 and len(big.fitness) == 16
    assert big.pop[:8] == small.pop, "saved members survive; growth adds explorers"
    assert all(f <= -1e8 for f in big.fitness[8:]), "new members start unscored"


# ------------------------------------------- anti-churn threshold + Kelly b

def test_adaptive_threshold_never_loosens_below_base():
    """The rate-targeting adaptor may only TIGHTEN the entry gate above the
    OOS-validated base_threshold — loosening below base to chase a trade-rate
    target was the marginal-entry churn generator."""
    from bingxbot.strategy.brain import TradingBrain
    brain = TradingBrain(base_threshold=0.30, threshold_adapt=True,
                         target_trades_per_hour=6.0, bars_per_hour=4.0)
    brain._score_hist.extend([0.05] * 300)   # edges far below base -> quantile tiny
    brain._adapt_threshold()
    assert brain.threshold >= 0.30 - 1e-9, \
        f"threshold {brain.threshold} loosened below the validated base"
    brain._score_hist.clear()
    brain._score_hist.extend([0.8] * 300)    # edges running hot -> bar rises
    brain._adapt_threshold()
    assert brain.threshold > 0.30


def test_kelly_payoff_ratio_is_measured_from_realized_trades():
    from bingxbot.risk.manager import RiskManager
    rm = RiskManager(RiskConfig(expected_rr=2.2))
    assert rm.payoff_ratio("trend") == pytest.approx(2.2), "no sample -> prior"
    # realized trades: winners +0.9R, losers -1.0R -> measured b ~0.9, far
    # below the 2.2 assumption -> Kelly must size on the evidence
    for _ in range(15):
        rm.health.r_hist.append(0.9)
        rm.health.r_hist.append(-1.0)
    b = rm.payoff_ratio("trend")
    assert 0.8 <= b <= 1.1, f"measured payoff {b} should reflect realized ~0.9"


def test_react_tail_matches_full_tail_features():
    """The reactive scanner's shorter tail must produce the same last-row
    features as the full tail (exact for windowed indicators, negligible EMA
    warmup drift) — speed must not change what the brain sees."""
    candles = synthetic_candles("BTC-USDT", "15m", 1500, seed=5)
    full = FeatureFrame(candles_to_arrays(candles), interval="15m")
    short = FeatureFrame(candles_to_arrays(candles[-640:]), interval="15m")
    for key in ("atr_pct", "atr_pctile", "bb_pctb", "bb_width_pctile", "rsi_14",
                "vwap_dev", "eff_ratio", "mtf_bias", "mtf_align", "dc_pos"):
        a, b = float(full.f[key][-1]), float(short.f[key][-1])
        if np.isnan(a) and np.isnan(b):
            continue
        assert a == pytest.approx(b, rel=1e-3, abs=2e-3), f"{key}: full={a} short={b}"


# --------------------------------------------- accounting identity, any phase

def test_accounting_identity_holds_at_every_timestamp_phase():
    """The epoch-anchored MTF ladder makes trade paths depend on where candle
    timestamps sit inside higher-TF buckets. Whatever path results — including
    a position forced closed at history's end — start + sum(pnl) - funding must
    equal final equity EXACTLY. (This was a real bug: the forced close changed
    cash after the last equity record, so stats['equity'] went stale by the
    final exit fee in some phases.)"""
    from bingxbot.engine.backtest import run_backtest
    base = synthetic_candles("BTC-USDT", "1m", 6000, seed=21)
    for k in (0, 7, 23, 41, 58):        # shift the whole tape by k minutes
        shifted = [type(c)(c.ts + k * 60_000, c.open, c.high, c.low, c.close, c.volume)
                   for c in base]
        res = run_backtest(shifted, "BTC-USDT", "1m", StrategyConfig(), RiskConfig(),
                           starting_balance=10_000.0, collect_series=False)
        st = res["stats"]
        # identity check needs the trades; rerun path already stores pnl in stats
        delta = (10_000.0 + st["total_pnl"] - st["funding_paid"]) - st["equity"]
        assert abs(delta) < 1e-4, f"phase {k}m: identity off by {delta}"


# --------------------------------------------- shared-frame cache parity

def test_shared_featureframe_caches_change_nothing():
    """The tuner's shared-frame fast path (cached rows + precomputed alpha
    scores) must produce EXACTLY the trades of the plain path — speed can
    never be allowed to change what the brain sees."""
    from bingxbot.engine.backtest import candles_to_arrays, run_backtest
    from bingxbot.strategy.features import FeatureFrame
    candles = synthetic_candles("BTC-USDT", "5m", 4000, seed=19)
    plain = run_backtest(candles, "BTC-USDT", "5m", StrategyConfig(), RiskConfig(),
                         starting_balance=10_000.0, collect_series=False)
    ff = FeatureFrame(candles_to_arrays(candles), interval="5m")
    shared = run_backtest(candles, "BTC-USDT", "5m", StrategyConfig(), RiskConfig(),
                          starting_balance=10_000.0, collect_series=False, ff=ff)
    assert plain["stats"] == shared["stats"], "cache path diverged from the reference"


# --------------------------------------------- scale-out + measured correlation

def test_portfolio_scale_out_accounting():
    pf = Portfolio(10_000.0, mode="paper")
    pos = Position(symbol="BTC-USDT", side=LONG, qty=2.0, entry_price=100.0,
                   opened_ts=1, stop_price=97.0, entry_fee=0.2)
    pos.init_risk = 3.0
    pf.positions["BTC-USDT"] = pos
    pf.cash -= 0.2                              # entry fee left cash at open
    tr = pf.scale_out("BTC-USDT", 0.5, 106.0, 2, exit_fee=0.05, reason="scale out")
    assert tr is not None and tr.qty == pytest.approx(1.0)
    # banked: 1.0 * (106-100) - 0.05 exit fee - 0.1 entry-fee share
    assert tr.pnl == pytest.approx(6.0 - 0.05 - 0.1)
    assert tr.r_multiple == pytest.approx(tr.pnl / 3.0, rel=1e-6)
    rem = pf.positions["BTC-USDT"]
    assert rem.qty == pytest.approx(1.0) and rem.entry_fee == pytest.approx(0.1)
    assert rem.scaled_out is True
    # remaining half closes: totals must equal a single full close's economics
    tr2 = pf.close_position("BTC-USDT", 106.0, 3, exit_fee=0.05, reason="trail", planned_risk=3.0)
    assert tr.pnl + tr2.pnl == pytest.approx(2.0 * 6.0 - 0.2 - 0.1)
    assert pf.cash == pytest.approx(10_000.0 - 0.2 + 12.0 - 0.1, abs=1e-9)


def test_scale_out_fires_once_at_r_and_only_for_trend():
    from bingxbot.strategy.exits import AdaptiveExitManager
    ex = AdaptiveExitManager(RiskConfig(scaleout_rr=1.5, be_rr=0.5))
    pos = Position(symbol="X", side=LONG, qty=1.0, entry_price=100.0, opened_ts=1,
                   stop_price=97.0)
    ex.attach(pos, atr=1.0, init_risk=3.0)
    row = {"eff_ratio": 0.5, "mtf_bias": 0.6}
    _, r1 = ex.manage(pos, 102.0, 102.0, 101.0, 1.0, row, 0.4, 0.3, "TREND_UP", 3)
    assert r1 is None, "below scaleout_rr: hold"
    _, r2 = ex.manage(pos, 105.0, 105.0, 104.0, 1.0, row, 0.4, 0.3, "TREND_UP", 4)
    assert r2 == "scale out"
    pos.scaled_out = True                        # caller marks after execution
    _, r3 = ex.manage(pos, 105.5, 105.6, 105.0, 1.0, row, 0.4, 0.3, "TREND_UP", 5)
    assert r3 != "scale out", "scale-out must fire at most once"


def test_corr_haircut_math():
    import numpy as np
    from bingxbot.risk.manager import corr_haircut
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 200)
    assert corr_haircut(a, a, 0.65) == pytest.approx(0.4), "perfect corr -> hard haircut"
    b = rng.normal(0, 1, 200)                    # independent
    assert corr_haircut(a, b, 0.65) > 0.85, "uncorrelated add ~ full size"
    assert corr_haircut(a[:10], b[:10], 0.65) == 0.65, "insufficient data -> fallback"
    assert corr_haircut(a, -a, 0.65) == pytest.approx(1.0), "negative corr = hedge, full size"


# ------------------------------------------- portfolio-fitness OOS validation

def test_portfolio_folds_are_purged_and_disjoint():
    from bingxbot.engine.search import portfolio_folds
    cbs = {"A": list(range(4000)), "B": list(range(4000))}
    folds = portfolio_folds(cbs, k=3, tail_frac=0.40, warmup=300)
    assert len(folds) == 3
    traded_starts = []
    for fc in folds:
        assert set(fc) == {"A", "B"}
        # traded region starts `warmup` bars into each slice
        traded_starts.append(fc["A"][300])
    # traded regions are sequential and disjoint
    assert traded_starts == sorted(traded_starts)
    ends = [fc["A"][-1] for fc in folds]
    for k in range(len(folds) - 1):
        assert ends[k] < traded_starts[k + 1] + 1, "traded regions must not overlap"
    assert folds[-1]["A"][-1] == 3999, "last fold reaches the end of history"


def test_validate_params_portfolio_smoke():
    from bingxbot.engine.search import validate_params_portfolio
    cbs = {s: synthetic_candles(s, "5m", 1400, seed=i + 3)
           for i, s in enumerate(["BTC-USDT", "ETH-USDT"])}
    out = validate_params_portfolio({}, cbs, "5m",
                                    {s: ContractSpec(s) for s in cbs},
                                    0.0005, 1.0, StrategyConfig(), RiskConfig())
    assert "fitness" in out and isinstance(out["fitness"], float)
    assert out["stats"].get("trades", 0) >= 0


# ------------------------------------------------------- tunable clamping

def test_apply_tunables_clamps_to_bounds():
    s, r = StrategyConfig(), RiskConfig()
    apply_tunables_inplace(s, r, {"risk_per_trade": 5.0, "base_threshold": -3.0,
                                  "time_stop_bars": 100000})
    assert r.risk_per_trade <= 0.014, "an out-of-box apply must be clamped"
    assert s.base_threshold >= 0.12
    assert r.time_stop_bars <= 200
