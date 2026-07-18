"""Tests for overhaul #2: carry lab replay, paper persistence, track record,
1m display aggregation, and focus selection."""
import asyncio
import time

import numpy as np

from bingxbot.config import BotConfig, RiskConfig
from bingxbot.data.feed import MarketState
from bingxbot.data.history import synthetic_candles
from bingxbot.engine.carrylab import (grid_search, recommend, replay_carry,
                                      synthetic_funding)
from bingxbot.engine.persist import (clear_paper_state, load_paper_state,
                                     restore_into, save_paper_state)
from bingxbot.engine.portfolio import Portfolio
from bingxbot.engine.record import TrackRecord
from bingxbot.exchange.models import LONG, SHORT, Candle, Position, Tick
from bingxbot.risk.manager import RiskManager


# --------------------------------------------------------------- carry lab

def _flat_candles(days: int, px: float = 100.0) -> list[Candle]:
    t0 = 1_700_000_000_000
    return [Candle(t0 + i * 3_600_000, px, px * 1.001, px * 0.999, px, 10.0)
            for i in range(days * 24)]


def test_carry_replay_collects_extreme_funding_on_flat_prices():
    candles = _flat_candles(30)
    # persistent +0.2%/8h funding (219% APR): shorts receive every settlement
    funding = [{"ts": candles[0].ts + i * 8 * 3_600_000, "rate": 0.002} for i in range(90)]
    r = replay_carry(funding, candles, min_apr=0.35, exit_apr=0.10)
    assert r["entries"] >= 1
    assert r["funding_ret"] > 0.02, "held through prints -> collected funding"
    assert abs(r["price_ret"]) < 0.01, "flat tape -> negligible price pnl"
    assert r["net"] > 0, "carry on flat prices with rich funding must be profitable"


def test_carry_replay_ignores_benign_funding():
    candles = _flat_candles(30)
    funding = [{"ts": candles[0].ts + i * 8 * 3_600_000, "rate": 0.0001} for i in range(90)]
    r = replay_carry(funding, candles, min_apr=0.35, exit_apr=0.10)
    assert r["entries"] == 0, "0.01%/8h is baseline, not an edge"


def test_carry_grid_and_recommend():
    candles = _flat_candles(40)
    funding = synthetic_funding(40, seed=3, start_ts=candles[0].ts)
    grids = [grid_search(funding, candles)]
    assert grids[0], "grid should produce combos"
    rec = recommend(grids)
    assert rec is None or (rec["min_apr"] >= 0.2 and rec["entries"] >= 0)


# ------------------------------------------------------------- persistence

def test_paper_state_roundtrip(tmp_path):
    p = tmp_path / "paper_state.json"
    pf = Portfolio(1000.0, mode="paper")
    pf.cash = 1042.5
    pf.funding_paid = 1.25
    pf.positions["BTC-USDT"] = Position(symbol="BTC-USDT", side=LONG, qty=0.01,
                                        entry_price=50_000.0, opened_ts=123,
                                        leverage=3, stop_price=49_000.0)
    pf.equity_curve.append((111, 1040.0))
    rm = RiskManager(RiskConfig())
    rm.state.day_realized = 7.5
    rm.state.consecutive_losses = 2
    save_paper_state(pf, rm.state, path=p)

    pf2 = Portfolio(1000.0, mode="paper")
    rm2 = RiskManager(RiskConfig())
    snap = load_paper_state(1000.0, path=p)
    assert snap is not None
    n = restore_into(pf2, rm2, snap)
    assert n == 1 and "BTC-USDT" in pf2.positions
    assert abs(pf2.cash - 1042.5) < 1e-9 and abs(pf2.funding_paid - 1.25) < 1e-9
    assert pf2.positions["BTC-USDT"].stop_price == 49_000.0
    assert rm2.state.day_realized == 7.5 and rm2.state.consecutive_losses == 2
    # changed starting balance -> refuse to restore (fresh account)
    assert load_paper_state(2000.0, path=p) is None
    clear_paper_state(path=p)
    assert load_paper_state(1000.0, path=p) is None


# ------------------------------------------------------------ track record

def test_track_record_rolls_on_day_change(tmp_path):
    tr = TrackRecord(path=tmp_path / "rec.jsonl")
    pf = Portfolio(1000.0, mode="paper")
    pf.equity_curve.append((1, 1010.0))
    tr._day = "2026-07-16"                      # pretend we started yesterday
    from bingxbot.exchange.models import TradeRecord
    import calendar
    day_ts = calendar.timegm(time.strptime("2026-07-16", "%Y-%m-%d")) * 1000 + 3_600_000
    pf.trades.append(TradeRecord("BTC-USDT", "LONG", 0.01, 100, 110, day_ts,
                                 day_ts + 3600_000, pnl=5.0, fees=0.1,
                                 reason_open="x", reason_close="y", mode="paper"))
    assert tr.maybe_roll(pf, "paper") is True   # UTC today != 2026-07-16
    assert tr.rows and tr.rows[-1]["d"] == "2026-07-16"
    assert tr.rows[-1]["pnl"] == 5.0 and tr.rows[-1]["trades"] == 1
    # same day again -> no double roll
    assert tr.maybe_roll(pf, "paper") is False
    # reload from disk keeps the row
    tr2 = TrackRecord(path=tmp_path / "rec.jsonl")
    assert tr2.rows[-1]["d"] == "2026-07-16"


# --------------------------------------------------------- display candles

def test_display_series_aggregates_ticks_to_1m():
    st = MarketState("BTC-USDT")
    t0 = 1_700_000_000_000 - (1_700_000_000_000 % 60_000)
    st.on_tick(Tick(ts=t0 + 1000, price=100.0, qty=1.0, is_buyer_maker=False))
    st.on_tick(Tick(ts=t0 + 30_000, price=103.0, qty=1.0, is_buyer_maker=False))
    st.on_tick(Tick(ts=t0 + 59_000, price=101.0, qty=1.0, is_buyer_maker=True))
    p = st.display.partial
    assert p is not None and p.ts == t0
    assert p.open == 100.0 and p.high == 103.0 and p.close == 101.0
    # next minute: previous bar closes into the series
    st.on_tick(Tick(ts=t0 + 61_000, price=102.0, qty=1.0, is_buyer_maker=False))
    assert len(st.display) == 1
    closed = st.display.tail(1)[0]
    assert closed.ts == t0 and closed.close == 101.0
    assert st.display.partial.ts == t0 + 60_000


# ------------------------------------------------------------- journal fix

def test_alignment_bucket_is_side_relative(tmp_path):
    """A SHORT taken while the higher-TF bias points DOWN is WITH-trend — the
    old bucket keyed on raw bias sign and mislabeled every such trade."""
    from bingxbot.engine.journal import TradeJournal
    j = TradeJournal(path=tmp_path / "j.jsonl")
    j.record({"pnl": 1.0, "side": "SHORT", "mtf_bias": -0.6, "mode": "paper"})   # with-trend short
    j.record({"pnl": 1.0, "side": "LONG", "mtf_bias": 0.6, "mode": "paper"})     # with-trend long
    j.record({"pnl": -1.0, "side": "LONG", "mtf_bias": -0.6, "mode": "paper"})   # counter-trend long
    s = j.summary()
    ali = s["by_alignment"]
    assert ali.get("with-trend+", {}).get("n") == 2
    assert ali.get("counter-trend+", {}).get("n") == 1


def test_default_params_cover_all_tunables():
    from bingxbot.engine.autotuner import _default_params
    from bingxbot.engine.backtest import TUNABLES
    d = _default_params()
    assert set(d) == set(TUNABLES)
    for k, (lo, hi, _g, _kind) in TUNABLES.items():
        assert lo - 1e-9 <= float(d[k]) <= hi + 1e-9 or isinstance(d[k], bool), k


# ----------------------------------------------------------------- focus

def test_focus_prefers_position_then_strongest_edge():
    from bingxbot.data.feed import SyntheticFeed
    from bingxbot.engine.portfolio import Portfolio as PF
    from bingxbot.engine.trader import TraderEngine

    async def run():
        cfg = BotConfig()
        cfg.symbols = ["BTC-USDT", "ETH-USDT"]
        feed = SyntheticFeed(cfg.symbols, "15m", warmup_bars=10, speed=100, seed=1)
        pf = PF(1000.0, mode="paper")
        rm = RiskManager(cfg.risk)
        eng = TraderEngine(cfg, feed, None, pf, rm, {})
        eng.ctx["BTC-USDT"].last_eval = {"edge": 0.10, "threshold": 0.3}
        eng.ctx["ETH-USDT"].last_eval = {"edge": 0.28, "threshold": 0.3}
        assert eng.focus_symbol() == "ETH-USDT"      # closest to firing
        pf.positions["BTC-USDT"] = Position(symbol="BTC-USDT", side=SHORT, qty=1,
                                            entry_price=100.0, opened_ts=1)
        assert eng.focus_symbol() == "BTC-USDT"      # a position always wins
    asyncio.run(run())
