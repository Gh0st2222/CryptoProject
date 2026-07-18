"""Tests for the money-focused overhaul: radar ranking, carry decisions,
log-wealth fitness ordering, and the liquidation-distance leverage guard."""
import numpy as np

from bingxbot.config import CarryConfig, RiskConfig
from bingxbot.engine.backtest import _fitness
from bingxbot.engine.carry import carry_entry_ok, carry_exit_reason, receiving_side
from bingxbot.engine.scanner import (annualize_funding, clean_perp,
                                     demo_universe, rank_universe,
                                     top_volume_universe, trend_read_4h)
from bingxbot.exchange.models import LONG, SHORT, ContractSpec
from bingxbot.risk.manager import MAINT_MARGIN_RATE, RiskManager


# ------------------------------------------------------------------- radar

def test_rank_universe_orders_by_edge_and_filters_illiquid():
    premium = [
        {"symbol": "HOT-USDT", "mark": 1.0, "funding_rate": 0.002, "next_funding_time": 0},
        {"symbol": "COLD-USDT", "mark": 1.0, "funding_rate": 0.00005, "next_funding_time": 0},
        {"symbol": "THIN-USDT", "mark": 1.0, "funding_rate": 0.005, "next_funding_time": 0},
    ]
    tickers = [
        {"symbol": "HOT-USDT", "last": 1.0, "quote_volume": 50e6, "change_pct": 1.0},
        {"symbol": "COLD-USDT", "last": 1.0, "quote_volume": 60e6, "change_pct": 0.2},
        {"symbol": "THIN-USDT", "last": 1.0, "quote_volume": 1e5, "change_pct": 9.0},  # illiquid
    ]
    rows = rank_universe(premium, tickers)
    syms = [r["symbol"] for r in rows]
    assert "THIN-USDT" not in syms, "illiquid perps must be filtered out"
    assert syms[0] == "HOT-USDT", "extreme funding should rank first"
    hot = rows[0]
    assert hot["kind"] == "carry"
    assert hot["carry_side"] == "SHORT"          # positive funding -> shorts receive
    assert abs(hot["funding_apr"] - 0.002 * 3 * 365) < 1e-9


def test_clean_perp_filters_junk_listings():
    # real majors pass
    for s in ("BTC-USDT", "ETH-USDT", "SOL-USDT", "DOGE-USDT", "PEPE-USDT", "LINK-USDT"):
        assert clean_perp(s), s
    # index products, USDC quotes, multiplied listings, long-tail memes don't
    for s in ("NCCOXAG2USD-USDT", "NCSINASDAQ100USD-USDT", "TRX-USDC", "UNI-USDC",
              "1000PEPE-USDT", "BROCCOLI-USDT", "MOODENG-USDT", "ESPORTS-USDT"):
        assert not clean_perp(s), s


def test_junk_carry_is_not_harvestable():
    """The exact failure from live: a 6.5M-volume meme printing -1010% APR must
    not rank as harvestable carry, and index products must not become trend."""
    premium = [
        {"symbol": "HOME-USDT", "mark": 1.0, "funding_rate": -0.00922, "next_funding_time": 0},   # -1010% APR
        {"symbol": "BTC-USDT", "mark": 60_000.0, "funding_rate": 0.0004, "next_funding_time": 0},  # 44% APR, liquid
        {"symbol": "NCCOXAG2USD-USDT", "mark": 30.0, "funding_rate": 0.0001, "next_funding_time": 0},
    ]
    tickers = [
        {"symbol": "HOME-USDT", "last": 1.0, "quote_volume": 8.1e6, "change_pct": -23.1},
        {"symbol": "BTC-USDT", "last": 60_000.0, "quote_volume": 2e9, "change_pct": 1.0},
        {"symbol": "NCCOXAG2USD-USDT", "last": 30.0, "quote_volume": 91e6, "change_pct": 1.6},
    ]
    trend = {"NCCOXAG2USD-USDT": {"er": 0.6, "dir": -1, "atr_pct": 0.01}}
    rows = {r["symbol"]: r for r in rank_universe(premium, tickers, trend)}
    assert "NCCOXAG2USD-USDT" not in rows, "index products are excluded entirely"
    assert rows["HOME-USDT"]["kind"] == "watch", "illiquid extreme funding = watch, never carry"
    assert rows["HOME-USDT"]["score"] < rows["BTC-USDT"]["score"], "junk cannot outrank a real carry"
    assert rows["BTC-USDT"]["kind"] == "carry"


def test_top_volume_universe_returns_actual_majors():
    tickers = [
        {"symbol": "BTC-USDT", "quote_volume": 2e9}, {"symbol": "ETH-USDT", "quote_volume": 9e8},
        {"symbol": "SOL-USDT", "quote_volume": 4e8}, {"symbol": "XRP-USDT", "quote_volume": 3e8},
        {"symbol": "DOGE-USDT", "quote_volume": 2e8},
        {"symbol": "NCCOXAG2USD-USDT", "quote_volume": 5e8},   # index product: out
        {"symbol": "BROCCOLI-USDT", "quote_volume": 6.5e6},    # long-tail meme: out
        {"symbol": "TRX-USDC", "quote_volume": 6.5e6},         # USDC quote: out
    ]
    uni = top_volume_universe(tickers, 10)
    assert uni[:2] == ["BTC-USDT", "ETH-USDT"]
    assert "NCCOXAG2USD-USDT" not in uni and "BROCCOLI-USDT" not in uni and "TRX-USDC" not in uni
    assert set(uni) == {"BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "DOGE-USDT"}


def test_trend_read_4h_direction():
    up = np.linspace(100, 130, 80) + np.random.default_rng(0).normal(0, 0.15, 80)
    dn = np.linspace(130, 100, 80) + np.random.default_rng(1).normal(0, 0.15, 80)
    assert trend_read_4h(up)["dir"] == 1
    assert trend_read_4h(dn)["dir"] == -1
    assert trend_read_4h(up)["er"] > 0.5, "clean line should read as efficient"


def test_demo_universe_ranks():
    prem, tick = demo_universe(seed=7)
    rows = rank_universe(prem, tick)
    assert rows, "demo board should produce rows"
    assert any(r["kind"] == "carry" for r in rows), "demo should include a squeezed perp"


# ------------------------------------------------------------------- carry

def test_carry_entry_and_exit_logic():
    cfg = CarryConfig()
    # rich funding, no opposing trend -> enter on the receiving side
    ok, _ = carry_entry_ok(apr=0.5, er_4h=0.1, dir_4h=0, cfg=cfg)
    assert ok
    assert receiving_side(0.001) == SHORT and receiving_side(-0.001) == LONG
    # receiving side would fight a strong 4h trend -> vetoed
    ok, why = carry_entry_ok(apr=0.5, er_4h=0.6, dir_4h=1, cfg=cfg)   # SHORT vs strong up
    assert not ok and "trend" in why
    # too small a payment -> not worth the risk
    ok, _ = carry_entry_ok(apr=0.1, er_4h=0.0, dir_4h=0, cfg=cfg)
    assert not ok
    # exits: normalized funding / flipped funding / max hold / trend turn
    assert carry_exit_reason(SHORT, apr=0.05, er_4h=0, dir_4h=0, held_hours=1, cfg=cfg) == "funding normalized"
    assert carry_exit_reason(SHORT, apr=-0.4, er_4h=0, dir_4h=0, held_hours=1, cfg=cfg) == "funding flipped"
    assert carry_exit_reason(SHORT, apr=0.5, er_4h=0, dir_4h=0, held_hours=99, cfg=cfg) == "carry max hold"
    assert carry_exit_reason(SHORT, apr=0.5, er_4h=0.6, dir_4h=1, held_hours=1, cfg=cfg) == "4h trend turned"
    assert carry_exit_reason(SHORT, apr=0.5, er_4h=0.1, dir_4h=0, held_hours=1, cfg=cfg) is None


# ----------------------------------------------------------------- fitness

def _st(ret, dd=0.05, pf=1.5, trades=30):
    return {"total_return": ret, "max_drawdown": dd, "profit_factor": pf,
            "trades": trades, "win_rate": 0.5}


def test_fitness_is_ordered_and_smooth():
    # more growth -> higher score
    assert _fitness(_st(0.20)) > _fitness(_st(0.10)) > _fitness(_st(0.02)) > 0
    # losers are negative and ordered: losing less scores higher
    assert 0 > _fitness(_st(-0.02, pf=0.9)) > _fitness(_st(-0.10, pf=0.9))
    # junkier losing (lower pf) is punished harder, never rewarded
    assert _fitness(_st(-0.05, pf=0.9)) > _fitness(_st(-0.05, pf=0.2))
    # convex drawdown penalty: same return, deeper dd -> much lower score
    assert _fitness(_st(0.10, dd=0.02)) > _fitness(_st(0.10, dd=0.08)) > _fitness(_st(0.10, dd=0.20))
    # sub-5-trade ramp keeps a gradient toward activity
    assert _fitness(_st(0.0, trades=4)) > _fitness(_st(0.0, trades=1))


# ---------------------------------------------------------------- liq guard

def test_liquidation_guard_caps_leverage():
    cfg = RiskConfig(min_leverage=2, max_leverage=7, risk_per_trade=0.01)
    rm = RiskManager(cfg)
    spec = ContractSpec("BTC-USDT", qty_precision=4, min_qty=0.0001, min_notional_usdt=2.0)
    price, equity = 50_000.0, 10_000.0
    # a WIDE stop (12%) at high leverage would sit beyond liquidation; the guard
    # must cap leverage so the stop fires first with headroom.
    stop_dist = price * 0.12
    sized = rm.size_entry(equity, price, stop_dist, LONG, spec)
    assert sized is not None
    stop_frac = stop_dist / price
    lev = sized.notional / equity
    liq_frac = 1.0 / lev - MAINT_MARGIN_RATE
    assert stop_frac <= 0.8 * liq_frac + 1e-9, \
        f"stop ({stop_frac:.3f}) must sit inside 80% of liq distance ({liq_frac:.3f}) at {lev:.2f}x"
    # a tight stop is unaffected by the guard (band still applies)
    tight = rm.size_entry(equity, price, price * 0.01, LONG, spec)
    assert tight is not None and tight.notional / equity <= cfg.max_leverage + 1e-9
