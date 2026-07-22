"""Tests for wave 4: MAE/MFE excursions, per-symbol specialists, live-evidence
stats, and the carry funding-timing window."""
from bingxbot.config import CarryConfig, RiskConfig, StrategyConfig
from bingxbot.data.history import synthetic_candles
from bingxbot.engine.autotuner import BRAIN_PARAMS, select_specialists
from bingxbot.engine.backtest import run_backtest
from bingxbot.engine.carry import ENTRY_WINDOW_MS, pick_carry_entry


# ------------------------------------------------------------------ MAE/MFE

def test_backtest_records_excursions():
    candles = synthetic_candles("BTC-USDT", "1m", 8000, seed=21)
    res = run_backtest(candles, "BTC-USDT", "1m", StrategyConfig(), RiskConfig(),
                       starting_balance=10_000.0)
    trades = res["trades"]
    assert trades, "need trades to measure excursions"
    assert all("mae_r" in t and "mfe_r" in t for t in trades)
    assert all(t["mae_r"] >= 0 and t["mfe_r"] >= 0 for t in trades)
    # excursions must actually move (bar extremes are tracked, not stuck at 0)
    assert any(t["mfe_r"] > 0.1 for t in trades)
    assert any(t["mae_r"] > 0.1 for t in trades)
    # a clear winner must have SHOWN at least roughly what it banked
    for t in trades:
        if t["r_multiple"] > 0.5:
            assert t["mfe_r"] >= t["r_multiple"] - 0.6, t


# ------------------------------------------------------------------ overlays

def _params(thr):
    p = {k: 1.0 for k in BRAIN_PARAMS}
    p["base_threshold"] = thr
    return p


def test_select_specialists_only_where_clearly_better():
    res = {
        # on BTC nothing beats the applied set -> no overlay
        "BTC-USDT": {"applied": (_params(0.3), 2.0, 1.4), "overlay": None,
                     "cands": [(_params(0.2), 1.0, 1.5), (_params(0.4), 0.8, 1.2)]},
        # on HYPE candidate 0.4 clearly wins with a profitable recent fold
        "HYPE-USDT": {"applied": (_params(0.3), 0.6, 1.1), "overlay": None,
                      "cands": [(_params(0.2), 0.5, 1.5), (_params(0.4), 4.0, 1.6)]},
    }
    ov = select_specialists(res)
    assert "BTC-USDT" not in ov, "nothing clearly beats the global set on BTC"
    assert ov["HYPE-USDT"]["params"]["base_threshold"] == 0.4
    assert set(ov["HYPE-USDT"]["params"]) <= set(BRAIN_PARAMS), "risk params never overlaid"


def test_select_specialists_gates_and_hysteresis():
    # a challenger with a losing recent fold (PF < 1) can NEVER take the seat
    ov = select_specialists({"ETH-USDT": {"applied": (_params(0.3), 0.5, 1.2), "overlay": None,
                                          "cands": [(_params(0.2), 9.0, 0.8)]}})
    assert ov == {}
    # clearly-negative fitness never wins either
    ov = select_specialists({"ETH-USDT": {"applied": (_params(0.3), -3.0, 0.5), "overlay": None,
                                          "cands": [(_params(0.2), -0.5, 1.3)]}})
    assert ov == {}
    # identical brain scalars to the applied set -> pointless, skipped
    ov = select_specialists({"ETH-USDT": {"applied": (_params(0.3), 1.0, 1.2), "overlay": None,
                                          "cands": [(_params(0.3), 5.0, 1.5)]}})
    assert ov == {}
    # hysteresis: a profitable incumbent that still beats the global keeps its
    # seat against a challenger that is only MARGINALLY better...
    res = {"SOL-USDT": {"applied": (_params(0.3), 1.0, 1.2),
                        "overlay": (_params(0.2), 2.0, 1.4),
                        "cands": [(_params(0.4), 2.05, 1.5)]}}
    ov = select_specialists(res)
    assert ov["SOL-USDT"]["params"]["base_threshold"] == 0.2, "incumbent keeps the seat"
    # ...but loses it to a challenger that clears the margin
    res["SOL-USDT"]["cands"] = [(_params(0.4), 3.0, 1.5)]
    ov = select_specialists(res)
    assert ov["SOL-USDT"]["params"]["base_threshold"] == 0.4
    # an incumbent that stopped beating the global set is cleared
    res["SOL-USDT"] = {"applied": (_params(0.3), 2.5, 1.4),
                       "overlay": (_params(0.2), 1.0, 1.1), "cands": []}
    assert select_specialists(res) == {}


# --------------------------------------------------------------- live stats

def test_champion_live_stats_recent_pf(tmp_path):
    from bingxbot.config import BotConfig
    from bingxbot.server import orchestrator as O
    orch = O.Orchestrator(BotConfig())
    orch.journal.rows = (
        [{"champion_id": "aaa", "pnl": 5.0}] * 10        # early: great
        + [{"champion_id": "aaa", "pnl": -2.0}] * 20     # recent: bleeding
    )
    allt = orch.champion_live_stats()["aaa"]
    recent = orch.champion_live_stats(recent_n=20)["aaa"]
    assert allt["trades"] == 30 and allt["pf"] > 1.0
    assert recent["trades"] == 20 and recent["pf"] == 0.0, \
        "recent window must expose the collapse the all-time numbers hide"


# ------------------------------------------------------------- carry timing

def _row(sym, apr, nft):
    return {"symbol": sym, "kind": "carry", "funding_apr": apr, "er_4h": 0.1,
            "dir_4h": 0, "next_funding_time": nft, "mark": 1.0, "funding_rate": apr / 1095}


def test_carry_entry_prefers_soonest_settlement_within_window():
    cfg = CarryConfig()
    now = 1_700_000_000_000
    rows = [_row("FAR-USDT", 0.9, now + 7 * 3_600_000),
            _row("SOON-USDT", 0.5, now + 1 * 3_600_000)]
    row, why = pick_carry_entry(rows, set(), cfg, now)
    assert row is not None and row["symbol"] == "SOON-USDT", why
    # all qualifiers outside the window -> wait, don't enter
    rows = [_row("FAR-USDT", 0.9, now + 7 * 3_600_000)]
    row, why = pick_carry_entry(rows, set(), cfg, now)
    assert row is None and "waiting funding window" in why
    # held symbols are skipped
    rows = [_row("SOON-USDT", 0.5, now + 3_600_000)]
    row, _ = pick_carry_entry(rows, {"SOON-USDT"}, cfg, now)
    assert row is None
    assert ENTRY_WINDOW_MS == 3 * 3_600_000
