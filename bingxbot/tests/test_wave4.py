"""Tests for wave 4: MAE/MFE excursions, per-symbol overlays, live-evidence
stats, and the carry funding-timing window."""
from bingxbot.config import CarryConfig, RiskConfig, StrategyConfig
from bingxbot.data.history import synthetic_candles
from bingxbot.engine.autotuner import BRAIN_PARAMS, select_overlays
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


def test_select_overlays_only_where_clearly_better():
    syms = ["BTC-USDT", "HYPE-USDT"]
    cand_params = [_params(0.2), _params(0.4)]
    # candidate 1 is far better on HYPE only; on BTC the applied set wins
    per_sym = [[1.0, 0.5], [0.8, 4.0]]
    applied_fits = [2.0, 0.6]
    ov = select_overlays(cand_params, per_sym, applied_fits, _params(0.3), syms)
    assert "BTC-USDT" not in ov, "nothing clearly beats the global set on BTC"
    assert "HYPE-USDT" in ov and ov["HYPE-USDT"]["params"]["base_threshold"] == 0.4
    assert set(ov["HYPE-USDT"]["params"]) <= set(BRAIN_PARAMS), "risk params never overlaid"


def test_select_overlays_requires_clear_positive():
    syms = ["ETH-USDT"]
    # best candidate is negative -> no overlay even though it beats the applied set
    ov = select_overlays([_params(0.2)], [[-0.5]], [-3.0], _params(0.3), syms)
    assert ov == {}
    # identical params to the applied set -> pointless, skipped
    ov = select_overlays([_params(0.3)], [[5.0]], [1.0], _params(0.3), syms)
    assert ov == {}


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


# ---------------------------------------------------------------- rotation

def test_research_rotation_advances_and_persists(tmp_path, monkeypatch):
    import asyncio

    import bingxbot.engine.autotuner as AT
    from bingxbot.config import BotConfig
    from bingxbot.engine.search import DEOptimizer
    from bingxbot.server.orchestrator import Orchestrator

    monkeypatch.setattr(AT, "ROTATE_EVERY_S", 0)   # rotate on every data pass
    orch = Orchestrator(BotConfig())
    tuner = AT.AutoTuner(orch)
    tuner.de.state_path = tmp_path / "de.json"

    class _Sc:
        top_volume = ["SOL-USDT", "XRP-USDT", "DOGE-USDT"]
    orch.scanner = _Sc()

    async def fake_get(sym):
        return [None] * 10
    tuner._get_candles = fake_get
    tuner._cache = {s: ([None] * 10, 1.0) for s in
                    ("SOL-USDT", "XRP-USDT", "DOGE-USDT", "BTC-USDT", "ETH-USDT")}

    seen = []
    for _ in range(4):
        asyncio.run(tuner._ensure_data())
        seen.append(tuner.research_symbol)
    assert len(set(seen[:3])) == 3, f"must tour the universe, saw {seen}"

    # the tour position survives a save/load (i.e. a restart)
    tuner.de.pop = [{"x": 1}] * 4
    tuner.de.fitness = [0.0] * 4
    tuner.de.save()
    de2 = DEOptimizer(state_path=tuner.de.state_path)
    de2.keys = tuner.de.keys
    raw = __import__("json").loads(tuner.de.state_path.read_text())
    assert raw["extra"]["research_symbol"] == tuner.research_symbol
    assert raw["extra"]["rot_idx"] == tuner._rot_idx


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
