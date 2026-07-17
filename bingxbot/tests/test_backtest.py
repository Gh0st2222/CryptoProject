from bingxbot.config import RiskConfig, StrategyConfig
from bingxbot.data.history import synthetic_candles
from bingxbot.engine.backtest import run_backtest, run_optimizer


def test_backtest_runs_and_accounts_consistently():
    candles = synthetic_candles("BTC-USDT", "1m", 8000, seed=21)
    res = run_backtest(candles, "BTC-USDT", "1m", StrategyConfig(), RiskConfig(),
                       starting_balance=10_000.0)
    assert "error" not in res
    stats = res["stats"]
    assert stats["trades"] > 0, "expected at least one trade on 8000 synthetic bars"
    # equity bookkeeping: final equity == start + sum(trade pnl) - funding drag
    # (the sim charges assumed funding at every 8h boundary while holding)
    total_pnl = sum(t["pnl"] for t in res["trades"])
    assert stats["funding_paid"] >= 0.0
    assert abs((10_000.0 + total_pnl - stats["funding_paid"]) - stats["equity"]) < 1e-4
    assert 0.0 <= stats["win_rate"] <= 1.0
    assert res["equity_curve"], "equity curve missing"
    assert res["markers"], "trade markers missing"
    entries = [m for m in res["markers"] if m["kind"] == "entry"]
    exits = [m for m in res["markers"] if m["kind"] == "exit"]
    assert entries and exits


def test_backtest_deterministic():
    candles = synthetic_candles("BTC-USDT", "1m", 5000, seed=5)
    a = run_backtest(candles, "BTC-USDT", "1m", StrategyConfig(), RiskConfig())
    b = run_backtest(candles, "BTC-USDT", "1m", StrategyConfig(), RiskConfig())
    assert a["stats"] == b["stats"]


def test_backtest_stops_bound_losses():
    """With leverage, a single trade's loss is bounded by the hard risk cap
    (plus a gap-through allowance), not by risk_per_trade."""
    candles = synthetic_candles("BTC-USDT", "1m", 8000, seed=13)
    risk = RiskConfig(risk_per_trade=0.005, max_risk_hard_pct=0.035)
    res = run_backtest(candles, "BTC-USDT", "1m", StrategyConfig(), risk,
                       starting_balance=10_000.0)
    worst = min((t["pnl"] for t in res["trades"]), default=0.0)
    # hard cap 3.5% + a 1.6x gap/slippage allowance; worse means broken stops
    assert worst > -0.035 * 1.6 * 10_000.0


def test_optimizer_small_run():
    candles = synthetic_candles("BTC-USDT", "1m", 6000, seed=31)
    res = run_optimizer(candles, "BTC-USDT", "1m", StrategyConfig(), RiskConfig(),
                        n_trials=6, seed=7)
    assert "error" not in res
    assert len(res["finalists"]) >= 1
    for f in res["finalists"]:
        assert "valid_fitness" in f and "params" in f
