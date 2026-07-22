"""Compiled-kernel parity: the numba engine must reproduce the Python engine
trade-for-trade (entry ts, exit ts, pnl) before it is allowed to rank anything.
"Fast but slightly different" is worse than slow."""
import numpy as np
import pytest

from bingxbot.config import RiskConfig, StrategyConfig
from bingxbot.data.history import synthetic_candles
from bingxbot.engine.backtest import candles_to_arrays, run_backtest
from bingxbot.exchange.models import ContractSpec
from bingxbot.strategy.features import FeatureFrame

numba = pytest.importorskip("numba")


@pytest.fixture(autouse=True)
def _no_meta(monkeypatch):
    # the kernel implements the meta-free brain; compare against the same
    import bingxbot.strategy.brain as brain_mod
    monkeypatch.setattr(brain_mod, "_get_meta", lambda *a, **k: None)


CASES = [
    (4, {}, {}),
    (21, {"entry_pullback_atr": 0.5}, {"scaleout_rr": 1.2, "trail_scale_trend": 1.3}),
    (7, {"entry_mode": "taker", "trade_range": True}, {"be_rr": 0.6}),
]


@pytest.mark.parametrize("seed,sp,rp", CASES)
def test_kernel_matches_python_trade_for_trade(seed, sp, rp):
    from bingxbot.engine.kernel import kernel_fitness
    strat, risk = StrategyConfig(**sp), RiskConfig(**rp)
    spec = ContractSpec("BTC-USDT")
    candles = synthetic_candles("BTC-USDT", "5m", 2500, seed=seed)
    ff = FeatureFrame(candles_to_arrays(candles), interval="5m")
    py = run_backtest(candles, "BTC-USDT", "5m", strat, risk, spec, taker_fee=0.0005,
                      slippage_bps=1.0, collect_series=True, ff=ff)
    kr = kernel_fitness(ff, strat, risk, spec, 0.0005, 1.0, "5m")
    pt = py["trades"]
    assert len(pt) == kr["stats"]["trades"], "trade count diverged"
    for a, ots, ets, pnl in zip(pt, kr["trade_open_ts"], kr["trade_ts"], kr["trade_pnl"]):
        assert a["entry_ts"] == ots and a["exit_ts"] == ets, "trade timing diverged"
        assert a["pnl"] == pytest.approx(pnl, abs=1e-9), "trade pnl diverged"
    assert py["stats"]["total_pnl"] == pytest.approx(kr["stats"]["total_pnl"], abs=1e-6)
    assert py["stats"]["max_drawdown"] == pytest.approx(kr["stats"]["max_drawdown"], abs=1e-6)


def test_score_fold_kernel_and_python_agree(monkeypatch):
    """The tuner's actual entry point must rank identically on both engines."""
    from bingxbot.engine.search import score_fold
    candles = synthetic_candles("BTC-USDT", "5m", 1500, seed=11)
    params = [{"base_threshold": 0.2}, {"base_threshold": 0.35}, {"risk_per_trade": 0.012}]
    args = ("BTC-USDT", "5m", ContractSpec("BTC-USDT"), 0.0005, 1.0,
            StrategyConfig(), RiskConfig(), params)
    fast = score_fold(candles, *args)
    monkeypatch.setenv("BOT_NO_KERNEL", "1")
    slow = score_fold(candles, *args)
    assert fast == pytest.approx(slow, abs=1e-9), f"kernel {fast} vs python {slow}"


def test_backtest_kernel_constants_mirror():
    """The kernel re-declares backtest constants for nopython access — if they
    ever drift apart, parity breaks silently. Pin them together here."""
    from bingxbot.engine import backtest as B
    from bingxbot.engine import kernel as K
    for name in ("ASSUMED_SPREAD_BPS", "EV_MARGIN", "FILL_THROUGH_BPS",
                 "STOP_SLIP_MULT", "FUNDING_MS", "ASSUMED_FUNDING_8H"):
        assert getattr(B, name) == getattr(K, name), f"constant drift: {name}"
