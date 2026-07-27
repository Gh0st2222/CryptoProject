"""The resume must say WHICH fold length is starving the funnel.

"No champions" reads identically whether the judge's folds are too short to
carry a verdict or the search's are too short to rank anything, and both of
those really happened. The self-check states both numbers, so an owner reading
the resume does not have to rediscover either.

A warning nobody ever fires is worth nothing, so these drive the real report
builder rather than the threshold arithmetic.
"""
import asyncio

import pytest

from bingxbot.config import BotConfig
from bingxbot.engine.autotuner import MIN_OOS_TRADED_BARS, MIN_TRAIN_TRADED_BARS
from bingxbot.server.report import build_report

HEALTHY = {"oos_folds": 4, "oos_traded_bars": MIN_OOS_TRADED_BARS + 500,
           "train_folds": 3, "train_traded_bars": MIN_TRAIN_TRADED_BARS + 300,
           "bar": 0.5, "best_fitness": 0.4, "cands_judged": 10, "pf_passed": 2}


@pytest.fixture
async def report():
    """The self-check only runs against a live engine, so these drive one."""
    from bingxbot.data.feed import SyntheticFeed
    from bingxbot.engine.autotuner import AutoTuner
    from bingxbot.engine.brokers import PaperBroker
    from bingxbot.engine.portfolio import Portfolio
    from bingxbot.engine.trader import TraderEngine
    from bingxbot.risk.manager import RiskManager
    from bingxbot.server.orchestrator import Orchestrator

    cfg = BotConfig()
    cfg.symbols = ["BTC-USDT"]
    orch = Orchestrator(cfg)
    feed = SyntheticFeed(cfg.symbols, "1m", warmup_bars=20, speed=1000.0, seed=6)
    pf = Portfolio(10_000.0, mode="paper")
    engine = TraderEngine(cfg, feed, PaperBroker(pf, feed.states, {}, 0.0005, 0.0),
                          pf, RiskManager(cfg.risk), {})
    await engine.start()
    orch.engine = engine
    orch.autotuner = AutoTuner(orch)
    orch.autotuner.cycles = 40
    try:
        def build(**last_cycle) -> str:
            orch.autotuner.last_cycle = {**HEALTHY, **last_cycle}
            return build_report(orch)
        yield build
    finally:
        await engine.stop()


async def test_healthy_fold_geometry_raises_neither_warning(report):
    txt = report()
    assert "training-folds-too-short" not in txt
    assert "judged-folds-too-short" not in txt


async def test_short_search_folds_are_named(report):
    """450 traded bars is what the core-count rule produced on an 8-worker host,
    and at that length the search's ranking correlates with out-of-sample return
    at rho +0.01."""
    txt = report(train_traded_bars=450)
    assert "training-folds-too-short" in txt
    assert "450" in txt
    assert "judged-folds-too-short" not in txt, "the judge's folds were fine"


async def test_short_judged_folds_are_named(report):
    txt = report(oos_traded_bars=600)
    assert "judged-folds-too-short" in txt
    assert "training-folds-too-short" not in txt


async def test_both_can_fire_at_once(report):
    txt = report(train_traded_bars=450, oos_traded_bars=600)
    assert "training-folds-too-short" in txt and "judged-folds-too-short" in txt


@pytest.mark.parametrize("value", [None, 0])
async def test_an_absent_measurement_is_not_a_failure(report, value):
    """A cycle that never recorded the number must not be reported as a broken
    one -- the report is read as evidence, so a guess in it is worse than a gap."""
    assert "training-folds-too-short" not in report(train_traded_bars=value)


def test_the_unreachable_bar_note_points_at_something_that_exists():
    """The advice in a warning is the whole value of the warning. This one sent
    the reader to MIN_ABS_FITNESS, a constant deleted when the stand-down moved
    onto pooled economics."""
    import inspect

    from bingxbot.server import report
    src = inspect.getsource(report)
    assert "MIN_ABS_FITNESS" not in src
    for name in ("MIN_OOS_TRADED_BARS", "MIN_TRAIN_TRADED_BARS"):
        assert name in src, f"the self-check should still name {name}"
