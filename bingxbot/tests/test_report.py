"""The diagnostic resume must always build — idle or running — and never leak
secrets."""
import asyncio

import pytest

from bingxbot.config import BotConfig
from bingxbot.server.report import build_report


def test_report_builds_when_idle():
    from bingxbot.server.orchestrator import Orchestrator
    orch = Orchestrator(BotConfig())
    txt = build_report(orch)
    assert isinstance(txt, str) and len(txt) > 500
    for hdr in ("HEADER / SESSION", "RISK & HEALTH", "AUTO-TUNER", "CHAMPION VAULT",
                "RADAR", "TRACK RECORD", "EFFECTIVE CONFIG"):
        assert hdr in txt, f"missing section {hdr}"
    assert "BINGX_API_KEY" not in txt and "api_secret" not in txt
    assert "no secrets" in txt


@pytest.mark.asyncio
async def test_report_builds_with_running_engine(monkeypatch):
    from bingxbot.data.feed import SyntheticFeed
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
    try:
        await asyncio.sleep(1.5)
        orch.engine = engine
        txt = build_report(orch)
        assert "ERROR building section" not in txt, txt
        assert "PER-SYMBOL BRAINS" in txt and "BTC-USDT" in txt
    finally:
        orch.engine = None
        await engine.stop()
