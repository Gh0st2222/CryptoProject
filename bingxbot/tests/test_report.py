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
                "RADAR", "TRACK RECORD", "EFFECTIVE CONFIG",
                "24H RANGE CONTEXT", "META-MODEL", "RUNTIME / ACCELERATION"):
        assert hdr in txt, f"missing section {hdr}"
    assert "BINGX_API_KEY" not in txt and "api_secret" not in txt
    assert "no secrets" in txt
    # runtime section must state whether the compiled kernel is in play
    assert "backtest_kernel" in txt


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
        assert "range_pos" in txt, "24h section must carry live per-symbol numbers"

        # browser-JSON safety: hot()/snapshot() must serialize under strict
        # JSON even when a symbol's 24h window is unfilled (young listing) —
        # a bare NaN would kill the websocket payload client-side.
        import json as json_
        ctx = engine.ctx["BTC-USDT"]
        if ctx.last_row:
            ctx.last_row = dict(ctx.last_row, hi_24h=float("nan"),
                                lo_24h=float("nan"), range_pos_24h=float("nan"))
        json_.dumps(engine.hot(), allow_nan=False, default=str)
        json_.dumps(engine.snapshot(), allow_nan=False, default=str)
    finally:
        orch.engine = None
        await engine.stop()


@pytest.mark.asyncio
async def test_ws_client_reader_surfaces_disconnect():
    """The websocket push loop's disconnect signal: the reader task must
    finish when the client sends a disconnect frame OR the transport errors.
    Without it, every closed/refreshed tab left a zombie push loop spamming
    'socket.send() raised exception.' several times a second, forever."""
    from bingxbot.server.app import _client_reader

    class _WS:
        def __init__(self, msgs, err=None):
            self._msgs = list(msgs)
            self._err = err

        async def receive(self):
            if self._msgs:
                return self._msgs.pop(0)
            raise (self._err or RuntimeError("closed"))

    await asyncio.wait_for(_client_reader(_WS(
        [{"type": "websocket.receive", "text": "ping"},
         {"type": "websocket.disconnect"}])), 2.0)
    await asyncio.wait_for(_client_reader(_WS([], ConnectionResetError())), 2.0)


def test_journal_range_entry_analytics(tmp_path):
    """Direction-relative 24h-entry buckets: a LONG at the daily low and a
    SHORT at the daily high are both 'best-25%'; rows without the field are
    skipped, and the R distribution / per-symbol splits come along."""
    from bingxbot.engine.journal import TradeJournal
    j = TradeJournal(tmp_path / "j.jsonl")
    base = {"pnl": 1.0, "r": 0.5, "mode": "paper", "mfe_r": 1.0, "mae_r": 0.2}
    j.record({**base, "symbol": "A-USDT", "side": "LONG", "rpos24": 0.10})    # bought the low
    j.record({**base, "symbol": "A-USDT", "side": "SHORT", "rpos24": 0.92})   # sold the high
    j.record({**base, "symbol": "B-USDT", "side": "LONG", "rpos24": 0.95,
              "pnl": -1.0, "r": -1.0})                                        # chased the top
    j.record({**base, "symbol": "B-USDT", "side": "LONG"})                    # old row: no field
    s = j.summary()
    bre = s["by_range_entry"]
    assert bre["best-25%"]["n"] == 2
    assert bre["worst-25%"]["n"] == 1 and bre["worst-25%"]["pnl"] == -1.0
    assert sum(g["n"] for g in bre.values()) == 3, "rows without rpos24 are skipped"
    assert s["by_symbol"]["B-USDT"]["n"] == 2
    assert s["r_dist"]["p50"] == 0.5 and s["r_dist"]["avg_loss_r"] == -1.0
