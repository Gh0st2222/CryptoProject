"""Entry-gate audit: the X-ray must price with the decision's own inputs, dead
knobs must not linger in the config surface, and the day roll stays daily."""
import json
import types

from bingxbot.config import CONFIG_VERSION, BotConfig, StrategyConfig
from bingxbot.engine.backtest import ASSUMED_SPREAD_BPS, gate_ev


def _engine(tmp_path):
    from bingxbot.data.feed import SyntheticFeed
    from bingxbot.engine.brokers import PaperBroker
    from bingxbot.engine.journal import TradeJournal
    from bingxbot.engine.portfolio import Portfolio
    from bingxbot.engine.trader import TraderEngine
    from bingxbot.risk.manager import RiskManager
    cfg = BotConfig()
    cfg.symbols = ["BTC-USDT"]
    feed = SyntheticFeed(cfg.symbols, "15m", warmup_bars=10, seed=1)
    pf = Portfolio(1000.0, mode="paper")
    eng = TraderEngine(cfg, feed, PaperBroker(pf, feed.states, {}, 5e-4, 0.0), pf,
                       RiskManager(cfg.risk), {}, journal=TradeJournal(tmp_path / "j.jsonl"))
    return eng


def test_gate_xray_prices_with_decision_inputs(tmp_path):
    """The deciding chain prices costs with ASSUMED_SPREAD_BPS; the panel used
    to price the same rows with the MEASURED book spread — an X-ray that could
    show red on a trade the machine just took. The cost and EV rows must now
    reproduce the decision's arithmetic exactly, while the measured spread
    stays visible on the risk row (can_enter is its real gate)."""
    eng = _engine(tmp_path)
    ctx = eng.ctx["BTC-USDT"]
    st = eng.feed.states["BTC-USDT"]
    st.spread_bps = types.SimpleNamespace(get=lambda d=1.0: 4.0)   # wide-ish book
    row = {"atr_pct": 0.004, "atr": 100.0, "mtf_bias": 0.5, "mtf_align": 0.5,
           "eff_ratio": 0.6, "bb_pctb": 0.5, "funding_rate": 0.0, "ts": 1}
    ev = {"edge": 0.6, "p_win": 0.62, "regime": "TREND_UP", "threshold": 0.3}
    eng._build_gates(ctx, st, row, ev)
    gates = {g["n"]: g for g in ctx.gates}

    fees_rt, slip = eng._entry_costs("BTC-USDT")
    want_cost = ctx.brain.entry_report(0.6, 0.62, row, fees_rt, ASSUMED_SPREAD_BPS, slip)
    not_want = ctx.brain.entry_report(0.6, 0.62, row, fees_rt, 4.0, slip)
    assert gates["cost"]["d"] == next(r["d"] for r in want_cost if r["n"] == "cost")
    assert gates["cost"]["d"] != next(r["d"] for r in not_want if r["n"] == "cost"), \
        "the sanity check needs inputs where assumed and measured spread differ"

    want_ev = gate_ev(eng.cfg.risk, eng.risk.payoff_ratio("trend"), 0.62, row,
                      fees_rt, ASSUMED_SPREAD_BPS, slip, "trend")
    assert (gates["EV floor"]["ok"], gates["EV floor"]["d"]) == want_ev
    assert "spread 4.0bp" in gates["risk"]["d"], \
        "the measured spread belongs to the risk row, where its real gate lives"


def test_block_reason_uses_decision_spread(tmp_path):
    """The first-failing-gate message must be computed with the same assumed
    spread the decision used — a message priced off the live book can name a
    gate the decision never failed."""
    eng = _engine(tmp_path)
    ctx = eng.ctx["BTC-USDT"]
    row = {"atr_pct": 0.004, "mtf_bias": 0.0, "mtf_align": 0.5, "eff_ratio": 0.6,
           "bb_pctb": 0.5, "funding_rate": 0.0}
    ev = {"edge": 0.6, "regime": "TREND_UP"}
    fees_rt, slip = eng._entry_costs("BTC-USDT")
    why = eng._block_reason(ctx.brain, 0.6, 0.62, row, ev, fees_rt, 9.9, slip, 2.0)
    ok, want = ctx.brain.entry_ok(0.6, 0.62, row, fees_rt, ASSUMED_SPREAD_BPS, slip)
    if not ok:
        assert why == want
    assert "9.9" not in why, "the measured spread must not leak into decision math"


def test_dead_confirm_knob_is_gone(tmp_path):
    """entry_confirm_scans gated reactive intra-bar entries, which no longer
    exist (entries decide at bar close only). A config knob that promises a
    behavior the engine doesn't have is worse than no knob; stale stored
    configs carrying it must still load cleanly."""
    assert not hasattr(StrategyConfig(), "entry_confirm_scans")
    from bingxbot.config import load_config
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"version": CONFIG_VERSION, "mode": "paper",
                             "strategy": {"entry_confirm_scans": 3, "interval": "15m"}}))
    cfg = load_config(path=p)
    assert cfg.strategy.interval == "15m"
    assert not hasattr(cfg.strategy, "entry_confirm_scans")
