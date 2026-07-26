"""The resume's SELF-CHECK section, and the engine invariant behind it.

A diagnostic dump that only prints values makes the reader do the
cross-referencing, and the failures worth catching are exactly the ones that
survive a careful read: a symbol whose spread cap it can never clear still
looks like a healthy symbol, and a brain holding a scalar that appears in
neither the config nor the overlay ledger looks like a normal brain.

`param_divergence` is the invariant: every writer of a brain's decision
scalars goes through `_apply_brain_params`, so a symbol's brain must always
equal (global set, overridden by that symbol's registered overlay).
"""
import pytest

from bingxbot.config import BotConfig
from bingxbot.data.feed import SyntheticFeed
from bingxbot.engine.brokers import PaperBroker
from bingxbot.engine.portfolio import Portfolio
from bingxbot.engine.trader import TraderEngine
from bingxbot.exchange.models import ContractSpec
from bingxbot.risk.manager import RiskManager


def _engine(symbols=("BTC-USDT", "ETH-USDT")):
    cfg = BotConfig()
    cfg.feed = "synthetic"
    cfg.symbols = list(symbols)
    cfg.strategy.interval = "1m"
    specs = {s: ContractSpec(s) for s in cfg.symbols}
    feed = SyntheticFeed(cfg.symbols, "1m", warmup_bars=60, speed=600.0, seed=5)
    pf = Portfolio(cfg.paper.starting_balance, mode="paper")
    broker = PaperBroker(pf, feed.states, specs, taker_fee=cfg.exchange.taker_fee,
                         slippage_bps=0.0)
    return TraderEngine(cfg, feed, broker, pf, RiskManager(cfg.risk), specs)


# ------------------------------------------------------- the engine invariant

def test_a_fresh_engine_has_no_divergence():
    assert _engine().param_divergence() == []


def test_setting_and_clearing_an_overlay_leaves_no_divergence():
    eng = _engine()
    eng.set_overlay("ETH-USDT", {"base_threshold": 0.06, "min_p_win": 0.4})
    assert eng.ctx["ETH-USDT"].brain.base_threshold == pytest.approx(0.06)
    assert eng.param_divergence() == [], "an applied overlay is not a divergence"
    eng.set_overlay("ETH-USDT", None)
    assert eng.ctx["ETH-USDT"].brain.base_threshold == pytest.approx(
        eng.cfg.strategy.base_threshold), "clearing must restore the global set"
    assert eng.param_divergence() == []


def test_an_overlay_dropped_without_reapplying_is_caught():
    """The exact shape this check exists for: the brain keeps overlay values
    while the overlay ledger no longer lists them, so every readout — config,
    overlays, dashboard — shows the global set while one symbol gates on
    something else entirely."""
    eng = _engine()
    eng.set_overlay("ETH-USDT", {"base_threshold": 0.06})
    eng.overlays.pop("ETH-USDT")          # ledger cleared, brain not re-applied
    found = eng.param_divergence()
    assert len(found) == 1
    d = found[0]
    assert d["symbol"] == "ETH-USDT" and d["param"] == "base_threshold"
    assert d["in_force"] == pytest.approx(0.06)
    assert d["expected"] == pytest.approx(eng.cfg.strategy.base_threshold)
    assert d["source"] == "global"


def test_a_stale_global_swap_is_caught_per_symbol():
    """A brain left on an old global value after the config moved on."""
    eng = _engine()
    eng.cfg.strategy.min_p_win = 0.55           # config moves, brains do not
    found = {(d["symbol"], d["param"]) for d in eng.param_divergence()}
    assert found == {("BTC-USDT", "min_p_win"), ("ETH-USDT", "min_p_win")}
    eng.hot_swap_params(eng.cfg.strategy)       # the supported way to move it
    assert eng.param_divergence() == []


def test_every_scalar_the_brain_decides_with_is_covered():
    """A scalar added to _apply_brain_params but not to the table would be
    written and never checked — the divergence class this catches would come
    straight back for that parameter."""
    eng = _engine()
    covered = {battr for _o, battr, _c, _f in eng._BRAIN_PARAMS}
    brain = eng.ctx["BTC-USDT"].brain
    for attr in covered:
        assert hasattr(brain, attr), f"{attr} is not a brain attribute"
    # threshold_adapt is global-only by construction (never overlaid)
    assert "threshold_adapt" not in covered


# --------------------------------------------------------- the report section

def _report(orch):
    from bingxbot.server.report import build_report

    return build_report(orch)


class _Orch:
    """The few attributes the report's self-check reads."""

    def __init__(self, eng):
        self.engine = eng
        self.cfg = eng.cfg
        self.autotuner = None
        self.champions = []
        self.symbol_overlays = {}
        self.journal = None
        self.record = None
        self.jobs = {}
        self.mode = "paper"
        self.scanner = None
        self.carry = None
        self.shadow = None
        self.active_champion_id = None


def test_the_report_reports_a_clean_engine_as_clean():
    orch = _Orch(_engine())
    txt = _report(orch)
    assert "SELF-CHECK" in txt
    body = txt.split("SELF-CHECK")[1]
    assert "no contradictions found" in body


def test_the_report_names_the_diverged_symbol_and_parameter():
    eng = _engine()
    eng.set_overlay("ETH-USDT", {"base_threshold": 0.06})
    eng.overlays.pop("ETH-USDT")
    txt = _report(_Orch(eng))
    assert "brain-params-diverged" in txt
    assert "ETH-USDT" in txt and "base_threshold" in txt


def test_the_report_flags_a_symbol_that_cannot_clear_its_spread_cap():
    eng = _engine()
    st = eng.feed.states["BTC-USDT"]
    st.spread_bps.value = eng.cfg.risk.max_spread_bps * 5      # a book it can never pass
    txt = _report(_Orch(eng))
    assert "spread-cap-unreachable" in txt
    assert "BTC-USDT" in txt


def test_a_failing_self_check_never_takes_the_report_down():
    """A diagnostic that can crash is useless exactly when it is needed: if the
    check itself throws, the section says so and every other section still
    builds."""
    eng = _engine()

    def boom():
        raise RuntimeError("boom")

    eng.param_divergence = boom
    txt = _report(_Orch(eng))
    assert "ERROR building section" in txt.split("SELF-CHECK")[1]
    assert "PER-SYMBOL BRAINS" in txt and "EFFECTIVE CONFIG" in txt
    assert len(txt) > 2000, "the rest of the report must still be there"
