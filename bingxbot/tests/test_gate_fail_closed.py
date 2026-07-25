"""A gate's job is to REFUSE, so the one direction it must never fail is OPEN.

Every guard in this project written as a bare comparison has failed open on NaN,
because every comparison with NaN is False. In the gates that meant permitting a
trade whose conviction, win probability or cost could not be evaluated at all —
including `gate_ev`, whose docstring calls it "the gate that refuses coin-flip
entries".

The oracle below only considers the arguments each gate actually RECEIVES: an
earlier version of this fuzz flagged gate_ev for a non-finite `edge`, which
gate_ev never sees, and that was a bug in the test rather than the code.
"""
import itertools
import math

import pytest

from bingxbot.config import RiskConfig, StrategyConfig
from bingxbot.engine.backtest import (_entry_signal_ok, gate_ev, gate_funding,
                                      gate_mtf_veto, gate_regime)
from bingxbot.strategy.brain import TradingBrain
from bingxbot.strategy.regime import detect_regime

NAN, INF = float("nan"), float("inf")
ROW_KEYS = ("atr", "atr_pct", "atr_pctile", "eff_ratio", "mtf_align", "mtf_bias",
            "close", "adx", "ema21_slope", "ema_8", "ema_21", "ema_55",
            "bb_pctb", "dc_hi", "dc_lo", "funding_rate", "rsi_14")

ROWS = {"zeros": {k: 0.0 for k in ROW_KEYS}, "nan": {k: NAN for k in ROW_KEYS},
        "inf": {k: INF for k in ROW_KEYS}, "empty": {},
        "healthy": {k: 0.5 for k in ROW_KEYS}}
EDGES = [0.0, 0.5, -0.5, NAN, INF]
PWINS = [0.0, 0.5, 0.9, NAN, INF, -1.0]
COSTS = [0.0, 0.001, NAN, INF]
BS = [1.0, 2.5, NAN, INF, -1.0]


def _brain():
    return TradingBrain(base_threshold=0.30)


def test_no_gate_raises_or_returns_a_non_bool():
    strat, risk, brain = StrategyConfig(), RiskConfig(), _brain()
    n = 0
    for rname, row in ROWS.items():
        for edge, p_win, fee, b in itertools.product(EDGES, PWINS, COSTS, BS):
            n += 1
            for name, res in (
                ("gate_mtf_veto", gate_mtf_veto(strat, edge, row)),
                ("gate_funding", gate_funding(strat, edge, row)),
                ("gate_regime", gate_regime(strat, edge, row, "TREND_UP")),
                ("gate_ev", gate_ev(risk, b, p_win, row, fee, 1.0, 1.0)),
                ("entry_ok", brain.entry_ok(edge, p_win, row, fee, 1.0, 1.0)),
            ):
                ok = res[0] if isinstance(res, tuple) else res
                assert isinstance(ok, bool), f"{name} returned {ok!r} on {rname}"
    assert n > 1000


def test_gate_ev_refuses_when_it_cannot_evaluate():
    """Only gate_ev's OWN inputs: payoff_b, p_win, row, fees, spread, slippage."""
    risk = RiskConfig()
    row = {"atr_pct": 0.01}
    for bad in ({"p_win": NAN}, {"p_win": INF}, {"b": NAN}, {"fee": NAN},
                {"fee": INF}, {"spread": NAN}, {"slip": NAN}):
        kw = {"b": 2.5, "p_win": 0.9, "fee": 0.001, "spread": 1.0, "slip": 1.0}
        kw.update(bad)
        ok, why = gate_ev(risk, kw["b"], kw["p_win"], row, kw["fee"],
                          kw["spread"], kw["slip"])
        assert ok is False, f"EV gate passed on {bad}: {why}"


def test_entry_ok_refuses_when_it_cannot_evaluate():
    brain = _brain()
    row = {"atr_pct": 0.01}
    for bad in ({"edge": NAN}, {"edge": INF}, {"p_win": NAN}, {"p_win": INF},
                {"fee": NAN}, {"spread": NAN}, {"slip": NAN}):
        kw = {"edge": 0.5, "p_win": 0.9, "fee": 0.001, "spread": 1.0, "slip": 1.0}
        kw.update(bad)
        ok, why = brain.entry_ok(kw["edge"], kw["p_win"], row, kw["fee"],
                                 kw["spread"], kw["slip"])
        assert ok is False, f"entry gate passed on {bad}: {why}"


def test_the_composite_chain_refuses_too():
    strat, risk, brain = StrategyConfig(), RiskConfig(), _brain()
    row = {"atr_pct": 0.01, "atr": 1.0, "mtf_bias": 0.0, "eff_ratio": 0.5}
    for edge, p_win in ((NAN, 0.9), (0.5, NAN), (INF, 0.9), (0.5, INF)):
        ev = {"edge": edge, "p_win": p_win, "threshold": 0.3, "regime": "TREND_UP"}
        assert _entry_signal_ok(brain, strat, risk, edge, p_win, row, ev,
                                0.001, 1.0, 2.0) is False, (edge, p_win)


def test_the_xray_never_shows_pass_where_the_gate_refuses():
    """The X-ray drives the dashboard. It must not claim a gate passed that the
    engine is refusing — that mismatch has bitten this project before."""
    brain = _brain()
    row = {"atr_pct": 0.01}
    for edge, p_win in ((NAN, 0.9), (0.5, NAN), (INF, INF)):
        ok, _ = brain.entry_ok(edge, p_win, row, 0.001, 1.0, 1.0)
        assert ok is False
        rep = brain.entry_report(edge, p_win, row, 0.001, 1.0, 1.0)
        assert not all(g["ok"] for g in rep), \
            f"X-ray showed every gate passing while entry_ok refused ({edge}, {p_win})"


def test_detect_regime_and_kelly_stay_finite():
    brain = _brain()
    for rname, row in ROWS.items():
        reg, conf = detect_regime(row)
        assert isinstance(reg, str) and math.isfinite(conf), rname
    for p, b in itertools.product(PWINS, BS):
        k = brain.kelly_size_mult(p, b)
        assert math.isfinite(k) and k >= 0, f"kelly({p}, {b}) = {k}"


def test_healthy_input_still_passes_every_gate():
    """The guards must refuse the unevaluable, not the tradeable."""
    strat, risk, brain = StrategyConfig(), RiskConfig(), _brain()
    row = {"atr_pct": 0.02, "atr": 1.0, "mtf_bias": 0.4, "eff_ratio": 0.6,
           "atr_pctile": 0.5}
    assert brain.entry_ok(0.5, 0.7, row, 0.0005, 1.0, 1.0)[0] is True
    assert gate_ev(risk, 2.5, 0.7, row, 0.0005, 1.0, 1.0)[0] is True
    assert gate_mtf_veto(strat, 0.5, row)[0] is True
