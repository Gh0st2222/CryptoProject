"""A champion that died in 2022 is not a bad champion. It is a champion that
should be flat when the tape looks like 2022.

The regime gauntlet used to spend real CPU scoring a parameter set across five
years of market personalities and then throw almost all of it away: one bit,
`weak`, which raised the promotion bar. A trend set that bled through a crash
was filed as slightly worse everywhere, including in the trending markets it was
built for — and a set that survived a crash by never trading got no credit for
the distinction.

Now each era's score is filed under that era's CHARACTER, classified by the same
`detect_regime` the live brain runs on every bar, and the profile becomes an
operating manual: refuse new entries while the live tape looks like a market
this set has actually lost money in.

The dangerous failure here is a gate that fires when it should not — a champion
that stands down forever trades nothing, which looks exactly like the frozen
board this project already spent 618 cycles inside. So most of these tests are
about SILENCE: no profile, no era, a merely-mediocre era, a malformed record,
all have to leave the engine trading.
"""
import pytest

from bingxbot.strategy.regime import REGIMES, RANGE, TREND_DOWN, TREND_UP, VOLATILE
from bingxbot.strategy.regime_profile import (DOMINANT_SHARE, LOSING_ERA_FIT,
                                              build_profile, classify_era,
                                              dominant_regime, stands_down)


def _era(fit, regime, name="era", pf=1.0):
    return {"fit": fit, "pf": pf, "regime": regime, "name": name}


# --------------------------------------------------------- building a profile

def test_an_era_score_is_filed_under_the_market_it_happened_in():
    prof = build_profile({"2022 crash": _era(-3.1, TREND_DOWN, "2022 crash"),
                          "2024 bull": _era(+1.2, TREND_UP, "2024 bull")})
    assert prof[TREND_DOWN]["losing"] is True
    assert prof[TREND_UP]["losing"] is False
    assert prof[TREND_DOWN]["names"] == ["2022 crash"]


def test_several_eras_of_one_character_are_combined_by_median():
    """One catastrophic era among three good ones is an anecdote, not a rule —
    the median says so where a mean would not."""
    prof = build_profile({"a": _era(+1.0, RANGE), "b": _era(+0.8, RANGE),
                          "c": _era(-9.0, RANGE)})
    assert prof[RANGE]["eras"] == 3
    assert prof[RANGE]["fit"] == 0.8
    assert prof[RANGE]["losing"] is False


def test_a_merely_mediocre_era_does_not_ground_the_champion():
    """Underperforming is not bleeding. The gate exists for markets that cost
    this set real money, not ones where it made less than usual."""
    prof = build_profile({"a": _era(LOSING_ERA_FIT + 0.01, VOLATILE)})
    assert prof[VOLATILE]["losing"] is False
    assert not stands_down(prof, VOLATILE)


def test_an_era_with_no_classification_is_dropped_not_guessed():
    prof = build_profile({"mixed": _era(-9.0, None), "clear": _era(-9.0, RANGE)})
    assert set(prof) == {RANGE}


def test_an_unknown_regime_label_cannot_smuggle_itself_in():
    prof = build_profile({"x": _era(-9.0, "SIDEWAYS_ISH")})
    assert prof == {}


# ------------------------------------------------------------ the live gate

def test_the_gate_fires_only_on_the_regime_that_lost_money():
    prof = build_profile({"2022 crash": _era(-3.1, TREND_DOWN),
                          "2024 bull": _era(+1.2, TREND_UP)})
    assert stands_down(prof, TREND_DOWN)
    for other in (TREND_UP, RANGE, VOLATILE):
        assert not stands_down(prof, other), other


def test_no_profile_means_trade_normally():
    """Every champion promoted before this existed, every offline run, and every
    first cycle has no profile. None of them may be grounded by its absence."""
    for empty in (None, {}, {"windows": {}}):
        for reg in REGIMES:
            assert not stands_down(empty, reg)


def test_a_regime_the_gauntlet_never_saw_is_not_a_conviction():
    """Six eras cannot cover four regimes evenly. Missing evidence has to read
    as 'no verdict', never as 'guilty'."""
    prof = build_profile({"2022 crash": _era(-3.1, TREND_DOWN)})
    assert set(prof) == {TREND_DOWN}
    for reg in (TREND_UP, RANGE, VOLATILE):
        assert not stands_down(prof, reg)


def test_a_malformed_profile_cannot_ground_the_engine():
    """This dict round-trips through champion JSON on disk, so it can come back
    as anything after an edit, a partial write or an older schema."""
    for junk in ({TREND_DOWN: None}, {TREND_DOWN: {}}, {TREND_DOWN: {"fit": -9.0}},
                 {TREND_DOWN: {"losing": None}}, {TREND_DOWN: {"losing": 0}}):
        assert not stands_down(junk, TREND_DOWN), junk
    assert stands_down({TREND_DOWN: {"losing": True}}, TREND_DOWN)


def test_an_empty_or_missing_regime_string_is_not_a_lookup():
    prof = build_profile({"a": _era(-3.0, TREND_DOWN)})
    assert not stands_down(prof, "")
    assert not stands_down(prof, None)


def test_a_champion_cannot_be_grounded_in_every_regime_at_once_by_accident():
    """The nightmare: a set whose profile says 'losing' everywhere trades
    nothing at all. That must require losing money in EVERY classified regime —
    which is a real verdict, not a glitch — and the gauntlet's own `weak` flag
    would have flagged it long before."""
    all_bad = build_profile({r: _era(-5.0, r, r) for r in REGIMES})
    assert all(stands_down(all_bad, r) for r in REGIMES)
    # ...and the ground lifts as soon as the evidence for a regime stops
    # pointing one way: two profitable ranges against one bad one is a set that
    # trades ranges, whatever the worst of them looked like.
    mixed = build_profile({**{r: _era(-5.0, r, r) for r in REGIMES},
                           "good1": _era(+2.0, RANGE, "good1"),
                           "good2": _era(+1.4, RANGE, "good2")})
    assert mixed[RANGE]["eras"] == 3
    assert not stands_down(mixed, RANGE), "a majority of earning eras must lift the ground"
    assert stands_down(mixed, TREND_DOWN), "...without lifting it anywhere else"


# ---------------------------------------------------- classifying real bars

def test_classification_uses_the_live_classifier_over_real_bars():
    from bingxbot.data.history import synthetic_candles
    shares = classify_era(synthetic_candles("BTC-USDT", "15m", 1500, seed=3), "15m")
    assert set(shares) == set(REGIMES)
    assert sum(shares.values()) == pytest.approx(1.0, abs=1e-9)
    assert all(v >= 0.0 for v in shares.values())


def test_too_few_bars_classifies_as_nothing_rather_than_as_RANGE():
    """Under the warmup every feature is non-finite and detect_regime answers
    RANGE at zero confidence. Counting those would label every short era the
    same way and quietly ground champions in ranges."""
    from bingxbot.data.history import synthetic_candles
    shares = classify_era(synthetic_candles("BTC-USDT", "15m", 200, seed=3), "15m")
    assert sum(shares.values()) == 0.0
    assert dominant_regime(shares) is None
    assert classify_era([], "15m") == {r: 0.0 for r in REGIMES}


def test_a_mixed_era_refuses_to_name_itself():
    even = {r: 1.0 / len(REGIMES) for r in REGIMES}
    assert 1.0 / len(REGIMES) < DOMINANT_SHARE
    assert dominant_regime(even) is None
    lopsided = dict(even)
    lopsided[TREND_UP] = DOMINANT_SHARE + 0.01
    assert dominant_regime(lopsided) == TREND_UP


def test_dominant_regime_tolerates_an_empty_share_map():
    assert dominant_regime({}) is None
    assert dominant_regime({r: 0.0 for r in REGIMES}) is None


# ------------------------------------------------- wired into the live engine

def _engine(tmp_path):
    from bingxbot.config import BotConfig
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
    return TraderEngine(cfg, feed, PaperBroker(pf, feed.states, {}, 5e-4, 0.0), pf,
                        RiskManager(cfg.risk), {}, journal=TradeJournal(tmp_path / "j.jsonl"))


def _xray(eng, regime):
    import types
    ctx = eng.ctx["BTC-USDT"]
    st = eng.feed.states["BTC-USDT"]
    st.spread_bps = types.SimpleNamespace(get=lambda d=1.0: 2.0)
    row = {"atr_pct": 0.004, "atr": 100.0, "mtf_bias": 0.5, "mtf_align": 0.5,
           "eff_ratio": 0.6, "bb_pctb": 0.5, "funding_rate": 0.0, "ts": 1}
    eng._build_gates(ctx, st, row,
                     {"edge": 0.6, "p_win": 0.62, "regime": regime, "threshold": 0.3})
    return {g["n"]: g for g in ctx.gates}


def test_a_fresh_engine_has_no_profile_and_shows_no_history_row(tmp_path):
    eng = _engine(tmp_path)
    assert eng.champion_regime_profile is None
    assert "champion history" not in _xray(eng, TREND_DOWN)


def test_the_xray_row_agrees_with_the_gate_that_refuses(tmp_path):
    """A dashboard that disagrees with the engine about why nothing is trading
    is worse than no dashboard — the exact bug this X-ray was built after."""
    eng = _engine(tmp_path)
    eng.champion_regime_profile = build_profile(
        {"2022 crash": _era(-3.1, TREND_DOWN, "2022 crash"),
         "2024 bull": _era(+1.2, TREND_UP, "2024 bull")})
    down = _xray(eng, TREND_DOWN)["champion history"]
    assert down["ok"] is False and "standing down" in down["d"]
    up = _xray(eng, TREND_UP)["champion history"]
    assert up["ok"] is True and "standing down" not in up["d"]
    # ...and the row is absent where the gauntlet has nothing to say
    assert "champion history" not in _xray(eng, VOLATILE)


def test_the_profile_follows_the_champion_the_orchestrator_activates():
    """The gauntlet writes the profile onto the champion record; activating that
    champion has to carry it to the engine, or the gate is dead code."""
    import types
    prof = build_profile({"2022 crash": _era(-3.1, TREND_DOWN, "2022 crash")})
    champ = {"id": "c1", "params": {}, "gauntlet_weak": True,
             "gauntlet": {"by_regime": prof}}
    eng = types.SimpleNamespace(active_champion_id=None, champion_gauntlet_weak=False,
                                champion_regime_profile=None)

    from bingxbot.server.orchestrator import Orchestrator
    orch = types.SimpleNamespace(champions=[champ], engine=eng, active_champion_id=None,
                                 find_champion=lambda cid: champ if cid == "c1" else None,
                                 save_champions=lambda: None)
    Orchestrator.mark_champion_used(orch, "c1")
    assert eng.active_champion_id == "c1"
    assert eng.champion_gauntlet_weak is True
    assert eng.champion_regime_profile == prof
    assert stands_down(eng.champion_regime_profile, TREND_DOWN)


def test_a_champion_without_a_gauntlet_clears_a_stale_profile():
    """Swapping to a champion that was never gauntleted must not leave the
    PREVIOUS champion's stand-downs in force — that would ground a set for a
    history that is not its own."""
    import types
    from bingxbot.server.orchestrator import Orchestrator
    champ = {"id": "c2", "params": {}}
    eng = types.SimpleNamespace(active_champion_id="c1", champion_gauntlet_weak=True,
                                champion_regime_profile={TREND_DOWN: {"losing": True}})
    orch = types.SimpleNamespace(champions=[champ], engine=eng, active_champion_id=None,
                                 find_champion=lambda cid: champ if cid == "c2" else None,
                                 save_champions=lambda: None)
    Orchestrator.mark_champion_used(orch, "c2")
    assert eng.champion_regime_profile is None
    assert eng.champion_gauntlet_weak is False
