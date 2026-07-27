"""A champion that died in 2022 is not a bad champion. It is a champion that
should be flat when the tape looks like 2022.

The regime gauntlet used to spend real CPU scoring a parameter set across five
years of market personalities and then throw almost all of it away: one bit,
`weak`, which raised the promotion bar. A trend set that bled through a crash
was filed as slightly worse everywhere, including in the trending markets it was
built for.

The first attempt at using that evidence labelled each ERA with one dominant
regime and took a median of era scores. Live data killed it: of six eras, "2023
chop" and "2025 range" named no regime at all — a mixed market has no majority —
so four of six eras contributed nothing, and the two that survived were medians
of two eras that cancelled each other out. The markets a champion is most likely
to bleed in were exactly the ones producing no verdict.

The profile is now built from the champion's OWN TRADES: each trade carries the
regime it was opened into, and they pool across all six eras. Hundreds of fills
per regime instead of six numbers, and no era is ever discarded for being mixed.

The dangerous failure here is a gate that fires when it should not — a champion
that stands down forever trades nothing, which looks exactly like the frozen
board this project already spent 618 cycles inside. So most of these tests are
about SILENCE: no profile, too few trades, a flat patch, a malformed record, all
have to leave the engine trading.
"""
import pytest

from bingxbot.strategy.regime import REGIMES, RANGE, TREND_DOWN, TREND_UP, VOLATILE
from bingxbot.strategy.regime_profile import (LOSING_PF, MIN_REGIME_TRADES,
                                              build_profile, classify_era,
                                              dominant_regime, stands_down)


def _era(name, **buckets):
    """One gauntlet era, as its per-regime trade ledger.

    buckets: regime -> (trades, gross_win, gross_loss)
    """
    by = {reg: {"trades": t, "gross_win": gw, "gross_loss": gl, "pnl": gw - gl}
          for reg, (t, gw, gl) in buckets.items()}
    return {"name": name, "fit": 0.0, "pf": 1.0,
            "trades": sum(b["trades"] for b in by.values()), "by_regime": by}


def _bled(n=40):
    """A bucket that clearly lost money on enough trades to convict."""
    return (n, 100.0, 400.0)


def _earned(n=40):
    return (n, 400.0, 100.0)


# --------------------------------------------------------- building a profile

def test_trades_pool_across_every_era_instead_of_a_median_of_era_scores():
    """The whole point of the rewrite: no era is discarded, and a regime's
    record is the sum of what actually happened in it."""
    prof = build_profile({
        "2022 crash": _era("2022 crash", TREND_DOWN=_bled(30)),
        "2023 chop": _era("2023 chop", TREND_DOWN=_bled(25), RANGE=_earned(30)),
        "2024 bull": _era("2024 bull", TREND_UP=_earned(50)),
    })
    assert prof[TREND_DOWN]["trades"] == 55, "both eras must contribute"
    assert prof[TREND_DOWN]["eras"] == 2
    assert prof[TREND_DOWN]["losing"] is True
    assert prof[RANGE]["losing"] is False and prof[TREND_UP]["losing"] is False


def test_a_mixed_era_is_no_longer_dropped():
    """'2023 chop' and '2025 range' named no dominant regime and contributed
    nothing at all. Now every trade they contain is counted, in whichever
    market it was taken."""
    prof = build_profile({"mixed": _era("mixed", RANGE=_bled(30), VOLATILE=_earned(30))})
    assert set(prof) == {RANGE, VOLATILE}
    assert prof[RANGE]["losing"] is True
    assert prof[VOLATILE]["losing"] is False


def test_too_few_trades_is_an_anecdote_not_a_conviction():
    """A handful of bad fills in one market must not ground a champion there."""
    few = MIN_REGIME_TRADES - 1
    prof = build_profile({"e": _era("e", TREND_DOWN=(few, 10.0, 900.0))})
    assert prof[TREND_DOWN]["trades"] == few
    assert prof[TREND_DOWN]["losing"] is False
    assert not stands_down(prof, TREND_DOWN)


def test_a_flat_patch_is_not_a_loss():
    """Under-performance is not bleeding. The gate exists for markets that cost
    this set real money."""
    prof = build_profile({"e": _era("e", VOLATILE=(40, 100.0, 100.0 / LOSING_PF - 1.0))})
    assert prof[VOLATILE]["losing"] is False
    assert not stands_down(prof, VOLATILE)


def test_an_unknown_bucket_cannot_smuggle_itself_in():
    """Trades opened before this field existed carry 'UNKNOWN', and a future
    build could add a regime this one has never heard of."""
    prof = build_profile({"e": _era("e", UNKNOWN=_bled(90), SIDEWAYS_ISH=_bled(90))})
    assert prof == {}


def test_an_era_with_no_ledger_at_all_is_survivable():
    """Cached gauntlet entries written by the previous build have no by_regime."""
    prof = build_profile({"old": {"name": "old", "fit": -3.0, "pf": 0.5}})
    assert prof == {}


# ------------------------------------------------------------ the live gate

def test_the_gate_fires_only_on_the_regime_that_lost_money():
    prof = build_profile({"2022 crash": _era("2022 crash", TREND_DOWN=_bled()),
                          "2024 bull": _era("2024 bull", TREND_UP=_earned())})
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
    prof = build_profile({"2022 crash": _era("2022 crash", TREND_DOWN=_bled())})
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
    prof = build_profile({"a": _era("a", TREND_DOWN=_bled())})
    assert not stands_down(prof, "")
    assert not stands_down(prof, None)


def test_a_champion_cannot_be_grounded_in_every_regime_at_once_by_accident():
    """The nightmare: a set whose profile says 'losing' everywhere trades
    nothing at all. That must require losing money in EVERY classified regime —
    which is a real verdict, not a glitch — and the gauntlet's own `weak` flag
    would have flagged it long before."""
    all_bad = build_profile({"e": _era("e", **{r: _bled() for r in REGIMES})})
    assert all(stands_down(all_bad, r) for r in REGIMES)
    # ...and the ground lifts as soon as the POOLED result for a regime turns
    # positive: profitable range trading in later eras outweighs one bad one,
    # because the trades add up rather than being medianed away.
    mixed = build_profile({"bad": _era("bad", **{r: _bled() for r in REGIMES}),
                           "good1": _era("good1", RANGE=_earned(60)),
                           "good2": _era("good2", RANGE=_earned(60))})
    assert mixed[RANGE]["eras"] == 3 and mixed[RANGE]["trades"] == 160
    assert not stands_down(mixed, RANGE), "pooled profit must lift the ground"
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


def test_the_era_character_label_is_context_only():
    """classify_era/dominant_regime still annotate the vault so a human can read
    "2022 crash was TREND_DOWN". They decide nothing — that job cost four of six
    eras their vote."""
    from bingxbot.strategy.regime_profile import DOMINANT_SHARE
    even = {r: 1.0 / len(REGIMES) for r in REGIMES}
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
        {"2022 crash": _era("2022 crash", TREND_DOWN=_bled()),
         "2024 bull": _era("2024 bull", TREND_UP=_earned())})
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
    prof = build_profile({"2022 crash": _era("2022 crash", TREND_DOWN=_bled())})
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
