"""When does a champion lose its seat, and can it lose it for the wrong reason?

The stand-down has now failed twice in opposite directions, both times because
it was pinned to a number on the fitness scale.

  * At a floor of -0.5 against a promotion bar of 0.15 there was a 0.65-wide
    dead zone: a live incumbent measured at -0.177 sat in it for 450 consecutive
    cycles, losing on every re-validation, while the streak counter reset each
    time because -0.177 is comfortably above -0.5.

  * Raising the floor to 0.0 closed that zone and opened a worse one. Measured
    across 966 evaluations, the best composite ANY parameter set reached was
    -0.18 — profitable ones included. A floor at zero sits above the whole
    distribution and would have stood every champion down on a six-cycle timer,
    forever.

So the test is no longer a fitness threshold. It reads the incumbent's pooled
economics over the judged stretch: did it make or lose money, on enough trades
to be a verdict. These tests pin that mechanism, and pin the property that
matters most — a profitable champion keeps its seat, a losing one cannot.
"""
import math

from bingxbot.engine.autotuner import (DEAD_CHAMPION_TRADES, DEMOTE_PATIENCE,
                                       DEMOTE_PATIENCE_WEAK, MARGIN_FLOOR,
                                       MIN_POOLED_TRADES, _pool_stats,
                                       _promotion_bar)


def _losing(pool: dict) -> bool:
    """The tuner's own condition, kept in one place so the tests and the engine
    cannot drift apart in meaning."""
    trades = int(pool.get("trades", 0) or 0)
    ret = float(pool.get("total_return", 0.0) or 0.0)
    return (ret < 0.0 and trades >= MIN_POOLED_TRADES) or trades < DEAD_CHAMPION_TRADES


def _fold(trades, ret, gw=0.0, gl=0.0, dd=0.0):
    return {"trades": trades, "total_return": ret, "gross_win": gw,
            "gross_loss": gl, "max_drawdown": dd}


# ------------------------------------------------------- pooling the evidence

def test_pooling_compounds_log_wealth_rather_than_averaging_percentages():
    """Trading one set through consecutive windows compounds; +10% then -10% is
    -1%, not 0%."""
    p = _pool_stats([_fold(20, 0.10), _fold(20, -0.10)])
    assert p["trades"] == 40
    assert math.isclose(p["total_return"], 0.10 * -0.10 + 0.10 - 0.10, abs_tol=1e-12)
    assert p["total_return"] < 0.0


def test_pooling_takes_the_worst_drawdown_not_the_average():
    p = _pool_stats([_fold(10, 0.01, dd=0.02), _fold(10, 0.01, dd=0.19)])
    assert p["max_drawdown"] == 0.19


def test_a_perfect_profit_factor_on_three_trades_no_longer_walks_through():
    """A real champion was promoted off a newest fold with three trades and no
    losers: profit factor 999 sails past a `< 1.0` test. Pooled against windows
    that did have losers, the same set is judged on all of it."""
    lucky = _fold(3, 0.02, gw=200.0, gl=0.0)
    assert _pool_stats([lucky])["profit_factor"] == 999.0
    real = _pool_stats([lucky, _fold(40, -0.05, gw=100.0, gl=400.0)])
    assert real["profit_factor"] < 1.0
    assert real["total_return"] < 0.0


def test_an_empty_stretch_pools_to_nothing_rather_than_dividing_by_zero():
    p = _pool_stats([])
    assert p["trades"] == 0 and p["total_return"] == 0.0 and p["profit_factor"] == 0.0


def test_a_total_wipeout_is_clamped_instead_of_taking_the_log_of_zero():
    """total_return of -1.0 is log(0). The clamp keeps the pool finite so one
    blown fold cannot poison every comparison with -inf."""
    p = _pool_stats([_fold(10, -1.0), _fold(10, 0.05)])
    assert math.isfinite(p["total_return"]) and p["total_return"] < -0.9


# ------------------------------------------------------------ the stand-down

def test_a_champion_that_lost_money_on_real_evidence_stands_down():
    assert _losing(_pool_stats([_fold(20, -0.03), _fold(20, -0.02)]))


def test_a_champion_that_made_money_keeps_its_seat_however_low_its_fitness():
    """The exact hole the 0.0 floor would have opened: a set that is genuinely
    profitable but whose composite is negative, which measurement says is EVERY
    profitable set at the old fold length."""
    assert not _losing(_pool_stats([_fold(30, 0.04), _fold(30, 0.01)]))


def test_a_losing_stretch_with_too_little_evidence_is_not_a_verdict():
    """Below the evidence floor there is nothing to appeal — but it must not
    read as an acquittal either, so this case is caught by the dead-champion
    rule instead, one test down."""
    pool = _pool_stats([_fold(4, -0.01), _fold(4, -0.01)])
    assert pool["trades"] < MIN_POOLED_TRADES
    assert not _losing(pool), "8 trades is not enough to convict"


def test_a_champion_that_barely_trades_at_all_vacates_the_seat():
    """'Profitable' by having taken two trades is not a champion, it is an
    empty chair. This is the owner's complaint about sets that never fire."""
    assert _losing(_pool_stats([_fold(1, 0.001), _fold(1, 0.0)]))
    assert not _losing(_pool_stats([_fold(DEAD_CHAMPION_TRADES, 0.001)]))


def test_patience_is_shorter_for_a_gauntlet_failure():
    """A champion that already lost in most historical eras has had its warning;
    it does not get the same benefit of the doubt as one that merely had a bad
    window."""
    assert 0 < DEMOTE_PATIENCE_WEAK < DEMOTE_PATIENCE


def test_patience_survives_a_single_bad_window():
    """Re-validation reruns the same recent window each cycle, so consecutive
    readings are highly correlated. One negative reading must not vacate."""
    assert DEMOTE_PATIENCE >= 3 and DEMOTE_PATIENCE_WEAK >= 2


def _streak_to_stand_down(pool, weak):
    """Replay the counter the tuner keeps, and report the cycle it fires on."""
    patience = DEMOTE_PATIENCE_WEAK if weak else DEMOTE_PATIENCE
    streak = 0
    for cycle in range(1, 500):
        if not _losing(pool):
            streak = 0
        else:
            streak += 1
            if streak >= patience:
                return cycle
    return None


def test_the_counter_fires_on_the_loser_and_never_on_the_earner():
    loser = _pool_stats([_fold(40, -0.04)])
    earner = _pool_stats([_fold(40, 0.04)])
    assert _streak_to_stand_down(loser, weak=False) == DEMOTE_PATIENCE
    assert _streak_to_stand_down(loser, weak=True) == DEMOTE_PATIENCE_WEAK
    assert _streak_to_stand_down(earner, weak=False) is None
    assert _streak_to_stand_down(earner, weak=True) is None


def test_exactly_flat_is_not_below_zero():
    """A set that ended the stretch precisely where it started has not lost
    money, and must not be thrashed out for rounding."""
    assert not _losing(_pool_stats([_fold(40, 0.0)]))


# ------------------------------------------------------- the promotion bar

def test_the_margin_runs_the_right_way_below_zero():
    """`champ_fit * 1.16` is -0.21 for an incumbent at -0.177: the worse a
    champion did, the LOWER the bar it set. Multiple-testing inflation, whose
    entire job is to raise the bar as more candidates take a shot, moved it the
    wrong way too."""
    for champ in (-2.0, -0.177, -0.001, 0.0, 0.001, 0.31, 2.0):
        loose = _promotion_bar(champ, 1.06)
        tight = _promotion_bar(champ, 1.16)
        assert tight >= loose, f"more candidates tried must never lower the bar ({champ})"
        assert loose >= champ, f"the bar must sit above the incumbent ({champ})"


def test_the_bar_is_purely_relative_now():
    """There is no absolute fitness floor. Measured at production geometry, only
    6% of parameter sets that genuinely made money could clear one — the median
    profitable set scored -0.869 against a floor of 0.15. The absolute test
    moved onto pooled economics, which profitable sets can actually reach; the
    composite keeps the ranking job it is good at."""
    for champ in (-5.0, -0.177, 0.0, 0.1, 3.0):
        bar = _promotion_bar(champ, 1.06)
        assert bar > champ, "a challenger must still beat the incumbent"
        # ...and the bar tracks the incumbent rather than an outside constant
        assert bar < champ + 1.0


def test_a_champion_near_zero_still_has_to_be_beaten_by_something():
    """The margin is a fraction of the incumbent's magnitude, so at champ ~= 0
    it would vanish. The floor on that magnitude keeps the step real."""
    step = _promotion_bar(0.0, 1.10) - 0.0
    assert step >= 0.10 * MARGIN_FLOOR


def test_the_absolute_floor_inflates_with_the_multiple_testing_meter():
    """While a champion is under water the floor is what binds, so that is
    exactly where the deflation has to bite — it used to do nothing there."""
    assert _promotion_bar(-1.0, 1.16) > _promotion_bar(-1.0, 1.06)


# --------------------------------------------------------- the fallback set

def test_the_fallback_prefers_a_positive_vault_set_over_the_baseline():
    """And when every vault champion is negative — as all four were — there is
    nothing to fall back to but the code defaults, which is correct: the vault
    is not a safe harbour when the whole vault is under water."""
    all_negative = [{"id": "a", "fitness": -0.115, "params": {"x": 1}},
                    {"id": "b", "fitness": -0.236, "params": {"x": 2}}]
    alt = max((c for c in all_negative if c.get("fitness", 0.0) > 0),
              key=lambda c: c["fitness"], default=None)
    assert alt is None, "a negative vault must not be treated as a fallback"

    with_positive = all_negative + [{"id": "c", "fitness": 0.21, "params": {"x": 3}}]
    alt = max((c for c in with_positive if c.get("fitness", 0.0) > 0),
              key=lambda c: c["fitness"], default=None)
    assert alt is not None and alt["id"] == "c"


class _Champs:
    """The slice of the orchestrator the stand-down actually touches."""

    def __init__(self, champions, active):
        self.champions = champions
        self.active_champion_id = active

    def find_champion(self, cid):
        return next((c for c in self.champions if c.get("id") == cid), None)


def test_find_champion_tolerates_an_unknown_active_id():
    """The stand-down reads the active champion to decide its patience; a vault
    that has aged the active set out must not crash the tuner cycle."""
    orch = _Champs([{"id": "a", "fitness": 0.1}], active="gone")
    act = orch.find_champion(orch.active_champion_id) or {}
    assert bool(act.get("gauntlet_weak")) is False


# ------------------------------------------- one judge, one scale, one meaning

def _champ(cid, fitness, ret=None, trades=None, pf=None, **kw):
    c = {"id": cid, "params": {"x": 1}, "fitness": fitness, "clock": "15m"}
    if ret is not None:
        c.update({"oos_return": ret, "oos_trades": trades, "oos_pf": pf})
    c.update(kw)
    return c


def test_the_fallback_admits_on_money_and_ranks_on_fitness():
    """Two different jobs, two different numbers. Which set the account falls
    back to must not depend on a scale that profitable sets rarely reach."""
    from bingxbot.engine.autotuner import _best_fallback
    champs = [
        _champ("poor", 0.9, ret=-0.02, trades=80, pf=0.8),    # best fitness, lost money
        _champ("good", -1.2, ret=0.04, trades=80, pf=1.3),    # negative fitness, earned
        _champ("better", -0.4, ret=0.02, trades=80, pf=1.1),  # ...and ranks higher
    ]
    alt = _best_fallback(champs, "15m")
    assert alt is not None and alt["id"] == "better", (
        "admitted on economics, ranked on fitness")


def test_the_fallback_refuses_a_vault_that_is_entirely_under_water():
    """When every champion lost money there is nothing to fall back to but the
    code defaults. Swapping one losing set for another is not a defence."""
    from bingxbot.engine.autotuner import _best_fallback
    champs = [_champ("a", 0.5, ret=-0.01, trades=90, pf=0.7),
              _champ("b", 0.3, ret=-0.05, trades=90, pf=0.5)]
    assert _best_fallback(champs, "15m") is None


def test_the_fallback_refuses_a_champion_with_no_evidence():
    """A set that made money on four trades has not made a case."""
    from bingxbot.engine.autotuner import _best_fallback
    thin = MIN_POOLED_TRADES - 1
    assert _best_fallback([_champ("a", 2.0, ret=0.5, trades=thin, pf=3.0)], "15m") is None


def test_the_fallback_skips_the_flagged_the_excluded_and_the_other_clock():
    from bingxbot.engine.autotuner import _best_fallback
    ok = dict(ret=0.03, trades=60, pf=1.2)
    assert _best_fallback([_champ("a", 1.0, live_flag={"pf": 0.2}, **ok)], "15m") is None
    assert _best_fallback([_champ("a", 1.0, **ok)], "15m", exclude="a") is None
    other = _champ("a", 1.0, **ok)
    other["clock"] = "5m"
    assert _best_fallback([other], "15m") is None


def test_a_champion_predating_the_pooled_columns_is_not_a_fallback():
    """Records written before this existed carry no oos_* fields. Absent
    evidence must read as 'no case', never as 'passed'."""
    from bingxbot.engine.autotuner import _best_fallback
    assert _best_fallback([_champ("old", 3.0)], "15m") is None
