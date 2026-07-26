"""How much evidence stands behind a promotion.

Measured on 181 parameter sets scored across purged portfolio folds, leaving
each fold out in turn as the unseen future:

  traded bars   median trades   folds under the   best composite   lift of the
  per fold      per fold        t<5 placeholder   anyone reaches   promoted set
  600                8               32%              -0.39          +0.01 pp
  960               14               14%              +0.86          +3.59 pp

Same judge, same statistic, same +0.15 bar. At the shorter length the judge's
output never reached its own promotion bar — not once in 966 evaluations — and
carried no usable ranking signal. At roughly double the length it cleared the
bar and picked sets that beat the field by ~3.6 points of return on a window
they had never been scored on, positive in all eight splits.

A 90-day lookback split four ways gave production 864 traded bars per judged
fold — between the two rows, and the live board settles which side: every
champion in the vault negative and 618 cycles without a promotion. These tests
pin the geometry so it cannot slide back there, and pin the pooled evidence
floor that replaced a single fold's trade count.
"""
from bingxbot.engine.autotuner import (LOOKBACK_DAYS, MIN_BARS,
                                       MIN_OOS_TRADED_BARS, MIN_POOLED_TRADES,
                                       MIN_VETO_TRADES, OOS_FOLDS,
                                       OOS_TAIL_FRAC, _oos_fold_count)
from bingxbot.engine.search import portfolio_folds

WARMUP = 300
BARS_PER_DAY_15M = 96


def _bars(days, interval_min=15):
    return int(days * 24 * 60 / interval_min)


# --------------------------------------------------------- the geometry itself

def test_the_default_lookback_gives_judged_folds_a_length_that_works():
    """The whole finding in one assertion: at the configured lookback, splitting
    the out-of-sample tail into OOS_FOLDS must leave each fold well clear of the
    length where the judge stopped meaning anything."""
    n = _bars(LOOKBACK_DAYS)
    traded = int(n * OOS_TAIL_FRAC) // OOS_FOLDS
    assert traded >= MIN_OOS_TRADED_BARS, (
        f"{LOOKBACK_DAYS}d split {OOS_FOLDS} ways gives {traded} traded bars per "
        f"fold; below {MIN_OOS_TRADED_BARS} the judge's verdict stops "
        f"correlating with the next window")


def test_the_old_ninety_day_geometry_is_the_one_that_failed():
    """Kept as the negative control: if someone lowers the lookback back to 90
    days this test says out loud what that costs."""
    traded = int(_bars(90) * OOS_TAIL_FRAC) // OOS_FOLDS
    assert traded < MIN_OOS_TRADED_BARS, (
        f"90d gave {traded} traded bars per fold — the geometry the live board "
        f"could not promote out of")


def test_thin_history_widens_the_folds_instead_of_slicing_confetti():
    """Below the length threshold the fold COUNT gives way to the fold LENGTH.
    Four 600-bar folds is a verdict about nothing; two 1200-bar folds is a
    verdict."""
    short = {"BTC-USDT": list(range(_bars(60)))}
    k = _oos_fold_count(short)
    assert 1 <= k <= OOS_FOLDS
    traded = int(len(short["BTC-USDT"]) * OOS_TAIL_FRAC) // k
    assert traded >= MIN_OOS_TRADED_BARS or k == 1


def test_plenty_of_history_still_uses_the_full_fold_count():
    """Widening folds is a concession to thin data, not the new normal — the
    median+worst composite needs several folds to have a median at all."""
    assert _oos_fold_count({"BTC-USDT": list(range(_bars(LOOKBACK_DAYS)))}) == OOS_FOLDS


def test_the_fold_count_never_returns_zero_or_negative():
    for days in (0.5, 2, 10, 30, 90, 180, 365):
        assert _oos_fold_count({"X": list(range(_bars(days)))}) >= 1
    assert _oos_fold_count({}) >= 1


def test_the_fold_count_is_monotone_in_history():
    """More data must never produce fewer judged folds."""
    ks = [_oos_fold_count({"X": list(range(_bars(d)))}) for d in (10, 30, 60, 90, 180, 365)]
    assert ks == sorted(ks), ks


def test_the_chosen_count_survives_contact_with_portfolio_folds():
    """_oos_fold_count and portfolio_folds must agree, or the tuner asks for a
    geometry it does not get. portfolio_folds silently drops undersized folds,
    which is exactly how four folds could quietly become two."""
    for days in (45, 90, 180):
        cbs = {"A": list(range(_bars(days))), "B": list(range(_bars(days)))}
        k = _oos_fold_count(cbs)
        folds = portfolio_folds(cbs, k=k, tail_frac=OOS_TAIL_FRAC, warmup=WARMUP)
        assert len(folds) == k, f"{days}d: asked for {k} folds, got {len(folds)}"


def test_the_shortest_basket_decides_the_geometry():
    """A basket is judged on its common grid, so one short symbol governs how
    long the folds can be — taking the longest would overstate the evidence."""
    mixed = {"long": list(range(_bars(180))), "short": list(range(_bars(40)))}
    assert _oos_fold_count(mixed) == _oos_fold_count({"short": list(range(_bars(40)))})


def test_the_data_floor_admits_the_geometry_it_asks_for():
    """MIN_BARS gates whether a symbol joins the judged basket at all. It must
    not admit a symbol too short to produce even one judgeable fold."""
    assert int(MIN_BARS * OOS_TAIL_FRAC) + WARMUP >= MIN_VETO_TRADES * 60


# ------------------------------------------------------- the evidence floor

def test_the_promotion_evidence_floor_is_pooled_not_per_fold():
    """Requiring five trades in ONE 6-day window rejected 49% of candidates on
    the coin flip of whether that particular window happened to fire. The floor
    that decides a promotion now counts the whole judged stretch."""
    assert MIN_POOLED_TRADES > MIN_VETO_TRADES
    assert MIN_POOLED_TRADES >= 30, (
        "measured: below ~15 trades a verdict was ANTI-correlated with the next "
        "window (r=-0.38); 30-60 is where it turned informative (r=+0.49)")


def test_the_pooled_floor_is_reachable_at_the_configured_geometry():
    """A floor nothing can clear is the same bug as a bar nothing can clear,
    one level down. At ~14 trades per judged fold, OOS_FOLDS folds must be able
    to carry the pooled floor with room to spare."""
    measured_trades_per_fold = 14      # median, 1260-bar folds, two symbols
    assert OOS_FOLDS * measured_trades_per_fold >= MIN_POOLED_TRADES
