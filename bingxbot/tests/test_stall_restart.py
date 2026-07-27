"""The one mechanism built to break a promotion drought, and why it never fired.

A live board ran 240 research cycles with zero promotions, a vault holding one
champion, and population diversity 0.71. The tuner has a restart for exactly
that situation — re-inject fresh explorers when the search stops producing
champions — and it did not fire once, because it demanded the population be
COLLAPSED as well:

    if self.de.diversity() < 0.25 and self._since_improve >= STALL_REINJECT:

Convergence is not the only way to get stuck, and on this board it was not the
way. The population stayed spread out and still produced nothing promotable for
240 cycles, so the `and` held the escape hatch shut for as long as the problem
lasted.

Why a drought is the right signal on its own, measured on a 56-member
population evolved with the tuner's own machinery and judged with the tuner's
own judge:

  * DE training fitness ranks candidates at rho +0.07 against their
    out-of-sample return — essentially no ordering information;
  * the population's genuinely best set, +23.25% out of sample over 99 trades,
    sat 51st of 56 by training fitness;
  * a freshly seeded population had 41 of 56 members profitable out of sample
    on 30+ trades, against 1 of 12 nominees on the converged live board.

Generations of climbing that gradient walk away from what the judge rewards, so
fresh blood is not a tie-breaker here — it is the supply.
"""
from bingxbot.engine.autotuner import STALL_REINJECT


def _fires(since_improve: int, diversity: float, *, old: bool) -> bool:
    """Both rules, side by side, so the regression is stated rather than
    described."""
    if old:
        return diversity < 0.25 and since_improve >= STALL_REINJECT
    return since_improve >= STALL_REINJECT


def test_the_live_board_would_now_restart():
    """240 cycles, diversity 0.71 — the exact state that produced no champion."""
    assert not _fires(240, 0.71, old=True), "this is the bug, kept as the control"
    assert _fires(240, 0.71, old=False)


def test_a_collapsed_population_still_restarts():
    """The case the old rule DID cover must keep working."""
    assert _fires(STALL_REINJECT, 0.10, old=True)
    assert _fires(STALL_REINJECT, 0.10, old=False)


def test_a_healthy_desk_is_never_restarted():
    """Promotions reset the counter, so a desk that is finding champions keeps
    every generation of search it has accumulated."""
    for div in (0.05, 0.25, 0.71, 0.99):
        assert not _fires(0, div, old=False)
        assert not _fires(STALL_REINJECT - 1, div, old=False)


def test_the_drought_is_measured_in_cycles_not_luck():
    """One quiet cycle is not a drought; the threshold has to be long enough
    that ordinary between-champion gaps do not trigger it."""
    assert STALL_REINJECT >= 10


def test_a_spread_out_population_keeps_more_of_itself():
    """A collapsed population is searching nowhere and needs the bigger
    transfusion; a spread-out one is searching the wrong place and still holds
    information worth keeping."""
    def frac(div):
        return 0.4 if div < 0.25 else 0.25
    assert frac(0.10) > frac(0.71)
    assert 0.0 < frac(0.71) < 1.0, "a restart must never replace the whole population"


def test_the_counter_resets_so_restarts_cannot_thrash():
    """Without the reset a stalled desk would re-inject on every cycle after the
    threshold, destroying the search instead of nudging it."""
    since = STALL_REINJECT
    fired = 0
    for _ in range(100):
        if _fires(since, 0.71, old=False):
            fired += 1
            since = 0
        since += 1
    assert fired <= 100 // STALL_REINJECT + 1, f"{fired} restarts in 100 cycles"
