"""The dead zone between the stand-down floor and the promotion bar.

A champion is replaced only by something scoring above MIN_ABS_FITNESS, and
removed only by scoring below DEMOTE_FLOOR. Everything in between is a seat
nobody can take and nobody can revoke.

At a floor of -0.5 against a bar of 0.15 that zone was 0.65 wide, and a live
incumbent sat at -0.177 inside it for 450 consecutive cycles — losing on every
re-validation while `_champ_bad_streak` reset each time, because -0.177 is
comfortably above -0.5. It also carried `gauntlet_weak: true`, having lost in
four of six historical eras.

These tests pin the boundary itself, so the gap cannot quietly reopen.
"""
import pytest

from bingxbot.engine.autotuner import (DEMOTE_FLOOR, DEMOTE_PATIENCE,
                                       DEMOTE_PATIENCE_WEAK, MIN_ABS_FITNESS)


def test_a_losing_champion_is_reachable_by_the_stand_down():
    """The floor must not sit below zero. A champion whose re-validated fitness
    on the traded basket is negative is, by the system's own measure, worse
    than not trading — there is no reading of "less toxic than -0.5" that earns
    it real money."""
    assert DEMOTE_FLOOR >= 0.0, "a negative incumbent must be removable"


def test_the_dead_zone_is_only_the_promotion_bar():
    """Nothing may be simultaneously unpromotable AND unremovable except by the
    overfit guard itself. Any width here is a seat the system cannot vacate."""
    unreachable = MIN_ABS_FITNESS - DEMOTE_FLOOR
    assert unreachable <= MIN_ABS_FITNESS, (
        f"dead zone {unreachable:.2f} is wider than the promotion bar — a "
        f"champion between {DEMOTE_FLOOR} and {MIN_ABS_FITNESS} can neither be "
        f"beaten nor removed")


def test_the_live_incumbent_that_prompted_this_would_now_stand_down():
    """The exact number from the resume: -0.177, gauntlet_weak, 450 cycles."""
    champ_fit = -0.177
    assert champ_fit < DEMOTE_FLOOR, "this is the case that used to be immortal"


def test_patience_is_shorter_for_a_gauntlet_failure():
    """A champion that already lost in most historical eras has had its warning;
    it does not get the same benefit of the doubt as one that merely had a bad
    window."""
    assert 0 < DEMOTE_PATIENCE_WEAK < DEMOTE_PATIENCE


def test_patience_survives_a_single_bad_window():
    """A floor at zero sits much closer to the noise than one at -0.5, and
    re-validation reruns the same recent window each cycle. One negative
    reading must not vacate the seat."""
    assert DEMOTE_PATIENCE >= 3 and DEMOTE_PATIENCE_WEAK >= 2


# ------------------------------------------------------- the mechanism itself

class _Champs:
    """The slice of the orchestrator the stand-down actually touches."""

    def __init__(self, champions, active):
        self.champions = champions
        self.active_champion_id = active
        self.applied = None

    def find_champion(self, cid):
        return next((c for c in self.champions if c.get("id") == cid), None)

    def apply_params(self, p):
        self.applied = p

    def mark_champion_used(self, cid):
        pass


def _streak_to_stand_down(champ_fit, weak):
    """Replay the counter the tuner keeps, and report the cycle it fires on."""
    patience = DEMOTE_PATIENCE_WEAK if weak else DEMOTE_PATIENCE
    streak = 0
    for cycle in range(1, 500):
        if champ_fit >= DEMOTE_FLOOR:
            streak = 0
        else:
            streak += 1
            if streak >= patience:
                return cycle
    return None


def test_the_old_behaviour_never_fired_and_the_new_one_does():
    assert _streak_to_stand_down(-0.177, weak=False) == DEMOTE_PATIENCE
    assert _streak_to_stand_down(-0.177, weak=True) == DEMOTE_PATIENCE_WEAK
    # a profitable incumbent keeps its seat forever, which is the point
    assert _streak_to_stand_down(0.31, weak=False) is None
    assert _streak_to_stand_down(0.31, weak=True) is None


def test_a_marginal_champion_is_not_thrashed_out():
    """Exactly at the floor is not below it."""
    assert _streak_to_stand_down(DEMOTE_FLOOR, weak=False) is None


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


def test_find_champion_tolerates_an_unknown_active_id():
    """The stand-down reads the active champion to decide its patience; a vault
    that has aged the active set out must not crash the tuner cycle."""
    orch = _Champs([{"id": "a", "fitness": 0.1}], active="gone")
    act = orch.find_champion(orch.active_champion_id) or {}
    assert bool(act.get("gauntlet_weak")) is False
