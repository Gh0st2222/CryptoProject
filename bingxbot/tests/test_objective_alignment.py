"""The search and the judge must maximize the SAME number.

The search climbed `robust_aggregate` (recency-weighted mean, minus 0.3 sd,
plus 0.2 * worst) over single-symbol training folds. The judge admitted on a
0.7 * median + 0.3 * min composite over purged portfolio folds. Two objectives,
one search: generations of climbing the first walked away from the second.

Measured on 56 candidates scored once and re-aggregated five ways -- same
candidates, same folds, so the comparison is of objectives only:

  training objective                rho vs OOS return   top-10 median OOS
  V0 the old aggregator                   +0.233             +2.72%
  V1 shrink losers by evidence            +0.145             -0.41%
  V2 the JUDGE's own blend                +0.338             +7.72%
  V3 both                                 +0.240             +4.44%
  V4 pooled economics                     +0.270             +4.93%

V2 shipped: `fold_composite` is now the one place either side turns per-fold
fitnesses into a score. These tests exist so the two cannot drift apart again,
and so a change to the blend cannot silently mix score scales on disk.
"""
import json

import pytest

from bingxbot.engine.autotuner import _oos_composite
from bingxbot.engine.search import (OBJECTIVE_VER, DEOptimizer, fold_composite,
                                    robust_aggregate)


def test_the_judge_and_the_search_score_folds_identically():
    """Not "these agree today" -- literally the same function. If someone
    re-implements either side, this fails."""
    for fits in ([1.0], [-2.0, 0.5, 3.0], [0.4, 0.4, 0.4, -9.0],
                 [-1.0, -1.0, -1.0], [5.0, -0.2, 1.1, 0.9, 2.2]):
        assert _oos_composite(fits) == fold_composite(fits) == robust_aggregate(fits)


def test_the_blend_is_median_and_worst():
    """0.7 * median + 0.3 * min. A candidate that is excellent on three folds
    and catastrophic on the fourth must not outrank a steady one."""
    steady = fold_composite([1.0, 1.0, 1.0, 1.0])
    spiky = fold_composite([3.0, 3.0, 3.0, -6.0])
    assert steady > spiky
    assert fold_composite([2.0, 4.0, 6.0]) == pytest.approx(0.7 * 4.0 + 0.3 * 2.0)


def test_weights_are_accepted_and_ignored():
    """`robust_aggregate` kept its weights parameter so old call sites still
    work, but recency weighting is no longer part of the objective. Passing
    weights must not change the answer -- otherwise the two sides diverge
    exactly where one of them happens to pass them."""
    fits = [-1.0, 0.5, 2.0, 0.1]
    assert robust_aggregate(fits, [1.0, 2.0, 3.0, 4.0]) == robust_aggregate(fits)


def test_empty_folds_score_worse_than_any_real_candidate():
    assert fold_composite([]) == -1.0


def test_scores_do_not_survive_an_objective_change(tmp_path):
    """Members carry fitness between cycles while the data window holds. A
    population loaded with scores from a different aggregator would win
    selection against freshly-scored trials on scale alone."""
    path = tmp_path / "tuner_state.json"
    de = DEOptimizer(pop_size=6, seed=1, state_path=path)
    de.seed_population(None)
    de.fitness = [3.0, 2.0, 1.0, 0.0, -1.0, -2.0]
    de.generation = 41
    de.save()

    d = json.loads(path.read_text())
    assert d["objective"] == OBJECTIVE_VER, "the objective version must be recorded"
    d["objective"] = OBJECTIVE_VER - 1
    path.write_text(json.dumps(d))

    back = DEOptimizer(pop_size=6, seed=2, state_path=path)
    assert back.load() is True
    assert back.pop == de.pop, "the GENES are worth keeping -- only scores are stale"
    assert all(f == -1e9 for f in back.fitness), "stale scores must be voided"
    assert back.generation == 41


def test_matching_objective_keeps_its_scores(tmp_path):
    path = tmp_path / "tuner_state.json"
    de = DEOptimizer(pop_size=5, seed=3, state_path=path)
    de.seed_population(None)
    de.fitness = [1.5, 0.5, -0.5, -1.5, -2.5]
    de.save()

    back = DEOptimizer(pop_size=5, seed=4, state_path=path)
    assert back.load() is True
    assert back.fitness == [1.5, 0.5, -0.5, -1.5, -2.5]


def test_state_written_before_versioning_is_treated_as_stale(tmp_path):
    """The file on the live box right now has no `objective` key at all."""
    path = tmp_path / "tuner_state.json"
    de = DEOptimizer(pop_size=4, seed=5, state_path=path)
    de.seed_population(None)
    de.fitness = [9.0, 8.0, 7.0, 6.0]
    de.save()

    d = json.loads(path.read_text())
    del d["objective"]
    path.write_text(json.dumps(d))

    back = DEOptimizer(pop_size=4, seed=6, state_path=path)
    assert back.load() is True
    assert all(f == -1e9 for f in back.fitness)
