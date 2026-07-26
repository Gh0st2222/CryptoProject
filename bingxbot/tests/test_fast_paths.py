"""The two hot-path rewrites, pinned to the behaviour they replaced.

Both are pure speed changes over code that decides real trades, so the only
acceptable evidence is equivalence, not "looks right":

  * `ml.fasttree` walks the meta model's own trees instead of re-entering
    sklearn once per boosting iteration for a single row. It must agree with
    `predict_proba` on every input shape the brain feeds it — including rows
    that are mostly NaN, which is the NORMAL case in a historical backtest
    (micro and context features have no data there).

  * `strategy.brain._ScoreWindow` replaces `deque(maxlen=720)` so the adaptive
    threshold's quantile needs no copy. It must trim, order and iterate
    exactly like the deque it stands in for.
"""
import math
import random
from collections import deque

import numpy as np
import pytest

from bingxbot.strategy.brain import _ScoreWindow

sklearn = pytest.importorskip("sklearn", reason="meta head is optional")


# ----------------------------------------------------------------- fasttree

def _tiny_model(n_features=12, n=900, seed=3):
    """A real HistGradientBoostingClassifier, shaped like the meta head:
    binary, no categorical features, NaNs present in training and at predict."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    y = (X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    X[rng.random(X.shape) < 0.15] = np.nan          # missing at train time too
    m = HistGradientBoostingClassifier(max_iter=25, max_leaf_nodes=8,
                                       learning_rate=0.1, early_stopping=False,
                                       random_state=seed)
    m.fit(X, y)
    return m


def _probe_rows(n_features, rows=250, seed=11):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(rows, n_features))
    X[rng.random(X.shape) < 0.35] = np.nan
    X[0, :] = np.nan                                 # every feature missing
    X[1, :] = 0.0
    X[2, :] = 1e9                                    # far past every threshold
    X[3, :] = -1e9
    X[4, :] = np.inf
    X[5, :] = -np.inf
    return X


def test_the_compiled_walk_reproduces_sklearn():
    from bingxbot.ml.fasttree import build

    m = _tiny_model()
    ft = build(m)
    assert ft is not None, "a plain binary numeric model must be supported"
    X = _probe_rows(m.n_features_in_)
    ref = m.predict_proba(X)[:, 1]
    for i in range(X.shape[0]):
        assert ft.proba(X[i]) == pytest.approx(float(ref[i]), abs=1e-12, rel=0), \
            f"row {i} disagrees with sklearn"


def test_raw_scores_match_bit_for_bit():
    """The probabilities go through two different sigmoids, so the exact test
    belongs one step earlier: the summed leaf values must be identical."""
    from bingxbot.ml.fasttree import build

    m = _tiny_model(seed=5)
    ft = build(m)
    X = _probe_rows(m.n_features_in_, rows=120, seed=23)
    ref = np.asarray(m._raw_predict(X)).ravel()
    for i in range(X.shape[0]):
        assert ft.raw(X[i]) == float(ref[i]), f"row {i} raw score drifted"


def test_values_landing_exactly_on_a_split_go_the_same_way():
    """Random probes never hit a threshold exactly, so they cannot tell `<=`
    from `<` — and one-hot regime flags and rounded edges are precisely the
    inputs that can. Probe ON the model's own thresholds."""
    from bingxbot.ml.fasttree import build

    m = _tiny_model(seed=17)
    ft = build(m)
    thresholds = []
    for pr in m._predictors:
        nodes = pr[0].nodes
        for nd in nodes:
            if not nd["is_leaf"]:
                thresholds.append((int(nd["feature_idx"]), float(nd["num_threshold"])))
    assert thresholds, "the model has no splits to probe"

    rows = []
    for feat, thr in thresholds[:150]:
        for base in (0.0, 1.0, -1.0):
            r = np.full(m.n_features_in_, base, dtype=np.float64)
            r[feat] = thr                             # exactly on the boundary
            rows.append(r)
    X = np.vstack(rows)
    ref = np.asarray(m._raw_predict(X)).ravel()
    for i in range(X.shape[0]):
        assert ft.raw(X[i]) == float(ref[i]), \
            f"row {i} took a different branch at a split boundary"


def test_a_multiclass_model_falls_back_instead_of_guessing():
    """More than one tree per iteration is a shape this walk does not handle;
    `build` must decline rather than silently read the wrong tree."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    from bingxbot.ml.fasttree import build

    rng = np.random.default_rng(1)
    X = rng.normal(size=(600, 6))
    y = rng.integers(0, 3, size=600)                 # three classes
    m = HistGradientBoostingClassifier(max_iter=10, early_stopping=False,
                                       random_state=1).fit(X, y)
    assert build(m) is None


def test_build_never_raises_on_junk():
    from bingxbot.ml.fasttree import build

    class NotAModel:
        pass

    assert build(NotAModel()) is None
    assert build(None) is None


def test_predict_one_uses_the_fast_path_and_agrees_with_sklearn():
    """End to end through MetaModel, which is what the brain actually calls."""
    from bingxbot.ml.meta import MetaModel

    m = _tiny_model(seed=9)
    mm = MetaModel(m, auc=0.60, n=10_000, trained_ts=0.0,
                   feature_names=[f"f{i}" for i in range(m.n_features_in_)],
                   base_rate=0.4)
    X = _probe_rows(m.n_features_in_, rows=80, seed=31)
    fast = [mm.predict_one(X[i]) for i in range(X.shape[0])]
    assert mm._fast, "the fast path should have been built and kept"
    mm._fast = False                                  # force sklearn
    slow = [mm.predict_one(X[i]) for i in range(X.shape[0])]
    assert fast == pytest.approx(slow, abs=1e-12, rel=0)


def test_predict_one_still_clamps_away_from_certainty():
    from bingxbot.ml.meta import MetaModel

    m = _tiny_model(seed=13)
    mm = MetaModel(m, auc=0.6, n=10_000, trained_ts=0.0,
                   feature_names=[], base_rate=0.4)
    X = _probe_rows(m.n_features_in_, rows=40, seed=7)
    for i in range(X.shape[0]):
        p = mm.predict_one(X[i])
        assert 0.05 <= p <= 0.95 and math.isfinite(p)


# ------------------------------------------------------------- score window

def test_score_window_matches_a_deque_under_fuzzing():
    cap = 32
    win, ref = _ScoreWindow(cap), deque(maxlen=cap)
    rng = random.Random(4)
    for step in range(600):
        if rng.random() < 0.15:
            vals = [rng.uniform(0, 1) for _ in range(rng.randint(0, 40))]
            win.extend(vals)
            ref.extend(vals)
        else:
            v = rng.uniform(0, 1)
            win.append(v)
            ref.append(v)
        assert len(win) == len(ref), f"length drifted at step {step}"
        assert list(win) == list(ref), f"contents drifted at step {step}"
        assert win.view().tolist() == list(ref)
        assert win.tail(10) == list(ref)[-10:]


def test_score_window_survives_far_more_than_its_capacity():
    """The buffer compacts every `cap` appends; the seam must not lose or
    duplicate a value."""
    cap = 16
    win, ref = _ScoreWindow(cap), deque(maxlen=cap)
    for i in range(cap * 50 + 7):
        win.append(float(i))
        ref.append(float(i))
    assert list(win) == list(ref)
    assert win.view().tolist() == list(ref)


def test_score_window_clears_and_refills():
    win = _ScoreWindow(8)
    win.extend([1.0, 2.0, 3.0])
    win.clear()
    assert len(win) == 0 and list(win) == [] and win.view().size == 0
    win.append(9.0)
    assert list(win) == [9.0]


def test_the_quantile_never_reorders_the_stored_window():
    """`_adapt_threshold` partitions the live view. np.partition copies, but if
    that ever changed the brain's own history would be silently shuffled."""
    win = _ScoreWindow(64)
    vals = [float(i % 17) for i in range(64)]
    win.extend(vals)
    np.partition(win.view(), 30)
    assert list(win) == vals
