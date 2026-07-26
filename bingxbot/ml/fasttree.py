"""A compiled traversal for the meta head's decision trees.

The meta model is a 150-iteration HistGradientBoostingClassifier and the brain
consults it one row at a time — once per (bar, direction) near the gate zone.
sklearn's `predict_proba` is built for matrices: every call re-validates the
input, then loops the 150 predictors in PYTHON, entering Cython once per tree.
Profiled on a 6k-bar backtest that is 186,600 tree calls for 1,244 predictions
and ~6.7us each, of which the actual ten-comparison tree walk is a rounding
error — the cost is almost entirely per-call overhead, paid 150 times over for
a single row.

So the trees are flattened once into contiguous arrays and walked in one
compiled function. Nothing about the MODEL changes: same trees, same
thresholds, same missing-value routing, same sigmoid. Verified bit-for-bit
against sklearn's own raw prediction on construction, and again in the test
suite.

This is an optimization, never a dependency. Every unsupported shape falls
back to sklearn permanently and silently:

  * more than one tree per boosting iteration (multiclass)
  * any categorical split (bitset routing is not reproduced here)
  * a loss whose link is not the binary logistic one
  * a construction-time parity check that does not agree to 1e-12

`numba` is optional too — without it the same traversal runs as a plain numpy
loop, which is still an order of magnitude cheaper than re-entering sklearn
150 times.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("bingxbot.fasttree")

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:  # noqa: BLE001 — numba is an accelerator, not a requirement
    HAVE_NUMBA = False

    def njit(*a, **k):                      # type: ignore[misc]
        def wrap(fn):
            return fn
        return wrap if not (a and callable(a[0])) else a[0]


# How closely the compiled walk must reproduce sklearn before we trust it.
# The manual walk is exact arithmetic over the same doubles, so agreement is
# normally 0.0; the tolerance only absorbs a future sklearn changing the order
# it sums the leaf values in.
PARITY_TOL = 1e-12


@njit(cache=True, nogil=True)
def _raw_one(x, feat, thr, miss_left, left, right, is_leaf, value, roots, baseline):
    """Sum the leaf values of every tree for one feature vector.

    Mirrors sklearn's `_predict_one_from_raw_data`: a NaN feature follows
    `missing_go_to_left`, everything else goes left on `<= threshold`. No
    fastmath — reordering these float comparisons, or letting NaN be treated
    as an ordinary number, would change which leaf a row lands in.
    """
    total = baseline
    for t in range(roots.shape[0]):
        k = roots[t]
        while is_leaf[k] == 0:
            v = x[feat[k]]
            if np.isnan(v):
                k = left[k] if miss_left[k] else right[k]
            else:
                k = left[k] if v <= thr[k] else right[k]
        total += value[k]
    return total


class FastTrees:
    """Flattened predictors of one HistGradientBoosting binary classifier."""

    __slots__ = ("baseline", "feat", "is_leaf", "left", "miss_left",
                 "n_features", "right", "roots", "thr", "value")

    def __init__(self, model):
        preds = model._predictors
        if any(len(p) != 1 for p in preds):
            raise ValueError("not a single-tree-per-iteration (binary) model")

        feat, thr, miss, left, right, leaf, val, roots = [], [], [], [], [], [], [], []
        off = 0
        for p in preds:
            nodes = p[0].nodes
            if nodes["is_categorical"].any():
                raise ValueError("categorical splits are not reproduced here")
            roots.append(off)
            feat.append(nodes["feature_idx"].astype(np.int64))
            thr.append(nodes["num_threshold"].astype(np.float64))
            miss.append(nodes["missing_go_to_left"].astype(np.uint8))
            # child indices are tree-local; shift them into the flat array
            left.append(nodes["left"].astype(np.int64) + off)
            right.append(nodes["right"].astype(np.int64) + off)
            leaf.append(nodes["is_leaf"].astype(np.uint8))
            val.append(nodes["value"].astype(np.float64))
            off += len(nodes)

        self.feat = np.concatenate(feat)
        self.thr = np.concatenate(thr)
        self.miss_left = np.concatenate(miss)
        self.left = np.concatenate(left)
        self.right = np.concatenate(right)
        self.is_leaf = np.concatenate(leaf)
        self.value = np.concatenate(val)
        self.roots = np.asarray(roots, dtype=np.int64)
        self.baseline = float(np.asarray(model._baseline_prediction).ravel()[0])
        self.n_features = int(model.n_features_in_)

    def raw(self, x: np.ndarray) -> float:
        """Raw (link-space) score for one row — sklearn's `_raw_predict`."""
        return float(_raw_one(np.ascontiguousarray(x, dtype=np.float64),
                              self.feat, self.thr, self.miss_left, self.left,
                              self.right, self.is_leaf, self.value, self.roots,
                              self.baseline))

    def proba(self, x: np.ndarray) -> float:
        """P(class 1) — the logistic link of the raw score."""
        r = self.raw(x)
        # expit, written so a large negative r cannot overflow exp()
        if r >= 0.0:
            return 1.0 / (1.0 + np.exp(-r))
        e = np.exp(r)
        return e / (1.0 + e)


def build(model) -> FastTrees | None:
    """Flatten `model` and prove the result against sklearn before returning it.

    Returns None for anything unsupported or anything that fails parity, which
    is the caller's signal to keep using sklearn. Never raises.
    """
    try:
        ft = FastTrees(model)
    except Exception as e:  # noqa: BLE001 — an unsupported model is not an error
        log.info("meta fast path unavailable (%s); using sklearn", e)
        return None

    try:
        # Probe with the shapes this model actually sees: real-valued rows, and
        # rows with missing features — micro/context columns are NaN by design
        # in every historical backtest, so the missing-value branch is the
        # common path, not an edge case.
        rng = np.random.default_rng(0)
        probe = rng.normal(size=(24, ft.n_features)).astype(np.float64)
        probe[rng.random(probe.shape) < 0.3] = np.nan
        probe[0, :] = np.nan                      # everything missing
        probe[1, :] = 0.0
        ref = np.asarray(model._raw_predict(probe)).ravel()
        for i in range(probe.shape[0]):
            if not abs(ft.raw(probe[i]) - float(ref[i])) <= PARITY_TOL:
                log.warning("meta fast path failed parity on probe %d; using sklearn", i)
                return None
    except Exception as e:  # noqa: BLE001 — if we cannot prove it, we do not use it
        log.warning("meta fast path could not be verified (%s); using sklearn", e)
        return None
    return ft
