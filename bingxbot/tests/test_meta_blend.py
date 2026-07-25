"""P(win) scale integrity: the meta head is a SECOND OPINION, not a second
scale. Regression tests for the defect that sat the machine flat for days."""
import math
import types

import pytest

from bingxbot.strategy.brain import META_MAX_SHIFT, TradingBrain, _logit, _sigmoid


class _StubMeta:
    """Stands in for a credentialed MetaModel."""

    def __init__(self, p, base_rate=0.25, weight=0.85):
        self.p = p
        self.base_rate = base_rate
        self._w = weight
        self.ready = True

    @property
    def blend_weight(self):
        return self._w

    def predict_one(self, _x):
        return self.p


def _brain_with_meta(monkeypatch, meta, cal_p=0.52):
    """A brain whose calibrator is pinned at cal_p and whose meta head is stubbed,
    so the test isolates the BLEND from everything upstream of it."""
    import bingxbot.strategy.brain as bm
    b = TradingBrain(base_threshold=0.12, threshold_adapt=False)
    b.threshold = 0.12
    monkeypatch.setattr(bm, "_get_meta", lambda: meta)
    monkeypatch.setattr(bm, "_meta_features", lambda *a, **k: None)
    monkeypatch.setattr(b.calibrator, "predict", lambda *a, **k: cal_p)
    return b


def _row():
    return {"close": 100.0, "atr": 1.0, "atr_pct": 0.01, "atr_pctile": 0.5,
            "ts": 1_700_000_000_000, "mtf_align": 0.5, "mtf_bias": 0.5,
            "eff_ratio": 0.5, "adx": 25.0, "bb_pctb": 0.5}


def _edge_scores(direction=1.0):
    """Alpha scores that fuse to a solid edge past 0.75*threshold, so the meta
    head is actually consulted."""
    from bingxbot.strategy.alphas import ALPHAS
    return {nm: 0.5 * direction for nm in ALPHAS}


def test_meta_at_its_base_rate_abstains(monkeypatch):
    """A model predicting exactly its training base rate is saying 'average
    candidate — no information'. It must leave P(win) untouched; the old linear
    blend instead dragged 0.52 down to 0.29 for saying nothing at all."""
    meta = _StubMeta(p=0.25, base_rate=0.25)
    b = _brain_with_meta(monkeypatch, meta, cal_p=0.52)
    ev = b.score(_row(), {}, {}, alpha_scores=_edge_scores())
    assert ev["p_win"] == pytest.approx(0.52, abs=1e-6)
    old_linear = 0.85 * 0.25 + 0.15 * 0.52
    assert old_linear < 0.30, "documents the defect the fix removes"


def test_bearish_meta_lowers_odds_without_importing_its_scale(monkeypatch):
    """The live numbers from the 2026-07-25 resume: calibrator ~0.52, meta 0.067
    against a 0.2478 base rate. The old blend produced P 14% — below every gate,
    on a scale min_p_win could never reach. The fix must still LOWER the
    probability (the model is genuinely bearish) but keep it on the decision's
    own scale."""
    meta = _StubMeta(p=0.0673, base_rate=0.2478)
    b = _brain_with_meta(monkeypatch, meta, cal_p=0.52)
    p = b.score(_row(), {}, {}, alpha_scores=_edge_scores())["p_win"]
    old_linear = 0.85 * 0.0673 + 0.15 * 0.52
    assert old_linear < 0.14, "the defect: gate-proof number"
    assert p < 0.52, "a bearish model must still reduce conviction"
    assert p > old_linear + 0.05, "but not by importing the labeler's base rate"


def test_bullish_meta_raises_odds(monkeypatch):
    meta = _StubMeta(p=0.60, base_rate=0.25)
    b = _brain_with_meta(monkeypatch, meta, cal_p=0.52)
    p = b.score(_row(), {}, {}, alpha_scores=_edge_scores())["p_win"]
    assert p > 0.52, "a model well above its base rate is evidence FOR the trade"
    assert p <= 0.95


def test_shift_is_bounded_and_skill_weighted(monkeypatch):
    """No single confident prediction may gate the account shut, and a model
    with weak credentials moves the odds proportionally less."""
    certain = _StubMeta(p=0.001, base_rate=0.25, weight=0.85)
    b = _brain_with_meta(monkeypatch, certain, cal_p=0.52)
    p_hi = b.score(_row(), {}, {}, alpha_scores=_edge_scores())["p_win"]
    floor = _sigmoid(_logit(0.52) - 0.85 * META_MAX_SHIFT)
    assert p_hi == pytest.approx(floor, abs=1e-6), "clamped at META_MAX_SHIFT"

    weak = _StubMeta(p=0.001, base_rate=0.25, weight=0.20)
    b2 = _brain_with_meta(monkeypatch, weak, cal_p=0.52)
    p_lo = b2.score(_row(), {}, {}, alpha_scores=_edge_scores())["p_win"]
    assert p_lo > p_hi, "less measured skill -> a smaller move"


def test_no_meta_leaves_the_calibrator_alone(monkeypatch):
    """Without a model the brain must produce exactly the calibrator's number —
    the property that made the defect visible (p_win jumped between ~0.52 and
    ~0.15 depending only on whether the meta was consulted)."""
    import bingxbot.strategy.brain as bm
    b = TradingBrain(base_threshold=0.12, threshold_adapt=False)
    monkeypatch.setattr(bm, "_get_meta", lambda: None)
    monkeypatch.setattr(b.calibrator, "predict", lambda *a, **k: 0.52)
    ev = b.score(_row(), {}, {}, alpha_scores=_edge_scores())
    assert ev["p_win"] == pytest.approx(0.52, abs=1e-9)
    assert b.meta_p is None


def test_labeler_barrier_is_a_fixed_constant():
    """The labeler's profit barrier must not follow a tuner-owned parameter:
    every champion swap would otherwise redefine what a 'win' is and move the
    dataset's base rate under the model."""
    import inspect

    from bingxbot.ml import meta as mm
    src = inspect.getsource(mm.build_samples)
    assert "LABEL_RR" in src
    assert "expected_rr" not in src, "the label must not depend on live config"
    assert mm.SCHEMA_VER >= 3, "a changed label definition must force a retrain"


def test_model_carries_its_base_rate(tmp_path):
    """base_rate must survive the pickle round-trip — the blend is meaningless
    without it (and a legacy model without one must degrade to 0.5, not to the
    old scale-mixing behaviour)."""
    from bingxbot.ml.meta import FEATURE_NAMES, MetaModel
    m = MetaModel(model=types.SimpleNamespace(), auc=0.61, n=9000, trained_ts=1.0,
                  feature_names=list(FEATURE_NAMES), base_rate=0.2478)
    assert m.base_rate == pytest.approx(0.2478)
    for bad in (0.0, 1.0, -1.0, float("nan")):
        assert MetaModel(model=None, auc=0.6, n=1, trained_ts=0.0,
                         feature_names=[], base_rate=bad).base_rate == 0.5


def test_logit_sigmoid_roundtrip():
    for p in (0.05, 0.2478, 0.5, 0.77, 0.95):
        assert _sigmoid(_logit(p)) == pytest.approx(p, abs=1e-9)
    assert math.isfinite(_logit(0.0)) and math.isfinite(_logit(1.0))
