"""Meta-labeling layer: barrier labels, walk-forward credentialing, planted-
pattern learning, and the quality-gated brain blend."""
import numpy as np
import pytest

from bingxbot.config import RiskConfig, StrategyConfig
from bingxbot.data.history import synthetic_candles
from bingxbot.ml.meta import (FEATURE_NAMES, MetaModel, build_samples,
                              features_from, train, triple_barrier_label)


def test_triple_barrier_labels():
    o = np.array([100.0] * 10)
    h = np.array([100.0, 100, 103, 100, 100, 100, 100, 100, 100, 100])
    l = np.array([100.0] * 10)
    # long from bar 0 (entry = o[1]): target 100+2*1=102 hit at bar 2 -> win
    assert triple_barrier_label(o, h, l, 0, 1, risk_dist=1.0, rr=2.0) == 1
    # stop wins ties (pessimistic): bar hits both extremes
    h2 = np.array([100.0, 100, 103, 100, 100, 100, 100, 100, 100, 100])
    l2 = np.array([100.0, 100, 98.9, 100, 100, 100, 100, 100, 100, 100])
    assert triple_barrier_label(o, h2, l2, 0, 1, risk_dist=1.0, rr=2.0) == 0
    # nothing hit inside the window -> time barrier -> 0
    flat_h = np.array([100.2] * 10)
    flat_l = np.array([99.8] * 10)
    assert triple_barrier_label(o, flat_h, flat_l, 0, 1, 1.0, 2.0, max_hold=5) == 0
    # not enough future bars -> None
    assert triple_barrier_label(o, h, l, 8, 1, 1.0, 2.0) is None


def test_features_vector_is_stable_and_nan_tolerant():
    x = features_from({}, {}, {}, edge=0.3, regime="TREND_UP")
    assert x.shape == (len(FEATURE_NAMES),)
    assert np.isnan(x[0]), "missing row features must be NaN, not 0 (GBM-native missing)"
    assert x[FEATURE_NAMES.index("regime_TREND_UP")] == 1.0
    assert x[FEATURE_NAMES.index("abs_edge")] == pytest.approx(0.3)


def test_train_learns_a_planted_pattern():
    """Feature 0 strongly predicts the label -> held-out AUC must be well
    above chance and the blend weight positive."""
    rng = np.random.default_rng(3)
    n = 6000
    X = rng.normal(0, 1, (n, len(FEATURE_NAMES))).astype(np.float32)
    p = 1 / (1 + np.exp(-2.2 * X[:, 0]))
    y = (rng.random(n) < p).astype(np.int8)
    mm = train(X, y)
    assert mm is not None
    assert mm.auc > 0.70, f"failed to learn planted pattern (AUC {mm.auc})"
    assert mm.ready and mm.blend_weight > 0.5


def test_train_survives_allnan_columns():
    """Offline training has 9 all-NaN columns BY DESIGN (micro/ctx features
    don't exist in backtest context) — sklearn 1.9's binner crashes on columns
    with zero finite values, which silently killed every live meta training.
    The fit must survive, and the model must still predict on live rows where
    those columns DO carry values."""
    rng = np.random.default_rng(5)
    n = 4000
    X = rng.normal(0, 1, (n, len(FEATURE_NAMES))).astype(np.float32)
    for name in ("micro_obi", "micro_flow", "micro_cvd_slope", "micro_spread_bps",
                 "ctx_funding_rate", "ctx_funding_z", "ctx_oi_change_pct",
                 "ctx_tide_dir", "ctx_tide_er"):
        X[:, FEATURE_NAMES.index(name)] = np.nan
    p = 1 / (1 + np.exp(-2.0 * X[:, 0]))
    y = (rng.random(n) < p).astype(np.int8)
    mm = train(X, y)
    assert mm is not None and mm.auc > 0.65
    live_x = np.nan_to_num(X[0], nan=0.5)   # live rows DO carry micro/ctx values
    assert 0.05 <= mm.predict_one(live_x) <= 0.95


def test_model_roundtrip_and_schema_guard(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (4000, len(FEATURE_NAMES))).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int8)
    mm = train(X, y)
    p = tmp_path / "meta.pkl"
    mm.save(p)
    back = MetaModel.load(p)
    assert back is not None and back.auc == pytest.approx(mm.auc)
    x1 = X[0]
    assert back.predict_one(x1) == pytest.approx(mm.predict_one(x1))


def test_build_samples_produces_labeled_candidates():
    candles = synthetic_candles("BTC-USDT", "5m", 6000, seed=8)
    X, y = build_samples(candles, "5m", StrategyConfig(), RiskConfig())
    assert len(X) == len(y) and len(y) > 50, f"only {len(y)} samples"
    assert X.shape[1] == len(FEATURE_NAMES)
    assert 0.05 < float(y.mean()) < 0.95, "labels must not be degenerate"


def test_build_samples_decoupled_from_live_threshold():
    """The dataset must mean the same thing whatever champion is running:
    labeling through the LIVE config's base_threshold starved training to
    ~1.4k samples ('insufficient data' forever) the moment a conservative
    set (thr 0.30) held the seat."""
    candles = synthetic_candles("BTC-USDT", "5m", 6000, seed=8)
    tight = StrategyConfig()
    tight.base_threshold = 0.5
    loose = StrategyConfig()
    loose.base_threshold = 0.1
    _, y1 = build_samples(candles, "5m", tight, RiskConfig())
    _, y2 = build_samples(candles, "5m", loose, RiskConfig())
    assert len(y1) == len(y2) > 200, "sample count must not depend on the live entry threshold"


def test_brain_blends_meta_when_ready(monkeypatch, tmp_path):
    """With a ready model on disk the brain's P(win) must move toward the
    model's opinion near the gate zone; use_meta=False must bypass it."""
    from bingxbot.ml import meta as meta_mod
    from bingxbot.strategy.brain import TradingBrain

    class _Fake:
        ready = True
        blend_weight = 0.8

        def predict_one(self, x):
            return 0.92

    monkeypatch.setattr(meta_mod, "get_meta", lambda path=None: _Fake())
    # brain imported get_meta by reference at module load — patch there too
    import bingxbot.strategy.brain as brain_mod
    monkeypatch.setattr(brain_mod, "_get_meta", lambda path=None: _Fake())

    row = {"atr_pct": 0.01, "close": 100.0, "atr_pctile": 0.5, "ts": 1_700_000_000_000}
    micro = {"obi": 0.0, "flow": 0.0, "cvd_slope": 0.0, "spread_bps": 1.0, "ticks_per_s": 0.0}
    b = TradingBrain(base_threshold=0.2, threshold_adapt=False)
    from bingxbot.strategy import alphas as alpha_mod
    from bingxbot.strategy.alphas import DESKS
    for nm in DESKS["trend"]:
        monkeypatch.setitem(alpha_mod.ALPHAS, nm, lambda r, m, c: 1.0)
    ev = b.score(row, micro, {})
    assert abs(ev["edge"]) >= 0.2
    assert ev["p_win"] > 0.80, f"meta opinion (0.92 @ w=0.8) must dominate, got {ev['p_win']}"
    b2 = TradingBrain(base_threshold=0.2, threshold_adapt=False)
    b2.use_meta = False
    ev2 = b2.score(row, micro, {})
    assert ev2["p_win"] < ev["p_win"], "use_meta=False must bypass the model"


def test_meta_predict_cached_per_bar(monkeypatch):
    """Live reactive scans re-score the SAME bar several times a second; the
    GBM predict must run once per (bar, direction) — a new bar re-predicts.
    (This is what cut eval_ms back down after the model went live.)"""
    from bingxbot.strategy.brain import TradingBrain
    calls = {"n": 0}

    class _Fake:
        ready = True
        blend_weight = 0.5

        def predict_one(self, x):
            calls["n"] += 1
            return 0.9

    import bingxbot.strategy.brain as brain_mod
    monkeypatch.setattr(brain_mod, "_get_meta", lambda path=None: _Fake())
    from bingxbot.strategy import alphas as alpha_mod
    from bingxbot.strategy.alphas import DESKS
    for nm in DESKS["trend"]:
        monkeypatch.setitem(alpha_mod.ALPHAS, nm, lambda r, m, c: 1.0)
    micro = {"obi": 0.0, "flow": 0.0, "cvd_slope": 0.0, "spread_bps": 1.0, "ticks_per_s": 0.0}
    row = {"atr_pct": 0.01, "close": 100.0, "atr_pctile": 0.5, "ts": 1_700_000_000_000}
    b = TradingBrain(base_threshold=0.2, threshold_adapt=False)
    b.score(row, micro, {})
    b.score(row, micro, {})
    b.score(dict(row), micro, {})            # same ts, fresh dict — still cached
    assert calls["n"] == 1, f"same bar+direction must reuse, got {calls['n']} predicts"
    b.score(dict(row, ts=row["ts"] + 60_000), micro, {})
    assert calls["n"] == 2, "a new bar must re-predict"
