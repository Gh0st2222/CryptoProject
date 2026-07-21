"""Meta-labeling: a learned second opinion on the linear brain's entry candidates.

The desk/alpha machine stays exactly as it is — interpretable, online — and
keeps PROPOSING trades. This layer learns which of its candidates actually pay,
which is the one thing a linear fusion structurally cannot represent
(conditional structure: "momentum works when funding is flat and BTC's tide
agrees").

Method (the boring, documented standard for this problem):

- **Triple-barrier labels.** A candidate bar (|fused edge| past the gate zone)
  is labeled by WHICH BARRIER the price path hits first from the next open:
  the profit barrier (expected_rr x an ATR risk unit), the stop barrier
  (1 risk unit, pessimistic on ties), or the time barrier. That makes the
  training target the same event the account actually gets paid on — not
  "did price drift up".
- **A small gradient-boosted tree** (HistGradientBoosting: NaN-native, fast,
  heavily regularized) maps the full feature row -> P(win). Trained strictly
  walk-forward: fit on the older 75%, scored by AUC on the newest 25% with a
  purge gap of one full barrier horizon so overlapping labels cannot leak
  across the split.
- **Quality-gated blending.** The model only influences trading when its
  held-out AUC clears a floor, and its weight vs the online calibrator scales
  with that AUC — a model with no measured skill has no vote. No model file,
  no sklearn, or a stale schema => the brain runs exactly as before.

The same model file is loaded by the live engine AND by backtest workers
(mtime-cached), so validation and trading always gate with the same brain.
"""
from __future__ import annotations

import logging
import math
import pickle
import time

import numpy as np

from ..config import ROOT

log = logging.getLogger("meta")

MODEL_PATH = ROOT / "data_cache" / "meta_model.pkl"
SCHEMA_VER = 1
MIN_AUC = 0.53          # below this the model has no measured skill -> no vote
MIN_SAMPLES = 3000
CAND_EDGE_FRAC = 0.8    # candidate = |edge| >= this x base_threshold (the gate zone)
BARRIER_RISK_ATR = 1.8  # risk unit for labeling, mid of the sl_atr search box
MAX_HOLD_BARS = 96      # time barrier for labeling

# Feature vector — one list, used by BOTH the dataset builder and live scoring,
# so train and trade can never disagree about what a column means. Missing
# values stay NaN (HistGradientBoosting handles them natively): backtests have
# no book/tape/tide, live does — the model learns to use them when present.
ROW_FEATURES = (
    "atr_pct", "atr_pctile", "vol_of_vol", "rsi_14", "rsi_7", "stoch_k", "adx",
    "bb_pctb", "bb_width_pctile", "dc_pos", "eff_ratio", "ema21_slope",
    "ema55_slope", "macd_hist", "macd_line", "roc_3", "roc_12", "roc_accel",
    "vwap_dev", "vwap_z", "vol_z", "mtf_align", "mtf_bias", "squeeze_on",
    "linreg_slope", "ret_1",
    # 24h range context: position within the day's high/low, ATR-distance to
    # each extreme, deviation from the 24h volume-weighted average
    "range_pos_24h", "dist_hi_24h", "dist_lo_24h", "vwap24_dev",
)
MICRO_FEATURES = ("obi", "flow", "cvd_slope", "spread_bps")
CTX_FEATURES = ("funding_rate", "funding_z", "oi_change_pct", "tide_dir", "tide_er")
REGIMES = ("TREND_UP", "TREND_DOWN", "RANGE", "VOLATILE")

FEATURE_NAMES = (list(ROW_FEATURES) + [f"micro_{m}" for m in MICRO_FEATURES]
                 + [f"ctx_{c}" for c in CTX_FEATURES]
                 + ["edge", "abs_edge", "hour_sin", "hour_cos"]
                 + [f"regime_{r}" for r in REGIMES])


def _f(v) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def features_from(row: dict, micro: dict, ctx: dict, edge: float, regime: str) -> np.ndarray:
    """The single feature-vector builder shared by training and live scoring."""
    out = [_f(row.get(k)) for k in ROW_FEATURES]
    out += [_f((micro or {}).get(k)) for k in MICRO_FEATURES]
    out += [_f((ctx or {}).get(k)) for k in CTX_FEATURES]
    out.append(_f(edge))
    out.append(abs(_f(edge)))
    ts = row.get("ts", 0.0)
    try:
        hour = (float(ts) / 3_600_000.0) % 24.0
    except (TypeError, ValueError):
        hour = 0.0
    out.append(math.sin(2 * math.pi * hour / 24.0))
    out.append(math.cos(2 * math.pi * hour / 24.0))
    out += [1.0 if regime == r else 0.0 for r in REGIMES]
    return np.asarray(out, dtype=np.float32)


def triple_barrier_label(o: np.ndarray, h: np.ndarray, l: np.ndarray, i: int,
                         direction: int, risk_dist: float, rr: float,
                         max_hold: int = MAX_HOLD_BARS) -> int | None:
    """Label the candidate at bar i (entry = next bar's open): 1 if the profit
    barrier (rr x risk_dist) is hit before the stop barrier (1 x risk_dist),
    else 0. Ties inside one bar resolve to the STOP (pessimistic, matching the
    backtester). The time barrier labels 0. None = not enough future bars."""
    n = len(o)
    if i + 2 >= n or risk_dist <= 0:
        return None
    entry = o[i + 1]
    if entry <= 0:
        return None
    stop = entry - direction * risk_dist
    target = entry + direction * rr * risk_dist
    last = min(i + 1 + max_hold, n - 1)
    for j in range(i + 1, last + 1):
        hit_stop = l[j] <= stop if direction > 0 else h[j] >= stop
        hit_tgt = h[j] >= target if direction > 0 else l[j] <= target
        if hit_stop:            # stop wins ties — pessimistic
            return 0
        if hit_tgt:
            return 1
    return 0                    # time barrier: never showed the profit


def build_samples(candles: list, interval: str, strat, risk) -> tuple[np.ndarray, np.ndarray]:
    """Run the PRIMARY signal (a default-parameter brain — a fixed, deterministic
    candidate rule) over history and emit (features, label) for every bar where
    the fused edge enters the gate zone. Offline micro/ctx stay NaN — exactly
    what backtest-context scoring sees."""
    from ..engine.backtest import NO_MICRO, candles_to_arrays
    from ..strategy.brain import TradingBrain
    from ..strategy.features import FeatureFrame
    from ..util import interval_ms

    if len(candles) < 700:
        return np.empty((0, len(FEATURE_NAMES)), np.float32), np.empty(0, np.int8)
    arrays = candles_to_arrays(candles)
    ff = FeatureFrame(arrays, interval=interval)
    o, h, l = arrays["open"], arrays["high"], arrays["low"]
    bph = 3_600_000 / interval_ms(interval)
    brain = TradingBrain(horizon_bars=strat.horizon_bars, base_threshold=strat.base_threshold,
                         threshold_adapt=True, target_trades_per_hour=strat.target_trades_per_hour,
                         bars_per_hour=bph)
    brain.use_meta = False      # the labeler must never consult the model it feeds
    X, y = [], []
    for i in range(300, ff.n):
        row = ff.row(i)
        ev = brain.evaluate(row, NO_MICRO, {})
        edge = ev["edge"]
        if abs(edge) < CAND_EDGE_FRAC * brain.threshold:
            continue
        lab = triple_barrier_label(o, h, l, i, 1 if edge > 0 else -1,
                                   BARRIER_RISK_ATR * row.get("atr", 0.0), risk.expected_rr)
        if lab is None:
            continue
        X.append(features_from(row, NO_MICRO, {}, edge, ev["regime"]))
        y.append(lab)
    if not X:
        return np.empty((0, len(FEATURE_NAMES)), np.float32), np.empty(0, np.int8)
    return np.vstack(X), np.asarray(y, dtype=np.int8)


class MetaModel:
    """Persisted GBM + its credentials (held-out AUC, sample count)."""

    def __init__(self, model, auc: float, n: int, trained_ts: float,
                 feature_names: list[str]):
        self.model = model
        self.auc = auc
        self.n = n
        self.trained_ts = trained_ts
        self.feature_names = feature_names

    @property
    def ready(self) -> bool:
        return self.auc >= MIN_AUC and self.n >= MIN_SAMPLES

    @property
    def blend_weight(self) -> float:
        """The model's vote scales with MEASURED held-out skill: AUC 0.53 -> ~0.2,
        0.58 -> ~0.7, capped at 0.85 so the online calibrator always keeps a say."""
        return float(min(max((self.auc - 0.50) * 10.0, 0.0), 0.85))

    def predict_one(self, x: np.ndarray) -> float:
        p = self.model.predict_proba(x.reshape(1, -1))[0, 1]
        return float(min(max(p, 0.05), 0.95))

    def save(self, path=MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump({"schema": SCHEMA_VER, "model": self.model, "auc": self.auc,
                         "n": self.n, "trained_ts": self.trained_ts,
                         "feature_names": self.feature_names}, f)
        tmp.replace(path)

    @staticmethod
    def load(path=MODEL_PATH) -> "MetaModel | None":
        try:
            with path.open("rb") as f:
                d = pickle.load(f)
        except (OSError, pickle.PickleError, EOFError):
            return None
        if d.get("schema") != SCHEMA_VER or d.get("feature_names") != FEATURE_NAMES:
            return None         # feature vector changed between builds -> retrain
        return MetaModel(d["model"], float(d["auc"]), int(d["n"]),
                         float(d.get("trained_ts", 0.0)), d["feature_names"])


def _fill_allnan_cols(X: np.ndarray) -> np.ndarray:
    """sklearn's HGB binner (1.9) crashes on a column with ZERO finite values
    (sliding_window_view over an empty distinct-value array). Offline training
    legitimately has such columns — micro/ctx features are NaN by design in
    backtest context — so fill them with a constant at FIT time only. A
    no-information column stays no-information (the tree never splits on a
    constant), and the live feature schema is untouched."""
    allnan = ~np.isfinite(X).any(axis=0)
    if allnan.any():
        X = X.copy()
        X[:, allnan] = 0.0
    return X


def train(X: np.ndarray, y: np.ndarray) -> MetaModel | None:
    """Walk-forward fit: train on the older 75%, credential on the newest 25%
    with a purge gap of one barrier horizon (overlapping labels can't leak)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    n = len(y)
    if n < MIN_SAMPLES // 2:
        return None
    cut = int(n * 0.75)
    tr_end = max(0, cut - MAX_HOLD_BARS)          # purge: labels look forward
    Xtr, ytr, Xva, yva = X[:tr_end], y[:tr_end], X[cut:], y[cut:]
    if len(ytr) < 500 or len(yva) < 200 or len(set(ytr.tolist())) < 2 or len(set(yva.tolist())) < 2:
        return None
    m = HistGradientBoostingClassifier(
        max_iter=150, max_leaf_nodes=15, learning_rate=0.08,
        l2_regularization=1.0, min_samples_leaf=60,
        early_stopping=False, random_state=7)
    m.fit(_fill_allnan_cols(Xtr), ytr)
    auc = float(roc_auc_score(yva, m.predict_proba(Xva)[:, 1]))
    # refit on everything so live uses all the data; the credential stays the
    # honest walk-forward number, never the refit's in-sample flattery.
    m.fit(_fill_allnan_cols(X), y)
    return MetaModel(m, auc=auc, n=n, trained_ts=time.time(), feature_names=list(FEATURE_NAMES))


def train_from_candles(candles_by_symbol: dict, interval: str, strat, risk,
                       path=MODEL_PATH) -> dict:
    """One training pass over the basket: build samples per symbol, pool them
    chronologically-per-symbol (each symbol's own bars stay ordered; the split
    purge handles the rest), train, and persist only if the model is at least
    as credentialed as the incumbent. Module-level + picklable so it runs in a
    research-pool worker."""
    Xs, ys = [], []
    for sym, candles in candles_by_symbol.items():
        X, y = build_samples(candles, interval, strat, risk)
        if len(y):
            Xs.append(X)
            ys.append(y)
    if not Xs:
        return {"trained": False, "reason": "no samples"}
    X, y = np.vstack(Xs), np.concatenate(ys)
    mm = train(X, y)
    if mm is None:
        return {"trained": False, "reason": f"insufficient data ({len(y)} samples)"}
    old = MetaModel.load(path)
    if old is not None and old.ready and mm.auc < old.auc - 0.02:
        return {"trained": False, "reason": f"new AUC {mm.auc:.3f} worse than incumbent {old.auc:.3f}",
                "auc": round(mm.auc, 4), "n": int(len(y))}
    mm.save(path)
    return {"trained": True, "auc": round(mm.auc, 4), "n": int(len(y)),
            "ready": mm.ready, "weight": round(mm.blend_weight, 3),
            "base_rate": round(float(y.mean()), 4)}


# ------------------------------------------------- mtime-cached process loader

_cache: dict = {"mtime": None, "model": None}


def get_meta(path=MODEL_PATH) -> MetaModel | None:
    """Cheap per-process loader: reloads only when the file changes, so live
    brains and backtest workers pick up a fresh model without restarts."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        _cache["mtime"], _cache["model"] = None, None
        return None
    if _cache["mtime"] != mtime:
        _cache["model"] = MetaModel.load(path)
        _cache["mtime"] = mtime
    return _cache["model"]
