"""TradingBrain — the whole firm in one object, per symbol.

Pipeline every closed bar:

    features ─► 18 alphas ─► 5 desks (within-desk Hedge weights)
             ─► meta-allocator (CIO weights desks by live performance)
             ─► regime gate ─► fused directional edge
             ─► probability calibrator ─► P(win)
             ─► adaptive threshold + cost gate + Kelly sizing hint

Everything is graded `horizon` bars later against the ATR-normalized realized
return: alphas update their within-desk Hedge weights, desks update the
allocator, and the outcome trains the calibrator. Three nested adaptive loops,
all online, all observable — no parameter is hand-set at runtime.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from ..util import clamp
from .alphas import ALPHA_META, ALPHAS, DESK_ORDER, DESKS, EVENT, MICRO_ALPHAS
from .allocator import MetaAllocator
from .calibration import ProbabilityCalibrator
from .regime import REGIME_DESK_MULT, detect_regime

try:  # meta-labeling layer (optional: no sklearn / no trained model => no-op)
    from ..ml.meta import features_from as _meta_features
    from ..ml.meta import get_meta as _get_meta
except Exception:  # noqa: BLE001 — the brain must run without the ML stack
    _get_meta = None
    _meta_features = None


@dataclass
class AlphaStat:
    calls: int = 0
    hits: int = 0
    payoff_sum: float = 0.0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.calls if self.calls else 0.0


@dataclass
class _Pending:
    idx: int
    close: float
    atr: float
    atr_pctile: float
    regime: str
    alpha_scores: dict
    desk_sig: dict
    edge: float


class TradingBrain:
    def __init__(self, eta=0.35, weight_floor=0.05, horizon_bars=5,
                 base_threshold=0.30, threshold_adapt=True,
                 target_trades_per_hour=1.5, bars_per_hour=12.0,
                 cost_multiple=1.4, min_p_win=0.50, kelly_fraction=0.30):
        self.eta = eta
        self.floor = weight_floor
        self.horizon = max(1, horizon_bars)
        self.base_threshold = base_threshold
        self.threshold_adapt = threshold_adapt
        self.target_rate = max(0.1, target_trades_per_hour)
        self.bars_per_hour = bars_per_hour
        self.cost_multiple = cost_multiple
        self.min_p_win = min_p_win
        self.kelly_fraction = kelly_fraction

        # within-desk Hedge weights (each desk's alphas sum to 1)
        self.alpha_w: dict[str, float] = {}
        for names in DESKS.values():
            for nm in names:
                self.alpha_w[nm] = 1.0 / len(names)
        self.alpha_stats = {nm: AlphaStat() for nm in ALPHAS}
        self.allocator = MetaAllocator(DESK_ORDER, floor=0.05)
        self.calibrator = ProbabilityCalibrator()

        self._pending: deque[_Pending] = deque()
        self._score_hist: deque[float] = deque(maxlen=720)
        self._idx = 0
        self.use_meta = True       # the ML dataset builder disables this on its
        self.meta_p = None         # own brain so the model never labels itself
        self._meta_key = None      # (bar_ts, direction) of the cached prediction
        self._meta_cached = None
        self.beta = 1.0
        self.threshold = base_threshold
        self.graded = 0
        self.last: dict = {}       # last decision snapshot
        self._sc: dict | None = None   # components of the most recent score()

    # ------------------------------------------------------------- evaluate

    def score(self, row: dict, micro: dict, ctx: dict | None = None,
              alpha_scores: dict | None = None) -> dict:
        """Pure decision pass: features -> alphas -> desks -> fused edge -> P(win).
        No learning state is touched, so this can run **as often as we like**
        between bar closes (on every tick / order-book update) to react to live
        price, flow and the multi-timeframe context without corrupting the online
        weights. `alpha_scores` may be supplied precomputed (they depend on the
        data, never on this brain's parameters — the backtester computes them
        once per fold and reuses them for every tuner candidate). Returns the
        decision snapshot (also stored on ``self.last``)."""
        ctx = ctx or {}
        regime, conf = detect_regime(row)

        if alpha_scores is None:
            alpha_scores = {nm: float(fn(row, micro, ctx)) for nm, fn in ALPHAS.items()}
        desk_sig, desk_conf, desk_active = {}, {}, {}
        for desk, names in DESKS.items():
            num = den = 0.0
            active = same = 0
            for nm in names:
                s = alpha_scores[nm]
                w = self.alpha_w[nm]
                num += w * s
                den += w
                if abs(s) > 0.05:
                    active += 1
            sig = clamp(num / den if den > 0 else 0.0, -1, 1)
            desk_sig[desk] = sig
            desk_active[desk] = active
            if active:
                same = sum(1 for nm in names if abs(alpha_scores[nm]) > 0.05 and alpha_scores[nm] * sig > 0)
                desk_conf[desk] = abs(sig) * (same / active)
            else:
                desk_conf[desk] = 0.0

        alloc = self.allocator.weights()
        rmult = REGIME_DESK_MULT[regime]
        num = den = 0.0
        for desk in DESK_ORDER:
            # a desk with NO speaking alphas has no opinion — it must not sit in
            # the denominator diluting the fused edge toward zero. Backtests run
            # with the micro + carry desks dormant (no book/tape/funding data),
            # which used to shrink every backtest edge ~40% vs live and break
            # threshold comparability between validation and trading. A desk
            # whose alphas speak but cancel is different: that IS an opinion
            # ("flat") and it stays in the average.
            if desk_active[desk] == 0:
                continue
            w = alloc[desk] * rmult.get(desk, 1.0)
            num += w * desk_sig[desk]
            den += w
        edge = clamp(num / den if den > 0 else 0.0, -1, 1)
        atr_pctile = row.get("atr_pctile", 0.5)
        p_win = self.calibrator.predict(edge, regime, atr_pctile)
        # meta-labeling second opinion: a walk-forward-credentialed GBM over the
        # FULL feature row (interactions the linear fusion can't represent).
        # Consulted only near the gate zone (where P(win) can change a decision
        # — keeps backtests fast), weighted by its measured held-out AUC, and
        # silently absent when there's no trained model. Same model file loads
        # in live brains and backtest workers, so validation and trading gate
        # with the same head.
        self.meta_p = None
        if (self.use_meta and _get_meta is not None
                and abs(edge) >= 0.75 * self.threshold):
            try:
                m = _get_meta()
                if m is not None and m.ready:
                    # one GBM predict per (bar, direction): live reactive scans
                    # re-score the SAME bar several times a second, and the
                    # single-row predict is the expensive part of the whole
                    # eval. Backtests step ts every call, so they never reuse —
                    # parity between modes is untouched.
                    key = (row.get("ts"), 1 if edge > 0 else -1)
                    if key == self._meta_key and self._meta_cached is not None:
                        self.meta_p = self._meta_cached
                    else:
                        self.meta_p = m.predict_one(_meta_features(row, micro, ctx, edge, regime))
                        self._meta_key, self._meta_cached = key, self.meta_p
                    w = m.blend_weight
                    p_win = clamp(w * self.meta_p + (1 - w) * p_win, 0.05, 0.95)
            except Exception:  # noqa: BLE001 — a broken model must never block trading
                pass

        # stash exactly what observe() needs to grade this bar later
        self._sc = {"alpha_scores": alpha_scores, "desk_sig": desk_sig,
                    "edge": edge, "atr_pctile": atr_pctile, "regime": regime}

        expected_move = self.beta * abs(edge) * row.get("atr_pct", 0.0) * math.sqrt(self.horizon)
        self.last = {
            "edge": edge, "p_win": p_win, "regime": regime, "regime_conf": conf,
            "threshold": self.threshold, "beta": self.beta,
            "alpha_scores": alpha_scores, "desk_sig": desk_sig, "desk_conf": desk_conf,
            "alloc": alloc, "expected_move": expected_move,
            "mtf_align": row.get("mtf_align", 0.0), "mtf_bias": row.get("mtf_bias", 0.0),
        }
        return self.last

    def observe(self, row: dict) -> None:
        """Learning step for a **closed** bar: grade matured predictions, record
        this bar's prediction for future grading, advance the clock and adapt the
        threshold. Call exactly once per closed bar — never on a reactive score."""
        sc = self._sc
        if sc is None:
            return
        self._grade(row)
        self._pending.append(_Pending(
            idx=self._idx, close=row.get("close", 0.0), atr=max(row.get("atr", 0.0), 1e-12),
            atr_pctile=sc["atr_pctile"], regime=sc["regime"], alpha_scores=dict(sc["alpha_scores"]),
            desk_sig=dict(sc["desk_sig"]), edge=sc["edge"],
        ))
        self._score_hist.append(abs(sc["edge"]))
        self._idx += 1
        self._adapt_threshold()

    def evaluate(self, row: dict, micro: dict, ctx: dict | None = None,
                 alpha_scores: dict | None = None) -> dict:
        """Score **and** learn from one closed bar (backtest + the live bar-close
        path). Identical behaviour to before the score/observe split."""
        self.score(row, micro, ctx, alpha_scores=alpha_scores)
        self.observe(row)
        return self.last

    # ------------------------------------------------------------- grading

    def _grade(self, row: dict) -> None:
        c = row.get("close", 0.0)
        if c <= 0:
            return
        while self._pending and self._idx - self._pending[0].idx >= self.horizon:
            p = self._pending.popleft()
            if p.close <= 0:
                continue
            ret = (c - p.close) / p.close
            norm = clamp(ret / max(p.atr / p.close, 1e-9), -2.5, 2.5)
            # 1) grade alphas -> within-desk Hedge weights
            for nm, s in p.alpha_scores.items():
                if abs(s) < 0.10:
                    continue
                payoff = math.copysign(min(abs(s), 1.0), s) * norm
                self.alpha_w[nm] *= math.exp(self.eta * clamp(payoff, -2, 2) * 0.25 / self.horizon)
                st = self.alpha_stats[nm]
                st.calls += 1
                st.payoff_sum += payoff
                if s * ret > 0:
                    st.hits += 1
            self._renormalize_desks()
            # 2) grade desks -> allocator
            for desk, sig in p.desk_sig.items():
                if abs(sig) < 0.10:
                    continue
                payoff = math.copysign(min(abs(sig), 1.0), sig) * norm
                self.allocator.update(desk, payoff, hit=(sig * ret > 0))
            # 3) grade calibrator on realized directional outcome
            if abs(p.edge) > 0.10:
                self.calibrator.update(p.edge, p.regime, p.atr_pctile, won=(p.edge * ret > 0))
            # 4) magnitude calibration (beta)
            pred = abs(p.edge) * (p.atr / p.close) * math.sqrt(self.horizon)
            if pred > 1e-9 and abs(p.edge) > 0.15:
                ratio = clamp(abs(ret) / pred, 0.1, 4.0)
                self.beta = clamp(self.beta + 0.03 * (ratio - self.beta), 0.3, 3.0)
            self.graded += 1

    def _renormalize_desks(self) -> None:
        lam = 0.004
        for names in DESKS.values():
            k = len(names)
            uni = 1.0 / k
            total = sum(self.alpha_w[nm] for nm in names) or 1.0
            for nm in names:
                self.alpha_w[nm] = (1 - lam) * (self.alpha_w[nm] / total) + lam * uni
            excess = {nm: max(self.alpha_w[nm] - self.floor, 0.0) for nm in names}
            tot = sum(excess.values())
            free = 1.0 - k * self.floor
            if tot <= 1e-12 or free <= 0:
                for nm in names:
                    self.alpha_w[nm] = uni
            else:
                for nm in names:
                    self.alpha_w[nm] = self.floor + excess[nm] * (free / tot)

    def _adapt_threshold(self) -> None:
        if not self.threshold_adapt or len(self._score_hist) < 120:
            self.threshold = self.base_threshold
            return
        # the target rate can never exceed what the bar clock allows (asking for
        # 3 trades/hour on 15m bars used to drive the quantile to its floor and
        # pin the gate open)
        rate = min(self.target_rate, 0.5 * self.bars_per_hour)
        p = clamp(1.0 - rate / self.bars_per_hour, 0.5, 0.995)
        hist = sorted(self._score_hist)
        q = hist[min(int(p * len(hist)), len(hist) - 1)]
        # the adaptive part may only TIGHTEN the gate above the OOS-validated
        # base_threshold — when edges run hot, the bar rises so we keep taking
        # only the fattest part of the distribution. It must never loosen BELOW
        # base to chase a trade-rate target: that was the marginal-churn
        # generator (P53% entries at thr 0.11), and it silently overrode the
        # one number the tuner actually validates out-of-sample.
        self.threshold = clamp(0.5 * q + 0.5 * self.base_threshold,
                               self.base_threshold, 0.92)

    # ------------------------------------------------------------- gating

    def entry_ok(self, edge: float, p_win: float, row: dict, fees_roundtrip: float,
                 spread_bps: float, slippage_bps: float) -> tuple[bool, str]:
        if abs(edge) < self.threshold:
            return False, f"edge {edge:+.2f} < threshold {self.threshold:.2f}"
        if p_win < self.min_p_win:
            return False, f"P(win) {p_win:.0%} < {self.min_p_win:.0%}"
        atr_pct = row.get("atr_pct", 0.0)
        if not math.isfinite(atr_pct) or atr_pct <= 0:
            return False, "no volatility estimate"
        predicted = self.beta * abs(edge) * atr_pct * math.sqrt(self.horizon)
        cost = fees_roundtrip + (spread_bps + 2 * slippage_bps) / 10_000.0
        if predicted < cost * self.cost_multiple:
            return False, f"edge {predicted*100:.3f}% < cost x{self.cost_multiple:.1f}"
        return True, "ok"

    def entry_report(self, edge: float, p_win: float, row: dict, fees_roundtrip: float,
                     spread_bps: float, slippage_bps: float) -> list[dict]:
        """Itemized version of entry_ok — every quality gate with its live numbers,
        pass or fail, for the UI's entry-gate X-ray. Same math, no early exit."""
        out = [{"n": "edge", "ok": abs(edge) >= self.threshold,
                "d": f"{edge:+.2f} vs thr {self.threshold:.2f}"},
               {"n": "p(win)", "ok": p_win >= self.min_p_win,
                "d": f"{p_win:.0%} vs min {self.min_p_win:.0%}"}]
        atr_pct = row.get("atr_pct", 0.0)
        if not math.isfinite(atr_pct) or atr_pct <= 0:
            out.append({"n": "cost", "ok": False, "d": "no volatility estimate"})
            return out
        predicted = self.beta * abs(edge) * atr_pct * math.sqrt(self.horizon)
        cost = (fees_roundtrip + (spread_bps + 2 * slippage_bps) / 10_000.0) * self.cost_multiple
        out.append({"n": "cost", "ok": predicted >= cost,
                    "d": f"move {predicted*100:.3f}% vs cost {cost*100:.3f}%"})
        return out

    def kelly_size_mult(self, p_win: float, payoff_ratio: float) -> float:
        """Fractional-Kelly multiplier on the base risk budget, from calibrated
        P(win) and the trade's reward:risk. Clamped so a hot streak can't run
        risk away; returns 0 for a negative-edge bet."""
        b = max(payoff_ratio, 0.1)
        f = p_win - (1 - p_win) / b
        if f <= 0:
            return 0.0
        return clamp(self.kelly_fraction * f * 4.0, 0.25, 1.75)

    def micro_confirms(self, edge: float, micro: dict) -> bool:
        lean = 0.6 * micro.get("flow", 0.0) + 0.4 * micro.get("obi", 0.0)
        return not (abs(lean) > 0.35 and lean * edge < 0)

    # ------------------------------------------------------------- persistence

    def state_dict(self) -> dict:
        """Everything the online learning has earned — hedge weights, desk
        performance, calibrator, beta, threshold history. Restarts used to
        reset all of it to uniform, throwing away days of adaptation."""
        # full precision throughout: persistence must restore EXACTLY what was
        # learned — rounding "for tidy JSON" made restored calibrator weights
        # drift by up to 5e-9, which is a different brain, not a saved one.
        return {
            "ver": 1, "alphas": list(ALPHAS),
            "alpha_w": dict(self.alpha_w),
            "stats": {nm: [st.calls, st.hits, st.payoff_sum]
                      for nm, st in self.alpha_stats.items()},
            "alloc_log_w": dict(self.allocator._log_w),
            "alloc_w": {d: p.weight for d, p in self.allocator.perf.items()},
            "alloc_perf": {d: [p.ew_payoff, p.ew_win, p.ew_var, p.graded, p.disabled]
                           for d, p in self.allocator.perf.items()},
            "cal": {"w": list(self.calibrator.w), "b": self.calibrator.b,
                    "n": self.calibrator.n},
            "beta": self.beta, "threshold": self.threshold,
            "graded": self.graded,
            "score_hist": list(self._score_hist)[-240:],
        }

    def load_state(self, d: dict) -> bool:
        """Restore a persisted learning state. Refuses (returns False) when the
        alpha roster changed between builds — stale weights for a different
        floor would be worse than a fresh start."""
        if not isinstance(d, dict) or d.get("ver") != 1 or d.get("alphas") != list(ALPHAS):
            return False
        try:
            for nm, w in d.get("alpha_w", {}).items():
                if nm in self.alpha_w:
                    self.alpha_w[nm] = float(w)
            # saved weights are already desk-normalized; re-normalizing here
            # would apply an extra shrink toward uniform. The next graded bar
            # renormalizes naturally.
            for nm, s in d.get("stats", {}).items():
                if nm in self.alpha_stats:
                    st = self.alpha_stats[nm]
                    st.calls, st.hits, st.payoff_sum = int(s[0]), int(s[1]), float(s[2])
            for dk, lw in d.get("alloc_log_w", {}).items():
                if dk in self.allocator._log_w:
                    self.allocator._log_w[dk] = float(lw)
            for dk, pv in d.get("alloc_perf", {}).items():
                perf = self.allocator.perf.get(dk)
                if perf is not None:
                    perf.ew_payoff, perf.ew_win = float(pv[0]), float(pv[1])
                    perf.ew_var, perf.graded = float(pv[2]), int(pv[3])
                    perf.disabled = bool(pv[4])
            # restore the exact stored weights — recomputing from the (already
            # decayed) log-weights would drift them; the next graded call
            # recomputes naturally anyway.
            for dk, w in d.get("alloc_w", {}).items():
                if dk in self.allocator.perf:
                    self.allocator.perf[dk].weight = float(w)
            cal = d.get("cal", {})
            if len(cal.get("w", [])) == len(self.calibrator.w):
                self.calibrator.w = [float(x) for x in cal["w"]]
                self.calibrator.b = float(cal.get("b", 0.0))
                self.calibrator.n = int(cal.get("n", 0))
            self.beta = float(d.get("beta", 1.0))
            self.threshold = float(d.get("threshold", self.base_threshold))
            self.graded = int(d.get("graded", 0))
            self._score_hist.extend(float(x) for x in d.get("score_hist", []))
            return True
        except (TypeError, ValueError, KeyError, IndexError):
            return False

    # ------------------------------------------------------------- exposure

    def alpha_state(self, name: str, score: float) -> str:
        if abs(score) > 0.05:
            return "firing"
        return "dormant" if ALPHA_META[name][1] == EVENT else "quiet"

    def viz(self) -> dict:
        """Tiny live wiring snapshot for the dashboard's cortex animation:
        every alpha's last score + hedge weight and every desk's signal +
        allocation — just enough to draw the brain firing, cheap enough to
        ride the 4 Hz hot channel. Non-finite scores become 0 (JSON-safe)."""
        sc = self._sc or {}
        scores = sc.get("alpha_scores") or {}
        desk_sig = sc.get("desk_sig") or {}
        alloc = self.allocator.weights()

        def f(x):
            v = float(x)
            return round(v, 3) if math.isfinite(v) else 0.0

        return {
            "a": [[nm, f(scores.get(nm, 0.0)), f(self.alpha_w.get(nm, 0.0))] for nm in ALPHAS],
            "d": {d: [f(desk_sig.get(d, 0.0)), f(alloc.get(d, 0.0))] for d in DESK_ORDER},
            "meta_p": self.meta_p,
            "thr": round(self.threshold, 4),
        }

    def snapshot(self) -> dict:
        last = self.last or {}
        scores = last.get("alpha_scores", {nm: 0.0 for nm in ALPHAS})
        alphas = {}
        for nm in ALPHAS:
            desk, kind = ALPHA_META[nm]
            st = self.alpha_stats[nm]
            alphas[nm] = {
                "desk": desk, "kind": kind,
                "score": round(scores.get(nm, 0.0), 3),
                "weight": round(self.alpha_w[nm], 4),
                "state": self.alpha_state(nm, scores.get(nm, 0.0)),
                "calls": st.calls, "hit_rate": round(st.hit_rate, 4),
                "payoff": round(st.payoff_sum, 3),
                "is_micro": nm in MICRO_ALPHAS,
            }
        desks = {}
        alloc = last.get("alloc", self.allocator.weights())
        perf = self.allocator.snapshot()
        for desk in DESK_ORDER:
            desks[desk] = {
                "signal": round(last.get("desk_sig", {}).get(desk, 0.0), 3),
                "conf": round(last.get("desk_conf", {}).get(desk, 0.0), 3),
                "alloc": round(alloc.get(desk, 0.0), 4),
                "alphas": DESKS[desk],
                **perf.get(desk, {}),
            }
        return {
            "edge": round(last.get("edge", 0.0), 4),
            "p_win": round(last.get("p_win", 0.5), 4),
            "regime": last.get("regime", "RANGE"),
            "regime_conf": round(last.get("regime_conf", 0.0), 3),
            "threshold": round(self.threshold, 4),
            "beta": round(self.beta, 3),
            "graded": self.graded,
            "expected_move": round(last.get("expected_move", 0.0), 6),
            "meta_p": round(self.meta_p, 4) if self.meta_p is not None else None,
            "calibration": self.calibrator.snapshot(),
            "alphas": alphas,
            "desks": desks,
        }
