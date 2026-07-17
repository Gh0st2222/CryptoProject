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
        self.beta = 1.0
        self.threshold = base_threshold
        self.graded = 0
        self.last: dict = {}       # last decision snapshot
        self._sc: dict | None = None   # components of the most recent score()

    # ------------------------------------------------------------- evaluate

    def score(self, row: dict, micro: dict, ctx: dict | None = None) -> dict:
        """Pure decision pass: features -> alphas -> desks -> fused edge -> P(win).
        No learning state is touched, so this can run **as often as we like**
        between bar closes (on every tick / order-book update) to react to live
        price, flow and the multi-timeframe context without corrupting the online
        weights. Returns the decision snapshot (also stored on ``self.last``)."""
        ctx = ctx or {}
        regime, conf = detect_regime(row)

        alpha_scores = {nm: float(fn(row, micro, ctx)) for nm, fn in ALPHAS.items()}
        desk_sig, desk_conf = {}, {}
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
            if active:
                same = sum(1 for nm in names if abs(alpha_scores[nm]) > 0.05 and alpha_scores[nm] * sig > 0)
                desk_conf[desk] = abs(sig) * (same / active)
            else:
                desk_conf[desk] = 0.0

        alloc = self.allocator.weights()
        rmult = REGIME_DESK_MULT[regime]
        num = den = 0.0
        for desk in DESK_ORDER:
            w = alloc[desk] * rmult.get(desk, 1.0)
            num += w * desk_sig[desk]
            den += w
        edge = clamp(num / den if den > 0 else 0.0, -1, 1)
        atr_pctile = row.get("atr_pctile", 0.5)
        p_win = self.calibrator.predict(edge, regime, atr_pctile)

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

    def evaluate(self, row: dict, micro: dict, ctx: dict | None = None) -> dict:
        """Score **and** learn from one closed bar (backtest + the live bar-close
        path). Identical behaviour to before the score/observe split."""
        self.score(row, micro, ctx)
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
        p = clamp(1.0 - self.target_rate / self.bars_per_hour, 0.5, 0.995)
        hist = sorted(self._score_hist)
        q = hist[min(int(p * len(hist)), len(hist) - 1)]
        self.threshold = clamp(0.5 * q + 0.5 * self.base_threshold,
                               0.55 * self.base_threshold, 0.92)

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

    # ------------------------------------------------------------- exposure

    def alpha_state(self, name: str, score: float) -> str:
        if abs(score) > 0.05:
            return "firing"
        return "dormant" if ALPHA_META[name][1] == EVENT else "quiet"

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
            "calibration": self.calibrator.snapshot(),
            "alphas": alphas,
            "desks": desks,
        }
