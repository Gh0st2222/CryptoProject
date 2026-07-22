"""Event-driven historical simulation running the SAME TradingBrain and
RiskManager code as live trading, plus a train/validate random-search
optimizer with an overfit guard.

Realism rules:
- decisions at bar close, fills at NEXT bar open (+slippage, taker fee)
- intrabar exits use bar extremes; if stop AND target are both inside one
  bar, the STOP is assumed to fill first (pessimistic)
- trailing stops advance on bar closes only
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict as dc_asdict
from dataclasses import asdict

import numpy as np

from ..config import RiskConfig, StrategyConfig
from ..exchange.models import LONG, SHORT, Candle, ContractSpec, Position
from ..risk.manager import RiskManager
from ..strategy.alphas import DESK_ORDER
from ..strategy.brain import TradingBrain
from ..strategy.exits import AdaptiveExitManager
from ..strategy.features import FeatureFrame
from ..util import clamp, interval_ms
from .portfolio import Portfolio

NO_MICRO = {"obi": 0.0, "flow": 0.0, "cvd_slope": 0.0, "spread_bps": 0.0, "ticks_per_s": 0.0}
NO_CTX: dict = {}


def _alpha_cache(ff) -> list[dict]:
    """Per-bar alpha scores for a SHARED FeatureFrame, computed once and reused
    by every tuner candidate: alphas are pure functions of the data (rows +
    the constant offline micro/ctx) — never of the parameters being tuned. On a
    28-member + 28-trial cycle this removes ~19 alpha calls x bars x candidates
    of redundant work, the biggest single cost in a tuner fold."""
    if getattr(ff, "_alpha", None) is None:
        from ..strategy.alphas import ALPHAS
        fns = list(ALPHAS.items())
        ff._alpha = [{nm: float(fn(ff.row_cached(i), NO_MICRO, NO_CTX)) for nm, fn in fns}
                     for i in range(ff.n)]
    return ff._alpha
ASSUMED_SPREAD_BPS = 1.0
EV_MARGIN = 0.02            # P(win) must clear the cost-adjusted breakeven prob by this
FILL_THROUGH_BPS = 1.0      # a resting limit fills only when price trades THROUGH it
                            # by this much — a wick that kisses the level never
                            # guarantees a maker fill (queue position is real)
STOP_SLIP_MULT = 2.0        # stops fire into momentum: pay double slippage there
TREND_REGIMES = {"TREND_UP", "TREND_DOWN"}
FUNDING_MS = 8 * 3600 * 1000        # perp funding settles every 8h
ASSUMED_FUNDING_8H = 0.0001         # 0.01%/8h baseline, charged as a COST while
                                    # holding (conservative: assume the paying side)


def gate_mtf_veto(strat, edge: float, row: dict) -> tuple[bool, str]:
    """HARD higher-timeframe trend filter — never fight a decided 15m/1h trend, in
    ANY regime. mtf_bias is the consensus of the rungs above the base; if it is
    clearly directional, a trade opposing it is refused outright. This is the
    rule that stops the account shorting into an uptrend."""
    bias = row.get("mtf_bias", 0.0)
    if strat.mtf_veto > 0 and abs(bias) >= strat.mtf_veto and bias * edge < 0:
        return False, f"15m/1h bias {bias:+.2f} vetoes edge {edge:+.2f}"
    return True, f"bias {bias:+.2f} · edge {edge:+.2f}"


def gate_funding(strat, edge: float, row: dict) -> tuple[bool, str]:
    """Funding awareness: if we'd pay meaningful funding to hold this side and the
    edge is only marginal, skip — the carry quietly eats a thin edge. (Funding is
    0 in the backtester's synthetic/historical klines, so this is live-only.)"""
    funding = row.get("funding_rate", 0.0) or 0.0
    if abs(funding) >= 0.0003 and (1 if edge > 0 else -1) * funding > 0 and abs(edge) < strat.base_threshold * 1.3:
        return False, f"funding {funding*100:+.4f}% vs thin edge {edge:+.2f}"
    return True, f"funding {funding*100:+.4f}%"


def gate_regime(strat, edge: float, row: dict, regime: str) -> tuple[bool, str]:
    """Regime-appropriate entry filter: ride confirmed, efficient, multi-timeframe-
    aligned trends, fade only the tails of a range, sit out chop entirely."""
    if not strat.discipline:
        if strat.trend_align_gate and regime in TREND_REGIMES:
            align = row.get("mtf_align", 0.0)
            if align * edge < 0 and abs(align) > 0.15:
                return False, f"MTF align {align:+.2f} opposes edge"
        return True, "discipline off"
    if regime in TREND_REGIMES:
        er = row.get("eff_ratio", 0.0)
        align = row.get("mtf_align", 0.0)
        if er < strat.min_efficiency:
            return False, f"trend ER {er:.2f} < {strat.min_efficiency:.2f}"
        if align * edge <= 0:
            return False, f"MTF align {align:+.2f} vs edge {edge:+.2f}"
        return True, f"ER {er:.2f} · align {align:+.2f}"
    if regime == "RANGE":
        if not strat.trade_range:
            return False, "range scalps off (tuner)"
        pctb = row.get("bb_pctb", 0.5)
        b = strat.range_band_edge
        if (pctb < b and edge > 0) or (pctb > 1 - b and edge < 0):
            return True, f"%B {pctb:.2f} at band"
        return False, f"%B {pctb:.2f} not at band ≤{b:.2f}"
    if strat.trade_volatile:
        return True, "volatile allowed"
    return False, "volatile chop (sitting out)"


def gate_ev(risk_cfg, payoff_b: float, p_win: float, row: dict, fees_rt: float,
            spread_bps: float, slippage_bps: float) -> tuple[bool, str]:
    """Expected-value floor — the gate that refuses coin-flip entries.

    P(win) must clear the breakeven probability at the MEASURED payoff ratio
    with the real round-trip cost expressed in stop-distance units:
    p_be = (1 + cost_R) / (1 + b). Every live loss so far entered at p_win
    48-51% with a measured payoff near 1:1 — mathematically dead before fees.
    As the exits' realized payoff decays, this bar rises automatically at
    exactly the rate the edge is decaying; when payoff is healthy the gate
    stays out of the way (prior b≈2.6 puts the floor near 30%)."""
    atr_pct = row.get("atr_pct", 0.0)
    if not (isinstance(atr_pct, float) and math.isfinite(atr_pct)) or atr_pct <= 0:
        return False, "no volatility estimate"
    stop_pct = max(risk_cfg.sl_atr_min * atr_pct, 1e-9)
    cost_r = (fees_rt + (spread_bps + 2.0 * slippage_bps) / 10_000.0) / stop_pct
    b = max(payoff_b, 0.1)
    need = min((1.0 + cost_r) / (1.0 + b) + EV_MARGIN, 0.92)
    if p_win < need:
        return False, f"P {p_win:.0%} < EV floor {need:.0%} (b {b:.2f})"
    return True, f"P {p_win:.0%} ≥ EV floor {need:.0%} (b {b:.2f})"


def _entry_signal_ok(brain, strat, risk_cfg, edge: float, p_win: float, row: dict, ev: dict,
                     fees_rt: float, slippage_bps: float, payoff_b: float) -> bool:
    """The full entry filter chain — brain quality gates, the hard MTF veto,
    funding awareness, the regime branch, then the expected-value floor. The
    single biggest fix for fee-drag: only trade where an edge actually exists."""
    ok, _ = brain.entry_ok(edge, p_win, row, fees_rt, ASSUMED_SPREAD_BPS, slippage_bps)
    if not ok:
        return False
    if not gate_mtf_veto(strat, edge, row)[0]:
        return False
    if not gate_funding(strat, edge, row)[0]:
        return False
    if not gate_regime(strat, edge, row, ev["regime"])[0]:
        return False
    return gate_ev(risk_cfg, payoff_b, p_win, row, fees_rt, ASSUMED_SPREAD_BPS, slippage_bps)[0]


def candles_to_arrays(candles: list[Candle]) -> dict[str, np.ndarray]:
    return {
        "ts": np.array([c.ts for c in candles], dtype=np.int64),
        "open": np.array([c.open for c in candles], dtype=np.float64),
        "high": np.array([c.high for c in candles], dtype=np.float64),
        "low": np.array([c.low for c in candles], dtype=np.float64),
        "close": np.array([c.close for c in candles], dtype=np.float64),
        "volume": np.array([c.volume for c in candles], dtype=np.float64),
    }


class _SymSim:
    """One symbol's event-driven simulation. It owns its brain and exit manager
    and its per-symbol state; the portfolio, risk manager and simulated clock are
    passed in and SHARED, so several _SymSim instances can trade a single account
    — that is what the portfolio backtest is."""

    def __init__(self, symbol, interval, candles, strat, risk_cfg, spec,
                 taker_fee, slippage_bps, collect_series, ff=None, pending_slots=None):
        self.symbol = symbol
        self.strat = strat
        self.spec = spec
        self.risk_cfg = risk_cfg
        self.collect = collect_series
        # shared across the sims of a portfolio backtest: a pending entry is a
        # RESERVED position slot, exactly as in the live engine — otherwise
        # several symbols' pendings can all fill and exceed max_open_positions.
        self._slots = pending_slots if pending_slots is not None else {"n": 0}
        self._shared_ff = ff is not None
        if ff is not None:
            # reuse a precomputed FeatureFrame AND its OHLC arrays — this is what
            # lets the tuner score dozens of candidates on one fold without
            # rebuilding indicators (or even the arrays) each time. Row dicts
            # and alpha scores are also cached ON the frame (candidate-invariant)
            # so they too are computed once per fold, not once per candidate.
            self.ff = ff
            f = ff.f
            self.o, self.h, self.l = f["open"], f["high"], f["low"]
            self.c, self.ts = f["close"], f["ts"]
            self.n = ff.n
        else:
            arrays = candles_to_arrays(candles)
            self.o, self.h, self.l = arrays["open"], arrays["high"], arrays["low"]
            self.c, self.ts = arrays["close"], arrays["ts"]
            self.n = len(self.c)
            self.ff = FeatureFrame(arrays, interval=interval)
        bph = 3_600_000 / interval_ms(interval)
        self.brain = TradingBrain(
            eta=strat.hedge_eta, weight_floor=strat.weight_floor, horizon_bars=strat.horizon_bars,
            base_threshold=strat.base_threshold, threshold_adapt=strat.threshold_adapt,
            target_trades_per_hour=strat.target_trades_per_hour,
            bars_per_hour=bph, cost_multiple=strat.cost_multiple,
            min_p_win=strat.min_p_win, kelly_fraction=strat.kelly_fraction)
        self.exits = AdaptiveExitManager(risk_cfg)
        self.taker = taker_fee
        self.maker_fee = getattr(spec, "maker_fee", 0.0002)
        self.slip = slippage_bps / 10_000.0
        self.slippage_bps = slippage_bps
        self.maker_adv = risk_cfg.maker_adverse_bps / 10_000.0
        self.maker_off = strat.maker_offset_bps / 10_000.0
        self.is_maker = strat.entry_mode == "maker"
        self.fees_rt = taker_fee + (self.maker_fee if self.is_maker else taker_fee)
        self.pending = None
        self.bars_held = 0
        self.planned_risk = 0.0
        self.corr_map: dict | None = None   # set by the portfolio backtest
        self.markers: list[dict] = []
        self.regime_counts: dict[str, int] = {}
        self.last_ev: dict = {}

    def open_at(self, i, pf, risk, side, px, atr, regime, style, size_mult, reason, maker, marks):
        d = 1 if side == LONG else -1
        eff = px * (1 + d * self.maker_adv) if maker else px * (1 + d * self.slip)
        row_i = self.ff.row_cached(i) if self._shared_ff else self.ff.row(i)
        br = self.exits.initial_bracket(eff, side, atr, row_i, regime, style)
        if br is None:
            return
        sized = risk.size_entry(pf.equity(marks), eff, br.init_risk, side, self.spec, size_mult)
        if sized is None:
            return
        fee = sized.qty * eff * (self.maker_fee if maker else self.taker)
        pos = Position(symbol=self.symbol, side=side, qty=sized.qty, entry_price=eff,
                       opened_ts=int(self.ts[i]), leverage=sized.leverage, style=style,
                       stop_price=br.stop, take_profit=br.take_profit,
                       entry_fee=fee, entry_reason=reason, entry_bar_ts=int(self.ts[i]))
        self.exits.attach(pos, atr, br.init_risk)
        pf.open_position(pos, fee)
        self.planned_risk = sized.risk_amount
        self.bars_held = 0
        if self.collect:
            self.markers.append({"ts": int(self.ts[i]), "kind": "entry", "side": side,
                                 "price": round(eff, 8), "reason": reason})

    def _pair_haircut(self, other: str) -> float:
        cm = self.corr_map
        if cm is None:
            return self.risk_cfg.correlation_haircut
        c = cm.get((self.symbol, other), cm.get((other, self.symbol)))
        if c is None:
            return self.risk_cfg.correlation_haircut
        return clamp(1.0 - 0.6 * max(c, 0.0), 0.4, 1.0)

    def scale_at(self, i, pf, risk, px, marks):
        """Bank scaleout_frac at the close (taker, slipped) and trail the rest;
        a dust-sized partial degrades to a full close."""
        pos = pf.positions.get(self.symbol)
        if pos is None:
            return
        frac = self.risk_cfg.scaleout_frac
        qty_out = pos.qty * frac
        if qty_out < self.spec.min_qty or (pos.qty - qty_out) < self.spec.min_qty:
            self.close_at(i, pf, risk, px, "scale out (full)", marks)
            return
        eff = px * (1 - self.slip) if pos.side == LONG else px * (1 + self.slip)
        fee = qty_out * eff * self.taker
        tr = pf.scale_out(self.symbol, frac, eff, int(self.ts[i]), fee, "scale out")
        if tr:
            risk.on_trade_closed(tr, pf.equity(marks))
            if self.collect:
                self.markers.append({"ts": int(self.ts[i]), "kind": "exit", "side": pos.side,
                                     "price": round(eff, 8), "pnl": round(tr.pnl, 6),
                                     "reason": "scale out"})

    def close_at(self, i, pf, risk, px, reason, marks, maker=False, slip_mult=1.0):
        pos = pf.positions.get(self.symbol)
        if pos is None:
            return
        if maker:
            fee = pos.qty * px * self.maker_fee
        else:
            s = self.slip * slip_mult
            px = px * (1 - s) if pos.side == LONG else px * (1 + s)
            fee = pos.qty * px * self.taker
        tr = pf.close_position(self.symbol, px, int(self.ts[i]), fee, reason, self.planned_risk)
        if tr:
            risk.on_trade_closed(tr, pf.equity(marks))
            if self.collect:
                self.markers.append({"ts": int(self.ts[i]), "kind": "exit", "side": pos.side,
                                     "price": round(px, 8), "pnl": round(tr.pnl, 6), "reason": reason})
        self.bars_held = 0

    def step(self, i, pf, risk, marks):
        sym = self.symbol
        o, h, l, c = self.o, self.h, self.l, self.c
        row = self.ff.row_cached(i) if self._shared_ff else self.ff.row(i)
        pos = pf.positions.get(sym)

        # 1) resolve a pending entry from the previous close
        if self.pending is not None and pos is None:
            p = self.pending
            side = p["side"]
            if p["mode"] == "taker":
                self.open_at(i, pf, risk, side, o[i], p["atr"], p["regime"], p["style"],
                             p["size_mult"], p["reason"], False, marks)
                self.pending = None
                self._slots["n"] -= 1
            else:
                lim = p["limit"]
                # maker fills require price to trade THROUGH the level, not
                # merely touch it — the exact wick-kiss "fill" the live engine
                # can never collect was flattering every pullback entry.
                thru = FILL_THROUGH_BPS / 10_000.0
                hit = (l[i] <= lim * (1 - thru)) if side == LONG else (h[i] >= lim * (1 + thru))
                if hit:
                    self.open_at(i, pf, risk, side, lim, p["atr"], p["regime"], p["style"],
                                 p["size_mult"], p["reason"], True, marks)
                    self.pending = None
                    self._slots["n"] -= 1
                elif i >= p["expires_bar"]:
                    self.pending = None
                    self._slots["n"] -= 1
            pos = pf.positions.get(sym)

        # funding drag: crossing an 8h settlement while holding costs the assumed
        # rate on notional — long holds must EARN their carry, exactly as live.
        if pos is not None and i > 0 and (self.ts[i] // FUNDING_MS) != (self.ts[i - 1] // FUNDING_MS):
            pf.charge_funding(pos.qty * c[i] * ASSUMED_FUNDING_8H)

        # 2) intrabar exits: stop taker, scalp target passive maker
        if pos is not None:
            d = pos.direction()
            stop, tp = pos.stop_price, pos.take_profit
            if stop > 0 and ((l[i] <= stop) if d > 0 else (h[i] >= stop)):
                gap = (o[i] < stop) if d > 0 else (o[i] > stop)
                # stops fire INTO momentum — they pay extra slippage, always
                self.close_at(i, pf, risk, o[i] if gap else stop,
                              "stop" if not pos.breakeven_moved else "trail stop", marks,
                              slip_mult=STOP_SLIP_MULT)
            elif tp > 0 and ((h[i] >= tp) if d > 0 else (l[i] <= tp)):
                self.close_at(i, pf, risk, tp, "target", marks, maker=True)
            pos = pf.positions.get(sym)

        # 3) brain (precomputed alpha scores when the frame is shared)
        ev = self.brain.evaluate(row, NO_MICRO, NO_CTX,
                                 alpha_scores=_alpha_cache(self.ff)[i] if self._shared_ff else None)
        self.last_ev = ev
        edge, p_win, regime = ev["edge"], ev["p_win"], ev["regime"]
        self.regime_counts[regime] = self.regime_counts.get(regime, 0) + 1

        # 4) bar-close management
        if pos is not None:
            self.bars_held += 1
            _, reason = self.exits.manage(pos, c[i], h[i], l[i], row.get("atr", 0.0),
                                          row, edge, ev["threshold"], regime, self.bars_held)
            if reason == "scale out":
                self.scale_at(i, pf, risk, c[i], marks)
            elif reason:
                self.close_at(i, pf, risk, c[i], reason, marks)
        elif self.pending is None:
            # 5) entry decision for the next bar (pendings reserve slots)
            equity = pf.equity(marks)
            ok, _ = risk.can_enter(equity, len(pf.positions) + self._slots["n"], ASSUMED_SPREAD_BPS)
            style = "scalp" if regime == "RANGE" else "trend"
            payoff_b = risk.payoff_ratio(style)
            if ok and _entry_signal_ok(self.brain, self.strat, self.risk_cfg, edge, p_win, row, ev,
                                       self.fees_rt, self.slippage_bps, payoff_b):
                kelly = self.brain.kelly_size_mult(p_win, payoff_b) if self.strat.use_kelly else 1.0
                size_mult = kelly * risk.health.scalar
                side = LONG if edge > 0 else SHORT
                side_d = 1 if side == LONG else -1
                # correlation haircut: shrink a same-direction add by the measured
                # pair correlation (portfolio backtests precompute it; single-
                # symbol runs have no siblings and never hit this).
                held_same = [s for s, pp in pf.positions.items()
                             if s != sym and pp.direction() == side_d]
                if held_same:
                    size_mult *= min(self._pair_haircut(s) for s in held_same)
                pend = {"side": side, "atr": row.get("atr", 0.0), "regime": regime, "style": style,
                        "size_mult": size_mult, "reason": f"{style} edge {edge:+.2f} P{p_win:.0%} {regime}"}
                pull = getattr(self.strat, "entry_pullback_atr", 0.0)
                atr_now = row.get("atr", 0.0)
                if pull > 0 and style == "trend" and atr_now > 0:
                    # pullback entry: rest the limit DEEP behind the signal and
                    # let the retrace come to us — a limit is maker by nature,
                    # whatever entry_mode says. Unfilled in the window = the
                    # move ran without us; the entry is abandoned, not chased.
                    pend["mode"] = "maker"
                    pend["limit"] = c[i] - (1 if side == LONG else -1) * pull * atr_now
                    pend["expires_bar"] = i + self.strat.maker_wait_bars
                elif self.is_maker:
                    pend["mode"] = "maker"
                    pend["limit"] = c[i] * (1 - self.maker_off) if side == LONG else c[i] * (1 + self.maker_off)
                    pend["expires_bar"] = i + self.strat.maker_wait_bars
                else:
                    pend["mode"] = "taker"
                self.pending = pend
                self._slots["n"] += 1


def run_backtest(
    candles: list[Candle],
    symbol: str,
    interval: str,
    strat: StrategyConfig,
    risk_cfg: RiskConfig,
    spec: ContractSpec | None = None,
    starting_balance: float = 10_000.0,
    taker_fee: float = 0.0005,
    slippage_bps: float = 1.5,
    warmup: int = 300,
    progress_cb=None,
    collect_series: bool = True,
    ff=None,
) -> dict:
    spec = spec or ContractSpec(symbol)
    n = len(candles)
    if n < warmup + 50:
        return {"error": f"not enough bars ({n}); need at least {warmup + 50}"}

    sim = _SymSim(symbol, interval, candles, strat, risk_cfg, spec, taker_fee,
                  slippage_bps, collect_series, ff=ff)
    ts, c = sim.ts, sim.c
    sim_ts = {"v": float(ts[warmup]) / 1000.0}
    risk = RiskManager(risk_cfg, clock=lambda: sim_ts["v"])
    pf = Portfolio(starting_balance, mode="backtest")
    weights_timeline: list[dict] = []
    stride = max(1, (n - warmup) // 160)

    for i in range(warmup, n):
        sim_ts["v"] = float(ts[i]) / 1000.0
        marks = {symbol: c[i]}
        sim.step(i, pf, risk, marks)
        pf.record_equity(int(ts[i]), marks, min_gap_ms=0)
        if collect_series and (i - warmup) % stride == 0 and sim.last_ev:
            alloc = sim.last_ev.get("alloc", {})
            weights_timeline.append({"ts": int(ts[i]), **{d: round(alloc.get(d, 0.0), 4) for d in DESK_ORDER}})
        if progress_cb and (i - warmup) % 500 == 0:
            progress_cb((i - warmup) / (n - warmup))

    if pf.positions.get(symbol) is not None:
        sim.close_at(n - 1, pf, risk, c[n - 1], "backtest end", {symbol: c[n - 1]})
        # the forced close changed cash AFTER the last curve point — re-record,
        # or stats["equity"] is stale by the final exit fee and the accounting
        # identity (start + sum(pnl) - funding == equity) silently breaks.
        pf.record_equity(int(ts[-1]), {symbol: c[n - 1]}, min_gap_ms=0)

    stats = pf.stats()
    curve = list(pf.equity_curve)
    if len(curve) > 2000:
        step = len(curve) / 2000.0
        curve = [curve[int(k * step)] for k in range(2000)] + [curve[-1]]
    brain = sim.brain
    markers = sim.markers
    regime_counts = sim.regime_counts
    result = {
        "symbol": symbol,
        "interval": interval,
        "bars": n,
        "start_ts": int(ts[0]),
        "end_ts": int(ts[-1]),
        "starting_balance": starting_balance,
        "stats": stats,
        "brain": brain.snapshot(),
        "regime_counts": regime_counts,
        "params": {"strategy": asdict(strat), "risk": asdict(risk_cfg),
                   "taker_fee": taker_fee, "slippage_bps": slippage_bps},
    }
    if collect_series:
        result["equity_curve"] = curve
        result["trades"] = [dc_asdict(t) for t in pf.trades]
        result["markers"] = markers[-800:]
        result["weights_timeline"] = weights_timeline
    if progress_cb:
        progress_cb(1.0)
    return result


# ------------------------------------------------------- portfolio backtest

def run_portfolio_backtest(
    candles_by_symbol: dict[str, list[Candle]],
    interval: str,
    strat: StrategyConfig,
    risk_cfg: RiskConfig,
    specs: dict[str, ContractSpec] | None = None,
    starting_balance: float = 10_000.0,
    taker_fee: float = 0.0005,
    slippage_bps: float = 1.5,
    warmup: int = 300,
    progress_cb=None,
) -> dict:
    """Trade several symbols on ONE shared account. Sizing, the position cap,
    the daily-loss kill switch and the health governor are all portfolio-level,
    and a correlation haircut shrinks same-direction adds — so the smoother,
    diversified equity curve can safely carry more size than any one symbol.
    Symbols are aligned on their common timestamp grid."""
    specs = specs or {}
    syms = [s for s, cs in candles_by_symbol.items() if len(cs) >= warmup + 50]
    if len(syms) < 1:
        return {"error": "need at least one symbol with enough bars"}
    shared_slots = {"n": 0}   # pendings reserve position slots account-wide
    sims = {s: _SymSim(s, interval, candles_by_symbol[s], strat, risk_cfg,
                       specs.get(s, ContractSpec(s)), taker_fee, slippage_bps, True,
                       pending_slots=shared_slots)
            for s in syms}
    # align on the intersection of timestamps
    ts_index = {s: {int(t): i for i, t in enumerate(sim.ts)} for s, sim in sims.items()}
    common = sorted(set.intersection(*[set(idx) for idx in ts_index.values()]))
    if len(common) < warmup + 50:
        return {"error": f"symbols share only {len(common)} aligned bars"}
    common = common[warmup:]   # every symbol has >= warmup bars before this point

    # measured pair correlations over the aligned grid feed the same-direction
    # size haircut (live measures rolling; here the window is the measurement)
    if len(syms) >= 2:
        aligned = {s: np.array([sims[s].c[ts_index[s][t]] for t in common], dtype=np.float64)
                   for s in syms}
        rets_by = {s: np.diff(a) / np.maximum(a[:-1], 1e-9) for s, a in aligned.items()}
        corr_map: dict = {}
        for ai in range(len(syms)):
            for bi in range(ai + 1, len(syms)):
                a, b = syms[ai], syms[bi]
                corr_map[(a, b)] = float(np.corrcoef(rets_by[a], rets_by[b])[0, 1])
        for s in syms:
            sims[s].corr_map = corr_map

    sim_ts = {"v": float(common[0]) / 1000.0}
    risk = RiskManager(risk_cfg, clock=lambda: sim_ts["v"])
    pf = Portfolio(starting_balance, mode="backtest")
    total = len(common)

    for k, t in enumerate(common):
        sim_ts["v"] = float(t) / 1000.0
        idxs = {s: ts_index[s][t] for s in syms}
        marks = {s: sims[s].c[idxs[s]] for s in syms}
        for s in syms:
            sims[s].step(idxs[s], pf, risk, marks)
        pf.record_equity(int(t), marks, min_gap_ms=0)
        if progress_cb and k % 500 == 0:
            progress_cb(k / total)

    last_t = common[-1]
    last_marks = {s: sims[s].c[ts_index[s][last_t]] for s in syms}
    forced = False
    for s in syms:
        if pf.positions.get(s) is not None:
            sims[s].close_at(ts_index[s][last_t], pf, risk, last_marks[s], "backtest end", last_marks)
            forced = True
    if forced:   # keep the equity curve's final point in sync with the closes
        pf.record_equity(int(last_t), last_marks, min_gap_ms=0)

    # portfolio + per-symbol stats
    stats = pf.stats()
    curve = list(pf.equity_curve)
    if len(curve) > 2000:
        step = len(curve) / 2000.0
        curve = [curve[int(k * step)] for k in range(2000)] + [curve[-1]]
    per_symbol = {}
    for s in syms:
        strades = [t for t in pf.trades if t.symbol == s]
        wins = sum(1 for t in strades if t.pnl > 0)
        per_symbol[s] = {
            "trades": len(strades),
            "win_rate": round(wins / len(strades), 4) if strades else 0.0,
            "pnl": round(sum(t.pnl for t in strades), 4),
        }
    # correlation of the symbols' close-to-close returns over the common grid
    corr = None
    if len(syms) >= 2:
        rets = []
        for s in syms:
            arr = np.array([sims[s].c[ts_index[s][t]] for t in common], dtype=np.float64)
            rets.append(np.diff(arr) / np.maximum(arr[:-1], 1e-9))
        cm = np.corrcoef(np.vstack(rets))
        # average off-diagonal correlation
        m = cm.shape[0]
        off = [cm[a][b] for a in range(m) for b in range(m) if a != b]
        corr = round(float(np.mean(off)), 3) if off else None

    if progress_cb:
        progress_cb(1.0)
    return {
        "symbols": syms,
        "interval": interval,
        "bars": len(common),
        "starting_balance": starting_balance,
        "stats": stats,
        "per_symbol": per_symbol,
        "avg_correlation": corr,
        "equity_curve": curve,
        "trades": [dc_asdict(t) for t in pf.trades[-400:]],
        "risk": risk.status(),
    }


# ------------------------------------------------------- walk-forward (honest)

def run_walkforward(
    candles: list[Candle],
    symbol: str,
    interval: str,
    strat: StrategyConfig,
    risk_cfg: RiskConfig,
    spec: ContractSpec | None = None,
    starting_balance: float = 10_000.0,
    taker_fee: float = 0.0005,
    slippage_bps: float = 1.5,
    folds: int = 5,
    trials: int = 20,
    progress_cb=None,
) -> dict:
    """The honest test. Split history into sequential folds; for each fold, tune
    parameters using ONLY the data before it, then trade that fold once,
    out-of-sample, with those frozen params — chaining equity fold to fold. The
    result is what you'd actually have earned walking the strategy forward through
    time, params and all, with no peeking. In-sample backtests flatter; this does
    not."""
    spec = spec or ContractSpec(symbol)
    n = len(candles)
    if n < folds * 1200:
        return {"error": f"walk-forward needs ~{folds * 1200}+ bars, got {n}"}
    fold_size = n // folds
    equity = starting_balance
    curve: list = []
    per_fold: list[dict] = []
    all_trades: list[dict] = []
    steps = folds - 1

    for i in range(1, folds):
        train = candles[: i * fold_size]
        lo = i * fold_size
        hi = n if i == folds - 1 else (i + 1) * fold_size
        test = candles[lo:hi]
        # choose params from the PAST only (train/valid split lives inside)
        opt = run_optimizer(train, symbol, interval, strat, risk_cfg, spec,
                            taker_fee, slippage_bps, n_trials=trials)
        best = opt.get("best")
        if best and best.get("params"):
            s, r = _apply_params(strat, risk_cfg, best["params"])
            params_used = best["params"]
        else:
            s, r, params_used = strat, risk_cfg, {}
        prev_equity = equity
        res = run_backtest(test, symbol, interval, s, r, spec, starting_balance=prev_equity,
                           taker_fee=taker_fee, slippage_bps=slippage_bps, collect_series=True)
        st = res.get("stats", {})
        equity = st.get("equity", prev_equity)
        curve.extend(res.get("equity_curve", []))
        all_trades.extend(res.get("trades", []))
        per_fold.append({
            "fold": i, "train_bars": len(train), "test_bars": len(test),
            "start_ts": res.get("start_ts"), "end_ts": res.get("end_ts"),
            "trades": st.get("trades", 0), "win_rate": st.get("win_rate", 0.0),
            "profit_factor": st.get("profit_factor", 0.0),
            "return_pct": round((equity / prev_equity - 1) * 100, 2) if prev_equity > 0 else 0.0,
            "max_drawdown": st.get("max_drawdown", 0.0),
            "tuned": bool(params_used),
        })
        if progress_cb:
            progress_cb(i / steps)

    if len(curve) > 2000:
        step = len(curve) / 2000.0
        curve = [curve[int(k * step)] for k in range(2000)] + [curve[-1]]
    wins = sum(1 for t in all_trades if t["pnl"] > 0)
    gross_w = sum(t["pnl"] for t in all_trades if t["pnl"] > 0)
    gross_l = -sum(t["pnl"] for t in all_trades if t["pnl"] <= 0)
    ntr = len(all_trades)
    curve_vals = [e for _, e in curve] or [starting_balance]
    peak, max_dd = curve_vals[0], 0.0
    for e in curve_vals:
        peak = max(peak, e)
        if peak > 0:
            max_dd = max(max_dd, (peak - e) / peak)
    return {
        "symbol": symbol,
        "interval": interval,
        "folds": folds,
        "starting_balance": starting_balance,
        "final_equity": round(equity, 4),
        "oos_return_pct": round((equity / starting_balance - 1) * 100, 2),
        "oos_trades": ntr,
        "oos_win_rate": round(wins / ntr, 4) if ntr else 0.0,
        "oos_profit_factor": round(gross_w / gross_l, 3) if gross_l > 0 else (999.0 if gross_w > 0 else 0.0),
        "oos_max_drawdown": round(max_dd, 4),
        "equity_curve": curve,
        "per_fold": per_fold,
        "note": "out-of-sample: every fold traded with params tuned only on prior data",
    }


# --------------------------------------------------------------- optimizer

# The full auto-owned parameter space the tuner searches. User-owned settings
# (symbols, feed, interval, warmup, leverage band, daily-loss, max positions,
# starting balance) are deliberately excluded. (lo, hi, target, kind)
TUNABLES: dict[str, tuple] = {
    # strategy — floors widened after live evidence: on real 1m data the DE pinned
    # base_threshold, target rate AND min_efficiency at their old floors (it wanted
    # to explore looser gates but the box stopped it). Wider floors let OOS
    # validation decide what actually pays instead of the box deciding a priori.
    # ceiling raised after the speaking-desks fusion fix: backtest edges now
    # sit on the same (undiluted) scale live always had, so the search must be
    # able to reach genuinely tight gates on that scale.
    "base_threshold":         (0.12, 0.55, "strategy", "float"),
    "target_trades_per_hour": (0.2, 6.0, "strategy", "float"),
    "cost_multiple":          (1.2, 3.2, "strategy", "float"),
    "hedge_eta":              (0.15, 0.60, "strategy", "float"),
    "horizon_bars":           (5, 16, "strategy", "int"),
    "min_efficiency":         (0.10, 0.50, "strategy", "float"),
    "min_p_win":              (0.40, 0.60, "strategy", "float"),
    "kelly_fraction":         (0.15, 0.60, "strategy", "float"),
    "maker_offset_bps":       (0.0, 3.0, "strategy", "float"),
    # pullback entries: rest the trend-entry limit this deep behind the signal
    # (in ATRs) and let the retrace fill us — 0 = chase at the touch. The tuner
    # owns the depth, so "wait for a dip" only survives if it beats "don't miss
    # the move" out-of-sample.
    "entry_pullback_atr":     (0.0, 1.2, "strategy", "float"),
    # range scalping is a tuner OPTION again (it was banned after it faded uptrends):
    # the hard MTF veto now blocks fading a decided 15m/1h trend, so an enabled range
    # scalp only takes the with-trend side of a range (or both sides in a truly flat
    # one). The tuner turns it on only if it survives out-of-sample.
    "trade_range":            (0, 1, "strategy", "bool"),
    "range_band_edge":        (0.10, 0.30, "strategy", "float"),
    # risk / exits
    "risk_per_trade":         (0.004, 0.014, "risk", "float"),
    "sl_atr_min":             (1.4, 2.6, "risk", "float"),
    "sl_atr_max":             (2.2, 3.8, "risk", "float"),
    "trail_atr_min":          (1.2, 2.6, "risk", "float"),
    "trail_atr_max":          (2.6, 4.6, "risk", "float"),
    "trail_tighten":          (0.30, 0.75, "risk", "float"),
    "be_rr":                  (0.5, 1.6, "risk", "float"),
    "giveback_rr":            (1.6, 3.6, "risk", "float"),
    "giveback_frac":          (0.35, 0.70, "risk", "float"),
    # regime-conditional exit geometry: the tuner can finally express "let it
    # run in trends, keep it tight in chop" instead of one compromise trail.
    "trail_scale_trend":      (0.7, 1.5, "risk", "float"),
    "trail_scale_chop":       (0.7, 1.5, "risk", "float"),
    # partial scale-out: bank half at this R, trail the rest (0 = off) — the
    # measured fix for "showed +0.6R and round-tripped to the stop".
    "scaleout_rr":            (0.0, 2.5, "risk", "float"),
    "hold_edge_frac":         (0.5, 1.0, "risk", "float"),
    "expected_rr":            (1.6, 3.0, "risk", "float"),
    "time_stop_bars":         (60, 200, "risk", "int"),
}


def _coerce(name: str, v):
    kind = TUNABLES[name][3]
    if kind == "int":
        return int(round(v))
    if kind == "bool":
        return bool(v >= 0.5) if not isinstance(v, bool) else v
    return round(float(v), 4)


def apply_tunables_inplace(strat: StrategyConfig, risk: RiskConfig, p: dict) -> None:
    for name, val in p.items():
        spec = TUNABLES.get(name)
        if not spec:
            continue
        lo, hi, grp, kind = spec
        target = strat if grp == "strategy" else risk
        # clamp into the searched box: params also arrive from the champion
        # vault and the HTTP apply endpoint, not only from the DE (whose
        # trials are already in-bounds) — an out-of-range value must not be
        # able to set e.g. risk_per_trade to something unbounded.
        if kind != "bool":
            val = clamp(float(val), lo, hi)
        setattr(target, name, _coerce(name, val))
    # keep dependent bounds coherent
    risk.sl_atr_max = max(risk.sl_atr_max, risk.sl_atr_min + 0.4)
    risk.trail_atr_max = max(risk.trail_atr_max, risk.trail_atr_min + 0.4)


def _apply_params(strat: StrategyConfig, risk: RiskConfig, p: dict) -> tuple[StrategyConfig, RiskConfig]:
    s = StrategyConfig(**{**asdict(strat)})
    r = RiskConfig(**{**asdict(risk)})
    apply_tunables_inplace(s, r, p)
    return s, r


FITNESS_VER = 3   # bump when the fitness scale changes — birth scores recorded
                  # under a different version are not comparable to current ones
                  # (v3: honest fills — trade-through limits, double stop slip —
                  # compressed the whole scale vs v2)


def _fitness(stats: dict) -> float:
    """LOG-WEALTH growth as a SMOOTH, ordered objective — what compounding
    actually maximizes (Kelly-consistent), net of fees AND funding drag, with a
    CONVEX drawdown penalty (at leverage, variance is not a nuisance, it's the
    thing that ends accounts). Scaled x100 so scores stay in a familiar range
    (+10% window growth ~ +9.5 before penalties). Still smooth and strictly
    ordered — a losing set scores negative and less-losing scores higher, so
    Differential Evolution always has a gradient; promotion is gated separately."""
    t = stats.get("trades", 0)
    if t < 5:
        # too little evidence to judge — a gentle ramp so the search is pulled
        # toward configs that at least trade, instead of a flat dead zone.
        return -2.0 + 0.2 * t
    ret = clamp(stats.get("total_return", 0.0), -0.95, 20.0)
    growth = 100.0 * math.log1p(ret)               # log-wealth, %-like scale
    dd = stats.get("max_drawdown", 1.0)
    dd_pen = 1.0 / (1.0 + (dd / 0.08) ** 2)        # convex: 4% dd -> x0.80, 8% -> x0.50, 16% -> x0.20
    pf = stats.get("profit_factor", 0.0)
    if growth > 0:
        quality = clamp(min(pf, 3.0) / 1.5, 0.3, 2.0)   # stability of the earning, capped
        score = growth * dd_pen * quality
        # among winners, mild preference for more evidence
        score *= clamp(0.8 + 0.2 * (t / 30.0), 0.8, 1.4)
    else:
        # losers: junkier losing (low pf) must score MORE negative, never less —
        # multiplying a negative by a small "quality" would invert the ordering.
        score = growth * (1.0 + (1.0 - clamp(pf, 0.0, 1.0)))
    return score


def robust_fitness(candles, symbol, interval, strat, risk_cfg, spec,
                   taker_fee=0.0005, slippage_bps=1.5, folds: int = 3) -> float:
    """Score a parameter set across several disjoint time windows and reward
    consistency: median fold fitness minus a penalty for how much it varies.
    A config that only prints in one window (overfit) scores poorly here — this
    is the main defense against shipping fragile params that die live."""
    n = len(candles)
    if n < folds * 1500:
        st = run_backtest(candles, symbol, interval, strat, risk_cfg, spec,
                          taker_fee=taker_fee, slippage_bps=slippage_bps, collect_series=False)
        return _fitness(st.get("stats", {})) if "error" not in st else -1.0
    size = n // folds
    fits = []
    for k in range(folds):
        lo = k * size
        hi = n if k == folds - 1 else (k + 1) * size
        st = run_backtest(candles[lo:hi], symbol, interval, strat, risk_cfg, spec,
                          taker_fee=taker_fee, slippage_bps=slippage_bps, collect_series=False)
        fits.append(_fitness(st.get("stats", {})) if "error" not in st else -1.0)
    import statistics
    med = statistics.median(fits)
    sd = statistics.pstdev(fits) if len(fits) > 1 else 0.0
    worst = min(fits)
    # median, penalized for instability and for any window that fell apart
    return med - 0.4 * sd + 0.15 * worst


def run_optimizer(
    candles: list[Candle],
    symbol: str,
    interval: str,
    strat: StrategyConfig,
    risk_cfg: RiskConfig,
    spec: ContractSpec | None = None,
    taker_fee: float = 0.0005,
    slippage_bps: float = 1.5,
    n_trials: int = 40,
    seed: int | None = None,
    progress_cb=None,
) -> dict:
    """Random search with a 70/30 train/validation split; candidates are
    ranked on VALIDATION fitness so parameters that only memorized the train
    segment fall away."""
    n = len(candles)
    if n < 2000:
        return {"error": f"optimizer needs at least 2000 bars, got {n}"}
    cut = int(n * 0.7)
    train, valid = candles[:cut], candles[max(0, cut - 300):]
    rng = random.Random(seed)
    trials: list[dict] = []

    for t in range(n_trials):
        p = {}
        for k, (lo, hi, _grp, kind) in TUNABLES.items():
            v = rng.uniform(lo, hi)
            p[k] = _coerce(k, v)
        s, r = _apply_params(strat, risk_cfg, p)
        # rank on robustness across sub-windows of the train segment
        fit_t = robust_fitness(train, symbol, interval, s, r, spec, taker_fee, slippage_bps, folds=2)
        trials.append({"params": p, "train_fitness": round(fit_t, 3)})
        if progress_cb:
            progress_cb(0.75 * (t + 1) / n_trials)

    trials.sort(key=lambda x: x["train_fitness"], reverse=True)
    finalists = trials[: max(5, n_trials // 6)]
    for j, tr in enumerate(finalists):
        s, r = _apply_params(strat, risk_cfg, tr["params"])
        res_v = run_backtest(valid, symbol, interval, s, r, spec,
                             taker_fee=taker_fee, slippage_bps=slippage_bps, collect_series=False)
        tr["valid"] = res_v.get("stats", {})
        tr["valid_fitness"] = round(robust_fitness(valid, symbol, interval, s, r, spec,
                                                   taker_fee, slippage_bps, folds=2), 3)
        if progress_cb:
            progress_cb(0.75 + 0.25 * (j + 1) / len(finalists))

    finalists.sort(key=lambda x: x.get("valid_fitness", -1), reverse=True)
    best = finalists[0] if finalists and finalists[0].get("valid_fitness", -1) > 0 else None
    return {
        "symbol": symbol,
        "interval": interval,
        "n_trials": n_trials,
        "train_bars": cut,
        "valid_bars": n - cut,
        "finalists": finalists,
        "best": best,
        "note": "ranked by validation fitness; apply best params from the UI if they hold up",
    }
