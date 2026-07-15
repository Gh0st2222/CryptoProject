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
ASSUMED_SPREAD_BPS = 1.0
TREND_REGIMES = {"TREND_UP", "TREND_DOWN"}


def _entry_signal_ok(brain, strat, edge: float, p_win: float, row: dict, ev: dict,
                     fees_rt: float, slippage_bps: float) -> bool:
    """Regime-appropriate entry filter. The single biggest fix for fee-drag:
    only trade where an edge actually exists — ride confirmed, efficient,
    multi-timeframe-aligned trends, fade only the tails of a range, and sit out
    choppy/volatile regimes entirely (that's where accounts quietly bleed)."""
    ok, _ = brain.entry_ok(edge, p_win, row, fees_rt, ASSUMED_SPREAD_BPS, slippage_bps)
    if not ok:
        return False
    regime = ev["regime"]
    if not strat.discipline:
        if strat.trend_align_gate and regime in TREND_REGIMES:
            align = row.get("mtf_align", 0.0)
            if align * edge < 0 and abs(align) > 0.15:
                return False
        return True
    if regime in TREND_REGIMES:
        er = row.get("eff_ratio", 0.0)
        align = row.get("mtf_align", 0.0)
        return er >= strat.min_efficiency and align * edge > 0
    if regime == "RANGE":
        if not strat.trade_range:
            return False
        pctb = row.get("bb_pctb", 0.5)
        b = strat.range_band_edge
        return (pctb < b and edge > 0) or (pctb > 1 - b and edge < 0)
    return strat.trade_volatile


def candles_to_arrays(candles: list[Candle]) -> dict[str, np.ndarray]:
    return {
        "ts": np.array([c.ts for c in candles], dtype=np.int64),
        "open": np.array([c.open for c in candles], dtype=np.float64),
        "high": np.array([c.high for c in candles], dtype=np.float64),
        "low": np.array([c.low for c in candles], dtype=np.float64),
        "close": np.array([c.close for c in candles], dtype=np.float64),
        "volume": np.array([c.volume for c in candles], dtype=np.float64),
    }


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
) -> dict:
    spec = spec or ContractSpec(symbol)
    n = len(candles)
    if n < warmup + 50:
        return {"error": f"not enough bars ({n}); need at least {warmup + 50}"}

    arrays = candles_to_arrays(candles)
    ff = FeatureFrame(arrays)
    o, h, l, c, ts = arrays["open"], arrays["high"], arrays["low"], arrays["close"], arrays["ts"]

    bars_per_hour = 3_600_000 / interval_ms(interval)
    brain = TradingBrain(
        eta=strat.hedge_eta, weight_floor=strat.weight_floor, horizon_bars=strat.horizon_bars,
        base_threshold=strat.base_threshold, threshold_adapt=strat.threshold_adapt,
        target_trades_per_hour=strat.target_trades_per_hour,
        bars_per_hour=bars_per_hour, cost_multiple=strat.cost_multiple,
        min_p_win=strat.min_p_win, kelly_fraction=strat.kelly_fraction,
    )
    sim_ts = {"v": float(ts[warmup]) / 1000.0}
    risk = RiskManager(risk_cfg, clock=lambda: sim_ts["v"])
    exits = AdaptiveExitManager(risk_cfg)
    pf = Portfolio(starting_balance, mode="backtest")
    slip = slippage_bps / 10_000.0
    maker_fee = getattr(spec, "maker_fee", 0.0002)
    is_maker = strat.entry_mode == "maker"
    entry_fee_rate = maker_fee if is_maker else taker_fee
    fees_rt = taker_fee + entry_fee_rate
    rt_cost = fees_rt + (ASSUMED_SPREAD_BPS + 2 * slippage_bps) / 10_000.0
    maker_off = strat.maker_offset_bps / 10_000.0

    pending: dict | None = None
    bars_held = 0
    planned_risk = 0.0
    weights_timeline: list[dict] = []
    regime_counts: dict[str, int] = {}
    markers: list[dict] = []

    def open_at(i: int, side: str, px: float, atr: float, regime: str, size_mult: float,
                reason: str, fee_rate: float) -> bool:
        nonlocal planned_risk, bars_held
        br = exits.initial_bracket(px, side, atr, ff.row(i), regime)
        if br is None:
            return False
        sized = risk.size_entry(pf.equity({symbol: px}), px, br.init_risk, side, spec, size_mult)
        if sized is None:
            return False
        fee = sized.qty * px * fee_rate
        pos = Position(symbol=symbol, side=side, qty=sized.qty, entry_price=px,
                       opened_ts=int(ts[i]), leverage=sized.leverage,
                       stop_price=br.stop, take_profit=br.take_profit,
                       entry_fee=fee, entry_reason=reason, entry_bar_ts=int(ts[i]))
        exits.attach(pos, atr, br.init_risk)
        pf.open_position(pos, fee)
        planned_risk = sized.risk_amount
        bars_held = 0
        if collect_series:
            markers.append({"ts": int(ts[i]), "kind": "entry", "side": side,
                            "price": round(px, 8), "reason": reason})
        return True

    def close_at(i: int, px: float, reason: str) -> None:
        nonlocal bars_held
        pos = pf.positions.get(symbol)
        if pos is None:
            return
        px = px * (1 - slip) if pos.side == LONG else px * (1 + slip)   # exits are taker
        fee = pos.qty * px * taker_fee
        tr = pf.close_position(symbol, px, int(ts[i]), fee, reason, planned_risk)
        if tr:
            risk.on_trade_closed(tr, pf.equity({symbol: px}))
            if collect_series:
                markers.append({"ts": int(ts[i]), "kind": "exit", "side": pos.side,
                                "price": round(px, 8), "pnl": round(tr.pnl, 6), "reason": reason})
        bars_held = 0

    for i in range(warmup, n):
        sim_ts["v"] = float(ts[i]) / 1000.0
        row = ff.row(i)
        pos = pf.positions.get(symbol)

        # 1) resolve a pending entry from the previous close
        if pending is not None and pos is None:
            side = pending["side"]
            if pending["mode"] == "taker":
                px = o[i] * (1 + slip) if side == LONG else o[i] * (1 - slip)
                open_at(i, side, px, pending["atr"], pending["regime"], pending["size_mult"],
                        pending["reason"], taker_fee)
                pending = None
            else:  # maker: rest post-only, fill only if price trades to the limit
                lim = pending["limit"]
                hit = (l[i] <= lim) if side == LONG else (h[i] >= lim)
                if hit:
                    open_at(i, side, lim, pending["atr"], pending["regime"], pending["size_mult"],
                            pending["reason"], maker_fee)
                    pending = None
                elif i >= pending["expires_bar"]:
                    pending = None  # unfilled -> cancelled, no trade
            pos = pf.positions.get(symbol)

        # 2) intrabar protective stop (and fixed TP if configured)
        if pos is not None:
            d = pos.direction()
            stop, tp = pos.stop_price, pos.take_profit
            if stop > 0 and ((l[i] <= stop) if d > 0 else (h[i] >= stop)):
                gap = (o[i] < stop) if d > 0 else (o[i] > stop)
                close_at(i, o[i] if gap else stop, "stop" if not pos.breakeven_moved else "trail stop")
            elif tp > 0 and ((h[i] >= tp) if d > 0 else (l[i] <= tp)):
                close_at(i, tp, "target")
            pos = pf.positions.get(symbol)

        # 3) run the full brain on this close (grades pending calls too)
        ev = brain.evaluate(row, NO_MICRO, NO_CTX)
        edge, p_win, regime = ev["edge"], ev["p_win"], ev["regime"]
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

        # 4) bar-close adaptive exit management
        if pos is not None:
            bars_held += 1
            _, exit_reason = exits.manage(pos, c[i], h[i], l[i], row.get("atr", 0.0),
                                          row, edge, ev["threshold"], regime, bars_held)
            if exit_reason:
                close_at(i, c[i], exit_reason)
        elif pending is None:
            # 5) entry decision for the next bar
            equity = pf.equity({symbol: c[i]})
            ok, _ = risk.can_enter(equity, len(pf.positions), ASSUMED_SPREAD_BPS)
            if ok and _entry_signal_ok(brain, strat, edge, p_win, row, ev, fees_rt, slippage_bps):
                kelly = brain.kelly_size_mult(p_win, risk.payoff_ratio()) if strat.use_kelly else 1.0
                size_mult = kelly * risk.health.scalar
                side = LONG if edge > 0 else SHORT
                pend = {"side": side, "atr": row.get("atr", 0.0), "regime": regime,
                        "size_mult": size_mult, "reason": f"edge {edge:+.2f} P{p_win:.0%} {regime}"}
                if is_maker:
                    pend["mode"] = "maker"
                    pend["limit"] = c[i] * (1 - maker_off) if side == LONG else c[i] * (1 + maker_off)
                    pend["expires_bar"] = i + strat.maker_wait_bars
                else:
                    pend["mode"] = "taker"
                pending = pend

        pf.record_equity(int(ts[i]), {symbol: c[i]}, min_gap_ms=0)
        if collect_series and (i - warmup) % max(1, (n - warmup) // 160) == 0:
            alloc = ev["alloc"]
            weights_timeline.append({"ts": int(ts[i]), **{d: round(alloc.get(d, 0.0), 4) for d in DESK_ORDER}})
        if progress_cb and (i - warmup) % 500 == 0:
            progress_cb((i - warmup) / (n - warmup))

    if pf.positions.get(symbol) is not None:
        close_at(n - 1, c[n - 1], "backtest end")

    stats = pf.stats()
    curve = list(pf.equity_curve)
    if len(curve) > 2000:
        step = len(curve) / 2000.0
        curve = [curve[int(k * step)] for k in range(2000)] + [curve[-1]]
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


# --------------------------------------------------------------- optimizer

# The full auto-owned parameter space the tuner searches. User-owned settings
# (symbols, feed, interval, warmup, leverage band, daily-loss, max positions,
# starting balance) are deliberately excluded. (lo, hi, target, kind)
TUNABLES: dict[str, tuple] = {
    # strategy
    "base_threshold":         (0.20, 0.46, "strategy", "float"),
    "target_trades_per_hour": (0.4, 3.0, "strategy", "float"),
    "cost_multiple":          (1.2, 3.2, "strategy", "float"),
    "hedge_eta":              (0.15, 0.60, "strategy", "float"),
    "horizon_bars":           (5, 16, "strategy", "int"),
    "min_efficiency":         (0.25, 0.50, "strategy", "float"),
    "min_p_win":              (0.48, 0.60, "strategy", "float"),
    "kelly_fraction":         (0.15, 0.60, "strategy", "float"),
    "maker_offset_bps":       (0.0, 3.0, "strategy", "float"),
    "trade_range":            (0, 1, "strategy", "bool"),
    # risk / exits
    "risk_per_trade":         (0.004, 0.014, "risk", "float"),
    "sl_atr_min":             (1.0, 2.2, "risk", "float"),
    "sl_atr_max":             (2.2, 3.8, "risk", "float"),
    "trail_atr_min":          (1.2, 2.6, "risk", "float"),
    "trail_atr_max":          (2.6, 4.6, "risk", "float"),
    "trail_tighten":          (0.30, 0.75, "risk", "float"),
    "be_rr":                  (0.5, 1.6, "risk", "float"),
    "giveback_rr":            (1.6, 3.6, "risk", "float"),
    "giveback_frac":          (0.35, 0.70, "risk", "float"),
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
        target = strat if spec[2] == "strategy" else risk
        setattr(target, name, _coerce(name, val))
    # keep dependent bounds coherent
    risk.sl_atr_max = max(risk.sl_atr_max, risk.sl_atr_min + 0.4)
    risk.trail_atr_max = max(risk.trail_atr_max, risk.trail_atr_min + 0.4)


def _apply_params(strat: StrategyConfig, risk: RiskConfig, p: dict) -> tuple[StrategyConfig, RiskConfig]:
    s = StrategyConfig(**{**asdict(strat)})
    r = RiskConfig(**{**asdict(risk)})
    apply_tunables_inplace(s, r, p)
    return s, r


def _fitness(stats: dict) -> float:
    t = stats.get("trades", 0)
    if t < 10:
        return -1.0
    pf_capped = min(stats.get("profit_factor", 0.0), 3.5)
    dd = stats.get("max_drawdown", 1.0)
    wr = stats.get("win_rate", 0.0)
    return pf_capped * math.sqrt(t) * (1.0 - clamp(dd * 2.5, 0.0, 0.9)) * (0.5 + wr)


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
        res_t = run_backtest(train, symbol, interval, s, r, spec,
                             taker_fee=taker_fee, slippage_bps=slippage_bps, collect_series=False)
        fit_t = _fitness(res_t.get("stats", {})) if "error" not in res_t else -1.0
        trials.append({"params": p, "train": res_t.get("stats", {}), "train_fitness": round(fit_t, 3)})
        if progress_cb:
            progress_cb(0.75 * (t + 1) / n_trials)

    trials.sort(key=lambda x: x["train_fitness"], reverse=True)
    finalists = trials[: max(5, n_trials // 6)]
    for j, tr in enumerate(finalists):
        s, r = _apply_params(strat, risk_cfg, tr["params"])
        res_v = run_backtest(valid, symbol, interval, s, r, spec,
                             taker_fee=taker_fee, slippage_bps=slippage_bps, collect_series=False)
        fit_v = _fitness(res_v.get("stats", {})) if "error" not in res_v else -1.0
        tr["valid"] = res_v.get("stats", {})
        tr["valid_fitness"] = round(fit_v, 3)
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
