"""Event-driven historical simulation running the SAME AlphaEnsemble and
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
from ..exchange.models import LONG, SHORT, Candle, ContractSpec
from ..risk.manager import RiskManager
from ..strategy.ensemble import AlphaEnsemble
from ..strategy.features import FeatureFrame
from ..util import clamp, interval_ms
from .portfolio import Portfolio

NO_MICRO = {"obi": 0.0, "flow": 0.0, "cvd_slope": 0.0, "spread_bps": 0.0, "ticks_per_s": 0.0}
ASSUMED_SPREAD_BPS = 1.0


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
    ens = AlphaEnsemble(
        eta=strat.hedge_eta, weight_floor=strat.weight_floor, horizon_bars=strat.horizon_bars,
        base_threshold=strat.base_threshold, threshold_adapt=strat.threshold_adapt,
        target_trades_per_hour=strat.target_trades_per_hour,
        bars_per_hour=bars_per_hour, cost_multiple=strat.cost_multiple,
    )
    sim_ts = {"v": float(ts[warmup]) / 1000.0}
    risk = RiskManager(risk_cfg, clock=lambda: sim_ts["v"])
    pf = Portfolio(starting_balance, mode="backtest")
    slip = slippage_bps / 10_000.0
    fees_rt = 2.0 * taker_fee
    rt_cost = fees_rt + (ASSUMED_SPREAD_BPS + 2 * slippage_bps) / 10_000.0

    pending_entry: dict | None = None
    bars_held = 0
    planned_risk = 0.0
    weights_timeline: list[dict] = []
    regime_counts: dict[str, int] = {}
    markers: list[dict] = []

    def fill(px: float, is_buy: bool) -> float:
        return px * (1 + slip) if is_buy else px * (1 - slip)

    def close_at(i: int, px: float, reason: str) -> None:
        nonlocal bars_held
        pos = pf.positions.get(symbol)
        if pos is None:
            return
        px = fill(px, is_buy=(pos.side == SHORT))
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

        # 1) execute entry decided on the previous close, at this bar's open
        if pending_entry is not None and pos is None:
            side = pending_entry["side"]
            px = fill(o[i], is_buy=(side == LONG))
            sized = risk.size_entry(pf.equity({symbol: px}), px, pending_entry["atr"], side, spec,
                                    pending_entry["regime"], roundtrip_cost_pct=rt_cost)
            if sized is not None:
                fee = sized.qty * px * taker_fee
                from ..exchange.models import Position
                pf.open_position(Position(
                    symbol=symbol, side=side, qty=sized.qty, entry_price=px,
                    opened_ts=int(ts[i]), leverage=sized.leverage,
                    stop_price=sized.stop_price, take_profit=sized.take_profit,
                    entry_fee=fee, entry_reason=pending_entry["reason"], entry_bar_ts=int(ts[i]),
                ), fee)
                planned_risk = sized.risk_amount
                bars_held = 0
                if collect_series:
                    markers.append({"ts": int(ts[i]), "kind": "entry", "side": side,
                                    "price": round(px, 8), "reason": pending_entry["reason"]})
            pending_entry = None
            pos = pf.positions.get(symbol)

        # 2) intrabar protective exits (stop first when both hit - pessimistic)
        if pos is not None:
            d = pos.direction()
            stop, tp = pos.stop_price, pos.take_profit
            stop_hit = stop > 0 and ((l[i] <= stop) if d > 0 else (h[i] >= stop))
            tp_hit = tp > 0 and ((h[i] >= tp) if d > 0 else (l[i] <= tp))
            if stop_hit:
                gap_through = (o[i] < stop) if d > 0 else (o[i] > stop)
                close_at(i, o[i] if gap_through else stop,
                         "stop loss" if not pos.breakeven_moved else "trailing stop")
            elif tp_hit:
                gap_through = (o[i] > tp) if d > 0 else (o[i] < tp)
                close_at(i, o[i] if gap_through else tp, "take profit")
            pos = pf.positions.get(symbol)

        # 3) evaluate ensemble on this close (grades pending alpha calls too)
        ev = ens.evaluate(row, NO_MICRO)
        regime_counts[ev["regime"]] = regime_counts.get(ev["regime"], 0) + 1

        # 4) bar-close position management
        if pos is not None:
            bars_held += 1
            if risk.time_stop_hit(bars_held):
                close_at(i, c[i], f"time stop ({bars_held} bars)")
            elif ev["score"] * pos.direction() < 0 and abs(ev["score"]) >= 0.85 * ev["threshold"]:
                close_at(i, c[i], f"opposite signal {ev['score']:+.2f}")
            else:
                risk.update_trailing(pos, c[i], row.get("atr", 0.0), ev["regime"])
        else:
            # 5) entry decision for the next open
            score = ev["score"]
            equity = pf.equity({symbol: c[i]})
            ok, _ = risk.can_enter(equity, len(pf.positions), ASSUMED_SPREAD_BPS)
            if ok:
                ok, _ = ens.entry_ok(score, row, fees_rt, ASSUMED_SPREAD_BPS, slippage_bps)
                if ok:
                    pending_entry = {
                        "side": LONG if score > 0 else SHORT,
                        "atr": row.get("atr", 0.0),
                        "regime": ev["regime"],
                        "reason": f"score {score:+.2f} thr {ev['threshold']:.2f} {ev['regime']}",
                    }

        pf.record_equity(int(ts[i]), {symbol: c[i]}, min_gap_ms=0)
        if collect_series and (i - warmup) % max(1, (n - warmup) // 160) == 0:
            weights_timeline.append({"ts": int(ts[i]), **{k: round(v, 4) for k, v in ens.weights.items()}})
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
        "ensemble": ens.snapshot(),
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

SEARCH_SPACE = {
    "base_threshold": (0.20, 0.48),
    "cost_multiple": (1.0, 2.4),
    "hedge_eta": (0.15, 0.60),
    "horizon_bars": (3, 9),
    "atr_sl_mult": (1.1, 2.4),
    "atr_tp_mult": (1.4, 3.4),
    "trail_atr_mult": (0.8, 1.8),
    "breakeven_rr": (0.5, 1.3),
    "time_stop_bars": (25, 100),
    "cost_floor_mult": (2.0, 5.0),
}


def _apply_params(strat: StrategyConfig, risk: RiskConfig, p: dict) -> tuple[StrategyConfig, RiskConfig]:
    s = StrategyConfig(**{**asdict(strat)})
    r = RiskConfig(**{**asdict(risk)})
    s.base_threshold = p["base_threshold"]
    s.cost_multiple = p["cost_multiple"]
    s.hedge_eta = p["hedge_eta"]
    s.horizon_bars = int(p["horizon_bars"])
    r.atr_sl_mult = p["atr_sl_mult"]
    r.atr_tp_mult = p["atr_tp_mult"]
    r.trail_atr_mult = p["trail_atr_mult"]
    r.breakeven_rr = p["breakeven_rr"]
    r.time_stop_bars = int(p["time_stop_bars"])
    r.cost_floor_mult = p["cost_floor_mult"]
    return s, r


def _fitness(stats: dict) -> float:
    t = stats.get("trades", 0)
    if t < 12:
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
        for k, (lo, hi) in SEARCH_SPACE.items():
            v = rng.uniform(lo, hi)
            p[k] = int(round(v)) if k in ("horizon_bars", "time_stop_bars") else round(v, 3)
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
