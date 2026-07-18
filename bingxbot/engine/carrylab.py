"""Carry Lab: measure the funding-carry edge on REAL history before trusting it.

The live desk was shipped on a sound thesis; this module replaces the thesis
with numbers. It replays the desk's exact rules over historical funding prints
plus a 1h price path — entries at settlement times when |APR| clears the bar
(trend-vetoed), funding collected at every settlement held, ATR stops checked
against 1h bar extremes, exits on normalization / flip / trend turn / max
hold — and reports what the strategy actually earned, per symbol and across a
threshold grid, so `min_apr` / `exit_apr` are chosen by evidence.

Pure functions; the orchestrator wraps them in a job. With no exchange access
a synthetic funding series (squeeze episodes over a random walk) keeps the lab
and its tests runnable offline.
"""
from __future__ import annotations

import math
import random

from ..exchange.models import LONG, SHORT, Candle
from .scanner import annualize_funding, trend_read_4h

import numpy as np

FUNDING_MS = 8 * 3600 * 1000
H1_MS = 3600 * 1000


def _trend_at(closes: np.ndarray, i: int) -> dict:
    """4h-ish trend read from the 1h closes up to bar i (aggregated by 4)."""
    window = closes[max(0, i - 320):i + 1:4]
    return trend_read_4h(window)


def replay_carry(funding: list[dict], candles_1h: list[Candle],
                 min_apr: float, exit_apr: float, stop_atr: float = 2.5,
                 max_hold_h: int = 30, trend_veto_er: float = 0.35,
                 fee_rt: float = 0.001) -> dict:
    """Replay the carry desk on one symbol's funding + 1h price history.
    Returns funding collected, price PnL, fees (all in return-on-notional
    units), trade count and worst excursion — the honest ledger of the edge."""
    if len(candles_1h) < 60 or not funding:
        return {"entries": 0, "funding_ret": 0.0, "price_ret": 0.0, "fees": 0.0,
                "net": 0.0, "worst": 0.0, "avg_hold_h": 0.0, "wins": 0}
    closes = np.array([c.close for c in candles_1h])
    ts0 = candles_1h[0].ts

    def bar_at(ts: int) -> int:
        i = int((ts - ts0) // H1_MS)
        return min(max(i, 0), len(candles_1h) - 1)

    atr_pct = float(np.mean(np.abs(np.diff(closes[-200:]) / closes[-200:-1]))) * 1.6 or 0.004

    entries = wins = 0
    funding_ret = price_ret = fees = 0.0
    worst = 0.0
    hold_hours: list[float] = []
    pos = None    # {side, entry_px, stop, opened_ts, collected}

    for k, f in enumerate(funding):
        ts, rate = f["ts"], f["rate"]
        i = bar_at(ts)
        if i < 40:
            continue
        apr = annualize_funding(rate)
        tr = _trend_at(closes, i)

        if pos is not None:
            # 1) stop check on bar extremes since the last print
            j0 = bar_at(pos["last_ts"])
            stopped = False
            for j in range(j0, i + 1):
                c = candles_1h[j]
                if (pos["side"] == LONG and c.low <= pos["stop"]) or \
                   (pos["side"] == SHORT and c.high >= pos["stop"]):
                    px = pos["stop"]
                    ret = (px - pos["entry_px"]) / pos["entry_px"] * (1 if pos["side"] == LONG else -1)
                    price_ret += ret
                    worst = min(worst, ret)
                    fees += fee_rt
                    hold_hours.append((candles_1h[j].ts - pos["opened_ts"]) / 3_600_000)
                    pos = None
                    stopped = True
                    break
            if stopped:
                continue
            # 2) collect this settlement (receiving if the CURRENT rate pays our side)
            recv_side = SHORT if rate > 0 else LONG
            transfer = abs(rate) * (1.0 if recv_side == pos["side"] else -1.0)
            funding_ret += transfer
            pos["collected"] += transfer
            pos["last_ts"] = ts
            # 3) strategy exits at the print
            held_h = (ts - pos["opened_ts"]) / 3_600_000
            side_d = 1 if pos["side"] == LONG else -1
            flipped = rate != 0 and recv_side != pos["side"]
            turned = tr["dir"] * side_d < 0 and tr["er"] >= trend_veto_er
            if held_h >= max_hold_h or abs(apr) < exit_apr or flipped or turned:
                px = closes[i]
                ret = (px - pos["entry_px"]) / pos["entry_px"] * side_d
                price_ret += ret
                worst = min(worst, ret)
                fees += fee_rt
                hold_hours.append(held_h)
                if ret + pos["collected"] - fee_rt > 0:
                    wins += 1
                pos = None
            continue

        # flat: enter at the print when the payment clears the bar
        if abs(apr) < min_apr:
            continue
        side = SHORT if rate > 0 else LONG
        side_d = 1 if side == LONG else -1
        if tr["dir"] * side_d < 0 and tr["er"] >= trend_veto_er:
            continue
        px = closes[i]
        stop = px * (1 - stop_atr * atr_pct) if side == LONG else px * (1 + stop_atr * atr_pct)
        pos = {"side": side, "entry_px": px, "stop": stop,
               "opened_ts": ts, "last_ts": ts, "collected": 0.0}
        entries += 1

    if pos is not None:  # close at history end
        px = closes[-1]
        side_d = 1 if pos["side"] == LONG else -1
        ret = (px - pos["entry_px"]) / pos["entry_px"] * side_d
        price_ret += ret
        worst = min(worst, ret)
        fees += fee_rt
        hold_hours.append((candles_1h[-1].ts - pos["opened_ts"]) / 3_600_000)
        if ret + pos["collected"] - fee_rt > 0:
            wins += 1

    net = funding_ret + price_ret - fees
    return {
        "entries": entries, "wins": wins,
        "funding_ret": round(funding_ret, 5), "price_ret": round(price_ret, 5),
        "fees": round(fees, 5), "net": round(net, 5), "worst": round(worst, 5),
        "avg_hold_h": round(sum(hold_hours) / len(hold_hours), 1) if hold_hours else 0.0,
    }


GRID_MIN_APR = (0.20, 0.35, 0.50, 0.80)
GRID_EXIT_APR = (0.05, 0.10, 0.20)


def grid_search(funding: list[dict], candles_1h: list[Candle]) -> list[dict]:
    """The desk's thresholds swept over a small grid — evidence, not taste."""
    out = []
    for min_apr in GRID_MIN_APR:
        for exit_apr in GRID_EXIT_APR:
            if exit_apr >= min_apr:
                continue
            r = replay_carry(funding, candles_1h, min_apr, exit_apr)
            r.update({"min_apr": min_apr, "exit_apr": exit_apr})
            out.append(r)
    return out


def recommend(grids: list[list[dict]]) -> dict | None:
    """Aggregate per-symbol grids and pick the combo with the best worst-case-
    tempered net (net + worst excursion) summed across symbols, requiring
    activity — a threshold that never trades measures nothing."""
    agg: dict[tuple, dict] = {}
    for grid in grids:
        for r in grid:
            key = (r["min_apr"], r["exit_apr"])
            a = agg.setdefault(key, {"net": 0.0, "worst": 0.0, "entries": 0,
                                     "min_apr": key[0], "exit_apr": key[1]})
            a["net"] += r["net"]
            a["worst"] = min(a["worst"], r["worst"])
            a["entries"] += r["entries"]
    cands = [a for a in agg.values() if a["entries"] > 0]
    if not cands:
        return None
    best = max(cands, key=lambda a: a["net"] + a["worst"])
    best["net"] = round(best["net"], 5)
    return best


def synthetic_funding(days: int = 60, seed: int | None = None, start_ts: int = 0) -> list[dict]:
    """Offline stand-in: mostly-benign funding with occasional squeeze episodes
    that decay over a few prints — the shape real squeezes have."""
    rng = random.Random(seed)
    n = days * 3
    out, ts = [], start_ts
    episode = 0.0
    for _ in range(n):
        if episode == 0.0 and rng.random() < 0.06:
            episode = rng.uniform(0.0008, 0.003) * rng.choice([1, -1])
        rate = episode + rng.gauss(0.0001, 0.00008)
        episode *= 0.6 if abs(episode) > 1e-9 else 0.0
        if abs(episode) < 0.0002:
            episode = 0.0
        out.append({"ts": ts, "rate": round(rate, 6)})
        ts += FUNDING_MS
    return out
