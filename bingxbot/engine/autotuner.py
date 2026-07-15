"""Continuous evolutionary auto-tuner — the firm's always-on research desk.

It never stops. Every cycle it:

  1. pulls a recent window of the primary symbol,
  2. builds candidates = the current champion + gaussian perturbations of it
     (exploit) + fresh random draws (explore) over the FULL tunable space,
  3. scores every candidate on a train split, validates the survivors on a
     held-out split (so overfit sets fall away),
  4. promotes a challenger into the live config ONLY if it beats the running
     champion on validation by a clear margin — otherwise it changes nothing.

It tunes the entire strategy + risk/exit parameter space but never touches the
settings the user owns (symbols, feed, interval, warmup, leverage band,
daily-loss limit, max positions, starting balance). Promotions persist to
config.json and hot-swap into the live brains without interrupting trading.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time

from ..config import MODE_IDLE
from ..exchange.models import ContractSpec
from ..util import clamp
from .backtest import (TUNABLES, _apply_params, _coerce, _fitness, apply_tunables_inplace,
                       run_backtest)

log = logging.getLogger("autotuner")

CYCLE_GAP_S = 120          # constant light background cadence
N_EVOLVE = 10              # perturbations of the champion per cycle
N_RANDOM = 8               # fresh random explorers per cycle
N_VALIDATE = 6             # top train candidates to validate
IMPROVE_MARGIN = 1.08      # challenger must beat champion validation fitness x this
MIN_ABS_FITNESS = 0.5
LOOKBACK_DAYS = 60.0       # enough trades in the validation slice to be meaningful
DATA_TTL_S = 1800          # refresh the tuning window this often


def _current_params(cfg) -> dict:
    p = {}
    for name, (_lo, _hi, grp, _kind) in TUNABLES.items():
        src = cfg.strategy if grp == "strategy" else cfg.risk
        p[name] = getattr(src, name)
    return p


def _random_params(rng: random.Random) -> dict:
    return {name: _coerce(name, rng.uniform(lo, hi)) for name, (lo, hi, _g, _k) in TUNABLES.items()}


def _perturb(champ: dict, rng: random.Random, scale: float = 0.22) -> dict:
    p = dict(champ)
    keys = list(TUNABLES)
    k = rng.randint(2, max(2, len(keys) // 2))
    for name in rng.sample(keys, k):
        lo, hi, _g, kind = TUNABLES[name]
        if kind == "bool":
            if rng.random() < 0.5:
                p[name] = not bool(p.get(name))
        else:
            p[name] = _coerce(name, clamp(p.get(name, (lo + hi) / 2) + rng.gauss(0, scale * (hi - lo)), lo, hi))
    return p


class AutoTuner:
    def __init__(self, orch):
        self.orch = orch
        self._task: asyncio.Task | None = None
        self.running = False
        self.rng = random.Random()
        self._candles: list = []
        self._data_ts = 0.0
        # observable state
        self.cycles = 0
        self.improvements = 0
        self.next_run_ts = 0.0
        self.champion_fitness = 0.0
        self.last_cycle: dict | None = None
        self.history: list[dict] = []

    def start(self) -> None:
        if self._task is None or self._task.done():
            self.running = True
            self._task = asyncio.create_task(self._loop(), name="autotuner")

    async def stop(self) -> None:
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def _loop(self) -> None:
        await asyncio.sleep(45)  # let the engine warm up first
        while self.running:
            cfg = self.orch.cfg
            if cfg.strategy.auto_tune and self.orch.mode != MODE_IDLE and self.orch.engine is not None:
                try:
                    await self._cycle()
                except Exception as e:  # noqa: BLE001
                    log.warning("auto-tune cycle failed: %s", e)
            self.next_run_ts = time.time() + CYCLE_GAP_S
            await asyncio.sleep(CYCLE_GAP_S)

    async def _ensure_data(self):
        if self._candles and time.time() - self._data_ts < DATA_TTL_S:
            return self._candles
        cfg = self.orch.cfg
        symbol, interval = cfg.symbols[0], cfg.strategy.interval
        synthetic = cfg.feed == "synthetic"
        self._candles = await self.orch._get_backtest_candles(symbol, interval, LOOKBACK_DAYS, synthetic, _NullJob())
        self._data_ts = time.time()
        return self._candles

    async def _cycle(self) -> None:
        cfg = self.orch.cfg
        symbol, interval = cfg.symbols[0], cfg.strategy.interval
        candles = await self._ensure_data()
        if len(candles) < 2500:
            return
        spec = self.orch.specs.get(symbol, ContractSpec(symbol))
        champ = _current_params(cfg)
        best, champ_valid, top = await asyncio.to_thread(
            self._run_cycle_sync, candles, symbol, interval, spec,
            spec.taker_fee, cfg.paper.slippage_bps, cfg.strategy, cfg.risk, champ)

        self.cycles += 1
        self.champion_fitness = round(champ_valid, 3)
        promoted = False
        if best["tag"] != "champion" and best["valid"] > max(champ_valid * IMPROVE_MARGIN, MIN_ABS_FITNESS):
            self.orch.apply_params(best["params"])
            promoted = True
            self.improvements += 1
            self.history.append({
                "ts": int(time.time() * 1000),
                "from_fitness": round(champ_valid, 3),
                "to_fitness": round(best["valid"], 3),
                "valid_wr": round(best["valid_stats"].get("win_rate", 0), 3),
                "valid_pf": round(best["valid_stats"].get("profit_factor", 0), 3),
                "params": {k: best["params"][k] for k in ("base_threshold", "risk_per_trade",
                           "sl_atr_min", "trail_atr_max", "giveback_rr", "target_trades_per_hour")
                           if k in best["params"]},
            })
            self.history = self.history[-25:]
            log.info("auto-tune PROMOTED: valid fit %.2f -> %.2f", champ_valid, best["valid"])
        self.last_cycle = {
            "ts": int(time.time() * 1000), "symbol": symbol,
            "champion_fitness": round(champ_valid, 3),
            "best_fitness": round(best["valid"], 3),
            "best_tag": best["tag"], "promoted": promoted,
            "candidates": len(top),
        }
        if self.orch._notify:
            await self.orch._notify("autotune")

    def _run_cycle_sync(self, candles, symbol, interval, spec, taker, slip, base_strat, base_risk, champ):
        n = len(candles)
        cut = int(n * 0.7)
        train, valid = candles[:cut], candles[max(0, cut - 300):]
        cands = [{"tag": "champion", "params": champ}]
        cands += [{"tag": "evolve", "params": _perturb(champ, self.rng)} for _ in range(N_EVOLVE)]
        cands += [{"tag": "random", "params": _random_params(self.rng)} for _ in range(N_RANDOM)]

        for c in cands:
            s, r = _apply_params(base_strat, base_risk, c["params"])
            st = run_backtest(train, symbol, interval, s, r, spec, taker_fee=taker,
                              slippage_bps=slip, collect_series=False).get("stats", {})
            c["train"] = _fitness(st)
        cands.sort(key=lambda x: x["train"], reverse=True)
        top = cands[:N_VALIDATE]
        champ_c = next(c for c in cands if c["tag"] == "champion")
        if champ_c not in top:
            top.append(champ_c)
        for c in top:
            s, r = _apply_params(base_strat, base_risk, c["params"])
            st = run_backtest(valid, symbol, interval, s, r, spec, taker_fee=taker,
                              slippage_bps=slip, collect_series=False).get("stats", {})
            c["valid"] = _fitness(st)
            c["valid_stats"] = st
        top.sort(key=lambda x: x["valid"], reverse=True)
        champ_valid = next((c["valid"] for c in top if c["tag"] == "champion"), 0.0)
        return top[0], champ_valid, top

    def snapshot(self) -> dict:
        return {
            "enabled": self.orch.cfg.strategy.auto_tune,
            "running": self.running,
            "cycles": self.cycles,
            "improvements": self.improvements,
            "champion_fitness": self.champion_fitness,
            "next_run_ts": int(self.next_run_ts * 1000),
            "last_cycle": self.last_cycle,
            "history": self.history[-12:][::-1],
        }


class _NullJob:
    progress = 0.0
