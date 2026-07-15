"""Background walk-forward auto-tuner — the firm's quant-research desk.

On a timer it pulls recent data for the primary symbol, runs the same
train/validate random search the UI optimizer uses, and — only if the best
candidate beats the *currently running* parameters on the held-out validation
slice by a clear margin — hot-swaps the new parameters into the live brains and
persists them. If nothing clears the bar, it changes nothing.

This is the "I don't want to tweak settings" contract: the machine researches
and re-tunes itself, conservatively, and reports what it did.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict

from ..config import MODE_IDLE
from ..exchange.models import ContractSpec
from .backtest import _apply_params, _fitness, run_backtest, run_optimizer

log = logging.getLogger("autotuner")

IMPROVE_MARGIN = 1.20     # new validation fitness must beat current by this factor
MIN_ABS_FITNESS = 0.5     # and clear this floor
TRIALS = 24
LOOKBACK_DAYS = 20.0


class AutoTuner:
    def __init__(self, orch):
        self.orch = orch
        self._task: asyncio.Task | None = None
        self.last_tune: dict | None = None
        self.next_run_ts = 0.0
        self.running = False

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
        # give the engine time to warm up before the first research pass
        await asyncio.sleep(90)
        while self.running:
            cfg = self.orch.cfg
            period = max(15, cfg.strategy.auto_tune_minutes) * 60
            self.next_run_ts = time.time() + period
            if cfg.strategy.auto_tune and self.orch.mode != MODE_IDLE and self.orch.engine is not None:
                try:
                    await self._tune_once()
                except Exception as e:  # noqa: BLE001
                    log.warning("auto-tune pass failed: %s", e)
            await asyncio.sleep(period)

    async def _tune_once(self) -> None:
        cfg = self.orch.cfg
        symbol = cfg.symbols[0]
        interval = cfg.strategy.interval
        synthetic = cfg.feed == "synthetic"
        candles = await self.orch._get_backtest_candles(symbol, interval, LOOKBACK_DAYS, synthetic, _NullJob())
        if len(candles) < 2500:
            log.info("auto-tune: not enough data (%d bars)", len(candles))
            return
        spec = self.orch.specs.get(symbol, ContractSpec(symbol))
        taker = spec.taker_fee
        slip = cfg.paper.slippage_bps

        # baseline: current params on the same validation slice used by the search
        n = len(candles)
        cut = int(n * 0.7)
        valid = candles[max(0, cut - 300):]
        base_res = await asyncio.to_thread(
            run_backtest, valid, symbol, interval, cfg.strategy, cfg.risk, spec,
            cfg.paper.starting_balance, taker, slip, 300, None, False)
        base_fit = _fitness(base_res.get("stats", {})) if "error" not in base_res else 0.0

        res = await asyncio.to_thread(
            run_optimizer, candles, symbol, interval, cfg.strategy, cfg.risk, spec,
            taker, slip, TRIALS, None, None)
        best = res.get("best")
        decided = "kept current"
        if best and best.get("valid_fitness", 0) > max(base_fit * IMPROVE_MARGIN, MIN_ABS_FITNESS):
            self.orch.apply_params(best["params"])
            decided = "applied new params"
            log.info("auto-tune: %s (valid fit %.2f > baseline %.2f)",
                     decided, best["valid_fitness"], base_fit)
        self.last_tune = {
            "ts": int(time.time() * 1000),
            "symbol": symbol,
            "baseline_fitness": round(base_fit, 3),
            "best_fitness": round(best.get("valid_fitness", 0), 3) if best else None,
            "decision": decided,
            "params": best.get("params") if best else None,
        }
        if self.orch._notify:
            await self.orch._notify("autotune")

    def snapshot(self) -> dict:
        return {
            "enabled": self.orch.cfg.strategy.auto_tune,
            "running": self.running,
            "next_run_ts": int(self.next_run_ts * 1000),
            "last_tune": self.last_tune,
        }


class _NullJob:
    progress = 0.0
