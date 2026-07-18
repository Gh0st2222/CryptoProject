"""Continuous auto-tuner — the firm's always-on research desk, now a real
optimizer instead of random-restart hill-climbing.

Each cycle:
  1. A **persistent Differential Evolution** population proposes trials over the
     full tunable space (it remembers what worked across cycles and restarts).
  2. Every member + trial is scored across several **time folds in parallel** on
     a dedicated **research pool** (one fold per core), building each fold's
     indicators once and reusing them for all candidates — so a cycle that used
     to pin one core now spreads across many and finishes far sooner.
  3. Candidates are ranked by a **risk-adjusted, recency-weighted, robust** score
     (rewards profitable frequency, punishes instability), then the population's
     best is **validated out-of-sample** on the most recent held-out window with
     an overfit penalty, and promoted into the live brains only if it clearly
     beats the running champion there.
  4. Every so often the champion vault is **re-validated on fresh data** and
     stale sets are retired.

It never touches user-owned settings.
"""
from __future__ import annotations

import asyncio
import logging
import math
import random
import time

from ..config import MODE_IDLE
from ..exchange.models import ContractSpec
from ..util import clamp
from .backtest import TUNABLES
from .search import (DEOptimizer, recency_weights, robust_aggregate, score_fold,
                     validate_params)

log = logging.getLogger("autotuner")

POP_SIZE = 28
IMPROVE_MARGIN = 1.06       # OOS challenger must beat champion OOS x this
DEFLATE_K = 0.03            # margin inflation per decade of candidates tried on this
                            # OOS window — multiple-testing honesty: after thousands
                            # of shots at the same gate, a marginal "win" is luck
DEFLATE_CAP = 0.10          # never demand more than +10% extra margin
MIN_ABS_FITNESS = 0.3       # ...and be clearly profitable (positive risk-adjusted score)
OVERFIT_LAMBDA = 0.5        # penalty weight on the in-sample -> OOS drop
TOP_K_VALIDATE = 5          # validate this many training-best members OOS, keep the best generalizer
VAULT_CANDIDATES = 4        # also re-validate this many top vault champions each cycle (candidate pool, not graveyard)
STALL_REINJECT = 20         # cycles without a promotion before a diversity restart
DEMOTE_FLOOR = -1.0         # incumbent scoring below this on the TRADED basket is toxic...
DEMOTE_PATIENCE = 3         # ...for this many consecutive cycles -> stand down
LOOKBACK_DAYS = 60.0
DATA_TTL_S = 1800
GAP_FAST = 20               # cadence right after a promotion (keep hammering)
GAP_SLOW = 60               # cadence when stable
VAULT_REVAL_EVERY = 15      # re-validate the champion vault every N cycles
MIN_BARS = 3000


def _current_params(cfg) -> dict:
    p = {}
    for name, (_lo, _hi, grp, _kind) in TUNABLES.items():
        src = cfg.strategy if grp == "strategy" else cfg.risk
        p[name] = getattr(src, name)
    return p


def _default_params() -> dict:
    """The code-default baseline for every tunable — the safe harbor the
    stand-down falls back to when the whole vault has gone cold."""
    from ..config import RiskConfig, StrategyConfig
    s, r = StrategyConfig(), RiskConfig()
    return {name: getattr(s if grp == "strategy" else r, name)
            for name, (_lo, _hi, grp, _kind) in TUNABLES.items()}


def _make_folds(candles: list, f: int) -> list[list]:
    n = len(candles)
    size = max(1, n // f)
    return [candles[i * size: (n if i == f - 1 else (i + 1) * size)] for i in range(f)]


class AutoTuner:
    def __init__(self, orch):
        self.orch = orch
        self._task: asyncio.Task | None = None
        self.running = False
        self.rng = random.Random()
        self.de = DEOptimizer(pop_size=POP_SIZE, seed=self.rng.randint(0, 2**31))
        self._cache: dict[str, tuple[list, float]] = {}   # symbol -> (candles, fetched_ts)
        self._rot_idx = -1
        self.research_symbol = ""   # rotates across the top-volume board each window
        self._data_ts = 0.0
        self._scored_ts = -1.0      # data window the population was last fully scored on
        self.cycles = 0
        self.improvements = 0
        self._since_improve = 0
        self._tested_oos = 0        # candidates tried against the current OOS window
        self._champ_bad_streak = 0  # consecutive cycles the incumbent scored toxic on traded symbols
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
        await asyncio.sleep(20)
        while self.running:
            cfg = self.orch.cfg
            promoted = False
            if cfg.strategy.auto_tune and self.orch.mode != MODE_IDLE and self.orch.engine is not None:
                try:
                    promoted = await self._cycle()
                except Exception as e:  # noqa: BLE001
                    log.warning("auto-tune cycle failed: %s", e)
            gap = GAP_FAST if promoted else GAP_SLOW
            self.next_run_ts = time.time() + gap
            await asyncio.sleep(gap)

    def _universe(self) -> list[str]:
        """The research universe: the radar's ACTUAL top-10 BingX perps by 24h
        USDT volume (clean majors — no index products, no long-tail memes; see
        scanner.top_volume_universe), plus the user's own symbols. Falls back to
        the configured symbols offline."""
        sc = getattr(self.orch, "scanner", None)
        uni = list(sc.top_volume) if sc is not None and sc.top_volume else []
        for s in self.orch.cfg.symbols:
            if s not in uni:
                uni.append(s)
        return uni or list(self.orch.cfg.symbols)

    async def _get_candles(self, symbol: str) -> list:
        hit = self._cache.get(symbol)
        if hit and time.time() - hit[1] < DATA_TTL_S:
            return hit[0]
        cfg = self.orch.cfg
        candles = await self.orch._get_backtest_candles(
            symbol, cfg.strategy.interval, LOOKBACK_DAYS, cfg.feed == "synthetic", _NullJob())
        self._cache[symbol] = (candles, time.time())
        if len(self._cache) > 6:   # bound the cache to the working set
            oldest = min(self._cache, key=lambda s: self._cache[s][1])
            if oldest != symbol:
                self._cache.pop(oldest, None)
        return candles

    async def _ensure_data(self) -> list:
        """Rotate the research symbol across the universe each data window: every
        ~30 min the DE trains against a different top-volume perp, so surviving
        parameters must work on the BOARD, not on one symbol's quirks."""
        uni = self._universe()
        rotate = (not self.research_symbol
                  or self.research_symbol not in uni
                  or time.time() - self._data_ts >= DATA_TTL_S)
        if rotate:
            self._rot_idx = (self._rot_idx + 1) % len(uni)
            self.research_symbol = uni[self._rot_idx]
        candles = await self._get_candles(self.research_symbol)
        self._data_ts = self._cache[self.research_symbol][1]
        return candles

    def _valid_window(self, candles: list) -> list:
        val_cut = int(len(candles) * 0.75)
        return candles[max(0, val_cut - 400):]             # recent held-out (+ warmup lead-in)

    def _traded_symbols(self) -> list[str]:
        """What the engine is actually running right now — the set promotions
        must be judged on."""
        eng = self.orch.engine
        if eng is not None and getattr(eng, "ctx", None):
            return list(eng.ctx.keys())
        return list(self.orch.cfg.symbols)

    async def _cycle(self) -> bool:
        cfg = self.orch.cfg
        interval = cfg.strategy.interval
        candles = await self._ensure_data()
        symbol = self.research_symbol
        if len(candles) < MIN_BARS:
            return False
        spec = self.orch.specs.get(symbol, ContractSpec(symbol))
        taker, slip = spec.taker_fee, cfg.paper.slippage_bps
        strat, risk = cfg.strategy, cfg.risk

        n = len(candles)
        val_cut = int(n * 0.75)
        train = candles[:val_cut]
        valid = self._valid_window(candles)

        # validation BASKET = the symbols the engine is ACTUALLY TRADING (user
        # symbols + adopted). Training may rotate across the majors board for
        # generality, but promotion answers one question only: "is this better
        # on what we are trading right now?" — a champion brilliant on BTC and
        # toxic on an adopted symbol must not win.
        basket: list[tuple[str, list]] = []
        for tsym in self._traded_symbols()[:4]:
            try:
                tc = await self._get_candles(tsym)
                if len(tc) >= MIN_BARS:
                    basket.append((tsym, self._valid_window(tc)))
            except Exception as e:  # noqa: BLE001 — basket breadth is best-effort
                log.debug("basket data %s: %s", tsym, e)
        if not basket:
            basket = [(symbol, valid)]

        champ = _current_params(cfg)
        if not self.de.ready():
            if not self.de.load():
                self.de.seed_population(champ)
        self.de.inject(champ)

        # folds scale with the research pool: more cores -> more (finer) folds,
        # one fold per worker, indicators built once per fold.
        nf = int(clamp(self.orch.research_workers, 3, 8))
        folds = _make_folds(train, nf)
        trials = self.de.trials()
        # Only re-score the whole population when the data window changed (every
        # ~30 min) or a member is unscored (freshly injected); otherwise member
        # fitness carries forward on the same folds and we score just the trials —
        # halving the work and roughly doubling generations-per-hour in steady state.
        need_members = (self._scored_ts != self._data_ts) or any(f <= -1e8 for f in self.de.fitness)
        if self._scored_ts != self._data_ts:
            self._tested_oos = 0    # fresh OOS window -> the multiple-testing meter resets
        self._scored_ts = self._data_ts
        candidates = (list(self.de.pop) + trials) if need_members else list(trials)
        args = [(fold, symbol, interval, spec, taker, slip, strat, risk, candidates) for fold in folds]
        fold_fits = await self.orch.map_cpu(score_fold, args, research=True)
        fold_fits = [ff for ff in fold_fits if ff and len(ff) == len(candidates)]
        if not fold_fits:
            return False
        w = recency_weights(len(fold_fits))
        robust = [robust_aggregate(list(fc), w) for fc in zip(*fold_fits)]
        p = len(self.de.pop)
        if need_members:
            member_fit, trial_fit = robust[:p], robust[p:p + len(trials)]
        else:
            member_fit, trial_fit = list(self.de.fitness), robust[:len(trials)]
        self.de.select(trials, trial_fit, member_fit)
        self.de.save()

        # Evaluate EVERY live candidate on the SAME current OOS window in one
        # parallel batch and run whichever wins: a freshly-evolved DE member, a
        # champion pulled back out of the vault that STILL fits today's market, or
        # the incumbent. The vault is a candidate pool, not a graveyard — the best
        # available champion drives trading, wherever it came from.
        topk = self.de.top_k(TOP_K_VALIDATE)          # (params, train_fit)
        vault = sorted(self.orch.champions, key=lambda c: c.get("fitness", 0.0),
                       reverse=True)[:VAULT_CANDIDATES]
        cands: list[dict] = [{"source": "de", "params": p, "train_fit": tfit, "cid": None}
                             for p, tfit in topk]
        cands += [{"source": "vault", "params": c.get("params", {}), "train_fit": None, "cid": c.get("id")}
                  for c in vault]

        # every candidate × every basket symbol, one parallel batch; a candidate's
        # OOS fitness is its MEAN across the basket (must earn on the board).
        nb = len(basket)
        val_args = []
        for c in cands + [{"params": champ}]:
            for bsym, bvalid in basket:
                bspec = self.orch.specs.get(bsym, ContractSpec(bsym))
                val_args.append((c["params"], bvalid, bsym, interval, bspec,
                                 bspec.taker_fee, slip, strat, risk))
        val_res = await self.orch.map_cpu(validate_params, val_args, research=True)

        def basket_fit(idx: int) -> tuple[float, dict]:
            rs = val_res[idx * nb:(idx + 1) * nb]
            fit = sum(r["fitness"] for r in rs) / max(len(rs), 1)
            return fit, rs[0]["stats"]          # stats shown from the research symbol

        champ_fit, _ = basket_fit(len(cands))
        best, best_adj, best_stats = None, -1e18, {}
        for i, c in enumerate(cands):
            oos, stats0 = basket_fit(i)
            # DE members carry an overfit penalty (in-sample -> OOS drop); vault
            # champions are scored raw — they've already proven out-of-sample.
            adj = oos - (OVERFIT_LAMBDA * max(0.0, c["train_fit"] - oos) if c["train_fit"] is not None else 0.0)
            if c["source"] == "vault" and c["cid"]:
                self.orch.set_champion_current(c["cid"], oos, stats0)  # keep its CURRENT eval fresh
            if adj > best_adj:
                best, best_adj, best_stats = c, adj, stats0

        self.cycles += 1
        self.champion_fitness = round(champ_fit, 3)
        promoted = False
        best_params = best["params"] if best else {}
        different = bool(best) and any(abs(best_params.get(k, 0) - champ.get(k, 0)) > 1e-9 for k in champ)
        # deflated margin: the more candidates have taken a shot at THIS OOS
        # window, the more a challenger must win by — a marginal beat after
        # thousands of tries is selection bias, not signal.
        self._tested_oos += len(cands)
        margin = IMPROVE_MARGIN + min(DEFLATE_CAP, DEFLATE_K * math.log10(1 + self._tested_oos / 10))
        if different and best_adj > max(champ_fit * margin, MIN_ABS_FITNESS):
            self.orch.apply_params(best_params)
            self.improvements += 1
            promoted = True
            self._tested_oos = 0    # a promotion resets the bias meter
            vs = best_stats
            # tag the champion now driving live trades: reuse the vault entry if
            # the winner came from the vault, otherwise mint a new one.
            cid = best["cid"] if (best["source"] == "vault" and best["cid"]) \
                else self.orch.record_champion(best_params, best_adj, vs)
            self.orch.mark_champion_used(cid)
            self.history.append({
                "ts": int(time.time() * 1000),
                "from_fitness": round(champ_fit, 3), "to_fitness": round(best_adj, 3),
                "valid_wr": round(vs.get("win_rate", 0), 3), "valid_pf": round(vs.get("profit_factor", 0), 3),
                "gen": self.de.generation, "source": best["source"], "champion_id": cid,
                "params": {k: best_params[k] for k in ("base_threshold", "risk_per_trade", "sl_atr_min",
                           "trail_atr_max", "giveback_rr", "target_trades_per_hour") if k in best_params},
            })
            self.history = self.history[-25:]
            log.info("auto-tune PROMOTED (gen %d, %s): OOS %.2f -> %.2f",
                     self.de.generation, best["source"], champ_fit, best_adj)
        else:
            self.orch.save_champions()   # persist the refreshed vault current-evals

        # DEFENSIVE STAND-DOWN: the promotion gate only swaps for something
        # BETTER — it never removed something TOXIC. If the incumbent keeps
        # scoring clearly negative on the symbols we actually trade and nothing
        # beats the bar, stop trading it: fall back to the best still-positive
        # vault set, else to the code-default baseline.
        if promoted or champ_fit >= DEMOTE_FLOOR:
            self._champ_bad_streak = 0
        else:
            self._champ_bad_streak += 1
            if self._champ_bad_streak >= DEMOTE_PATIENCE:
                self._champ_bad_streak = 0
                alt = max((c for c in self.orch.champions if c.get("fitness", 0.0) > 0),
                          key=lambda c: c.get("fitness", 0.0), default=None)
                fb_params = alt["params"] if alt else _default_params()
                fb_name = "vault fallback" if alt else "baseline reset"
                if any(abs(fb_params.get(k, 0) - champ.get(k, 0)) > 1e-9 for k in champ):
                    self.orch.apply_params(fb_params)
                    if alt:
                        self.orch.mark_champion_used(alt["id"])
                    self.history.append({
                        "ts": int(time.time() * 1000),
                        "from_fitness": round(champ_fit, 3),
                        "to_fitness": round(alt.get("fitness", 0.0), 3) if alt else 0.0,
                        "valid_wr": 0.0, "valid_pf": 0.0, "gen": self.de.generation,
                        "source": f"defensive ({fb_name})",
                        "params": {k: fb_params[k] for k in ("base_threshold", "risk_per_trade",
                                   "target_trades_per_hour") if k in fb_params},
                    })
                    self.history = self.history[-25:]
                    log.warning("auto-tune STAND-DOWN: incumbent %.2f on traded basket -> %s",
                                champ_fit, fb_name)

        # diversity restart: if the population has converged without finding a
        # champion for a long time, it's stuck in an overfit basin — re-inject
        # fresh explorers so it keeps searching instead of grinding the same region.
        self._since_improve = 0 if promoted else self._since_improve + 1
        if self.de.diversity() < 0.25 and self._since_improve >= STALL_REINJECT:
            k = self.de.reinject(0.4)
            self._since_improve = 0
            log.info("auto-tune: converged without a champion -> re-injected %d explorers", k)

        if self.cycles % VAULT_REVAL_EVERY == 0:
            await self._revalidate_vault(basket, interval, slip, strat, risk)

        self.last_cycle = {
            "ts": int(time.time() * 1000), "symbol": symbol,
            "generation": self.de.generation, "population": len(self.de.pop),
            "diversity": round(self.de.diversity(), 3), "folds": len(fold_fits),
            "research_cores": self.orch.research_workers,
            "champion_fitness": round(champ_fit, 3), "best_fitness": round(best_adj, 3),
            "promoted": promoted, "candidates": len(candidates),
            "vault_candidates": len(vault), "de_candidates": len(topk),
            "champion_source": (best["source"] if best else None),
            "research_symbol": symbol,
            "basket": [s for s, _ in basket],
        }
        if self.orch._notify:
            await self.orch._notify("autotune")
        return promoted

    async def _revalidate_vault(self, basket, interval, slip, strat, risk) -> None:
        """Re-score every saved champion on the TRADED basket's freshest windows
        and refresh its CURRENT evaluation (shown next to what it was born at) —
        so 'current fitness' always means 'on what we trade today'. We DON'T drop
        the temporarily-cold — pruning ages out the never-used and protects the
        most-used, so a proven champion having a bad week survives."""
        vault = self.orch.champions
        if not vault or not basket:
            return
        nb = len(basket)
        args = []
        for c in vault:
            for bsym, bvalid in basket:
                bspec = self.orch.specs.get(bsym, ContractSpec(bsym))
                args.append((c.get("params", {}), bvalid, bsym, interval, bspec,
                             bspec.taker_fee, slip, strat, risk))
        results = await self.orch.map_cpu(validate_params, args, research=True)
        for i, c in enumerate(vault):
            rs = results[i * nb:(i + 1) * nb]
            fit = sum(r.get("fitness", 0.0) for r in rs) / max(len(rs), 1)
            self.orch.set_champion_current(c["id"], fit, rs[0].get("stats", {}))
        self.orch.prune_champions()
        log.info("vault revalidated on traded basket %s: %d champions",
                 [s for s, _ in basket], len(self.orch.champions))

    def snapshot(self) -> dict:
        return {
            "enabled": self.orch.cfg.strategy.auto_tune,
            "running": self.running,
            "cycles": self.cycles,
            "improvements": self.improvements,
            "champion_fitness": self.champion_fitness,
            "generation": self.de.generation,
            "population": len(self.de.pop),
            "research_cores": self.orch.research_workers,
            "research_symbol": self.research_symbol,
            "next_run_ts": int(self.next_run_ts * 1000),
            "last_cycle": self.last_cycle,
            "history": self.history[-12:][::-1],
        }


class _NullJob:
    progress = 0.0
