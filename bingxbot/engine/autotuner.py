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
from .search import (DEOptimizer, portfolio_folds, recency_weights,
                     robust_aggregate, score_fold, validate_params,
                     validate_params_portfolio)

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
OOS_FOLDS = 3               # purged sequential portfolio folds a champion must earn across
VAULT_CANDIDATES = 4        # also re-validate this many top vault champions each cycle (candidate pool, not graveyard)
STALL_REINJECT = 20         # cycles without a promotion before a diversity restart
DEMOTE_FLOOR = -1.0         # incumbent scoring below this on the TRADED basket is toxic...
DEMOTE_PATIENCE = 3         # ...for this many consecutive cycles -> stand down
LIVE_MIN_SAMPLE = 10        # real trades needed before live evidence can demote
LIVE_RECENT_N = 20          # judge a champion on its most recent real trades
LIVE_PF_FLOOR = 0.7         # real profit factor below this (and < 1/2 expectation) -> demote
LOOKBACK_DAYS = 60.0
DATA_TTL_S = 1800
GAP_FAST = 20               # cadence right after a promotion (keep hammering)
GAP_SLOW = 60               # cadence when stable
DUTY_CYCLE = 0.35           # research may use at most ~this fraction of wall time:
                            # the sleep stretches with the measured cycle length so
                            # a slow host isn't pinned wall-to-wall by the tuner
VAULT_REVAL_EVERY = 15      # re-validate the champion vault every N cycles
META_TRAIN_EVERY = 12       # retrain the meta-labeling model every N cycles
MIN_BARS = 3000
MIN_FOLD_BARS = 450         # a training fold below this can't produce a meaningful
                            # backtest (warmup + a real trading tail)


def _current_params(cfg) -> dict:
    p = {}
    for name, (_lo, _hi, grp, _kind) in TUNABLES.items():
        src = cfg.strategy if grp == "strategy" else cfg.risk
        p[name] = getattr(src, name)
    return p


# Brain-only scalars that may differ PER SYMBOL. Risk/exit geometry stays
# global — one account, one risk policy; but BTC and a hot adopted mid-cap
# genuinely should not share one edge threshold.
BRAIN_PARAMS = ("base_threshold", "target_trades_per_hour", "cost_multiple",
                "hedge_eta", "horizon_bars", "min_p_win", "kelly_fraction")


def select_overlays(cand_params: list[dict], per_sym_fits: list[list[float]],
                    applied_fits: list[float], applied_params: dict, syms: list[str],
                    margin: float = IMPROVE_MARGIN, floor: float = MIN_ABS_FITNESS) -> dict:
    """Pick a per-symbol brain overlay wherever some candidate CLEARLY beats the
    globally applied set on that specific symbol (clearly positive too). Pure —
    the per-symbol validation numbers already exist; this stops averaging them
    away. Returns {symbol: {params, fitness, vs}}."""
    out: dict[str, dict] = {}
    for j, sym in enumerate(syms):
        base = applied_fits[j] if j < len(applied_fits) else 0.0
        best_i, best_f = -1, -1e18
        for i in range(len(cand_params)):
            f = per_sym_fits[i][j]
            if f > best_f:
                best_i, best_f = i, f
        if best_i < 0 or best_f <= max(base * margin, floor):
            continue
        ov = {k: cand_params[best_i][k] for k in BRAIN_PARAMS if k in cand_params[best_i]}
        same = all(abs(float(ov.get(k, 0)) - float(applied_params.get(k, 0))) < 1e-9 for k in ov)
        if ov and not same:
            out[sym] = {"params": ov, "fitness": round(best_f, 3), "vs": round(base, 3)}
    return out


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
        self.last_meta: dict | None = None   # latest meta-model training result
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
            t0 = time.monotonic()
            if cfg.strategy.auto_tune and self.orch.mode != MODE_IDLE and self.orch.engine is not None:
                try:
                    promoted = await self._cycle()
                except Exception as e:  # noqa: BLE001
                    log.warning("auto-tune cycle failed: %s", e)
            cycle_s = time.monotonic() - t0
            # duty-cycle pacing: the sleep stretches with how long the cycle
            # actually took, so research never monopolizes a slow host — a 4-min
            # cycle is followed by a ~7-min breather, while a 10s cycle keeps
            # the old fast cadence.
            gap = max(GAP_FAST if promoted else GAP_SLOW,
                      cycle_s * (1.0 - DUTY_CYCLE) / DUTY_CYCLE)
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
        """The held-out recent window, with a lead-in EXACTLY equal to the
        backtester's warmup (300) so OOS trading starts precisely at the 75%
        cut. The old 400-bar lead-in started trading 100 bars early — 100 bars
        of TRAINING data silently counted toward every 'out-of-sample' score."""
        val_cut = int(len(candles) * 0.75)
        return candles[max(0, val_cut - 300):]

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
                # cold start: the vault's best sets join the gene pool so the
                # search resumes from everything already proven, not from noise.
                for c in sorted(self.orch.champions, key=lambda c: c.get("fitness", 0.0),
                                reverse=True)[:3]:
                    if c.get("params"):
                        self.de.inject(c["params"])
        self.de.inject(champ)

        # folds scale with the research pool: more cores -> more (finer) folds,
        # one fold per worker, indicators built once per fold — but never so
        # many that a fold drops below what a backtest needs (score_fold
        # returns -1 under 360 bars; on an 8-core host with minimal data that
        # used to zero out EVERY fold and turn DE selection into pure noise).
        max_folds_by_data = max(1, len(train) // MIN_FOLD_BARS)
        nf = int(clamp(min(self.orch.research_workers, max_folds_by_data), 1, 8))
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
        vault = sorted((c for c in self.orch.champions if not c.get("live_flag")),
                       key=lambda c: c.get("fitness", 0.0), reverse=True)[:VAULT_CANDIDATES]
        cands: list[dict] = [{"source": "de", "params": p, "train_fit": tfit, "cid": None}
                             for p, tfit in topk]
        cands += [{"source": "vault", "params": c.get("params", {}), "train_fit": None, "cid": c.get("id")}
                  for c in vault]
        # dedupe identical parameter sets (a converged population's top-k are
        # often clones, and a vault champion may equal a DE member): validating
        # duplicates wastes basket runs and double-counts the multiple-testing
        # meter for what is really one candidate.
        uniq: list[dict] = []
        for c in cands:
            if not any(self.orch._params_match(c["params"], u["params"]) for u in uniq):
                uniq.append(c)
        cands = uniq

        # OOS validation is now PORTFOLIO fitness across purged sequential
        # folds: every candidate runs the shared-account simulator over the
        # traded basket for each of the last K disjoint windows. Promotion
        # answers "does this make the ACCOUNT richer, consistently" — not
        # "does it flatter one symbol in one lucky window".
        cbs_full: dict[str, list] = {}
        for tsym in self._traded_symbols()[:4]:
            hit = self._cache.get(tsym)
            if hit and len(hit[0]) >= MIN_BARS:
                cbs_full[tsym] = hit[0]
        if not cbs_full:
            cbs_full = {symbol: candles}
        folds_cbs = portfolio_folds(cbs_full, k=OOS_FOLDS)
        if not folds_cbs:
            return False
        nf_oos = len(folds_cbs)
        specs_map = {s: self.orch.specs.get(s, ContractSpec(s)) for s in cbs_full}
        val_args = []
        for c in cands + [{"params": champ}]:
            for fc in folds_cbs:
                val_args.append((c["params"], fc, interval, specs_map, taker, slip, strat, risk))
        val_res = await self.orch.map_cpu(validate_params_portfolio, val_args, research=True)

        def cand_fit(idx: int) -> tuple[float, dict, list[float]]:
            rs = val_res[idx * nf_oos:(idx + 1) * nf_oos]
            fits = [r["fitness"] for r in rs]
            fit = 0.7 * sum(fits) / len(fits) + 0.3 * min(fits)
            return fit, rs[-1]["stats"], fits    # stats from the most recent fold

        champ_fit, _, champ_folds = cand_fit(len(cands))
        best, best_i, best_adj, best_stats = None, -1, -1e18, {}
        best_folds: list[float] = []
        for i, c in enumerate(cands):
            oos, stats0, fold_fits_oos = cand_fit(i)
            # DE members carry an overfit penalty (in-sample -> OOS drop); vault
            # champions are scored raw — they've already proven out-of-sample.
            adj = oos - (OVERFIT_LAMBDA * max(0.0, c["train_fit"] - oos) if c["train_fit"] is not None else 0.0)
            if c["source"] == "vault" and c["cid"]:
                self.orch.set_champion_current(c["cid"], oos, stats0)  # keep its CURRENT eval fresh
            if adj > best_adj:
                best, best_i, best_adj, best_stats = c, i, adj, stats0
                best_folds = fold_fits_oos

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
        # rank stability: beyond beating the champion's blended score, the
        # challenger must beat it in a MAJORITY of the purged folds — a set
        # that wins on average but loses most windows is one lucky window.
        beat = sum(1 for a, b in zip(best_folds, champ_folds) if a > b)
        if (different and best_adj > max(champ_fit * margin, MIN_ABS_FITNESS)
                and beat * 2 > nf_oos):
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
                "folds_beaten": f"{beat}/{nf_oos}",
                "fold_fits": [round(f, 2) for f in best_folds],
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
                alt = max((c for c in self.orch.champions
                           if c.get("fitness", 0.0) > 0 and not c.get("live_flag")),
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

        # PER-SYMBOL OVERLAYS: the per-symbol validation numbers already exist —
        # stop averaging them away. Wherever a candidate clearly beats the
        # globally applied set on one specific traded symbol, that symbol's brain
        # gets the winner's scalars as an overlay (risk geometry stays global).
        try:
            # overlays still need per-symbol evidence: one single-symbol batch
            # (cands + the applied set) on the most recent held-out window.
            nb = len(basket)
            ov_args = []
            applied_params = best_params if promoted else champ
            for c in cands + [{"params": applied_params}]:
                for bsym, bvalid in basket:
                    bspec = self.orch.specs.get(bsym, ContractSpec(bsym))
                    ov_args.append((c["params"], bvalid, bsym, interval, bspec,
                                    bspec.taker_fee, slip, strat, risk))
            ov_res = await self.orch.map_cpu(validate_params, ov_args, research=True)
            per_sym = [[ov_res[i * nb + j]["fitness"] for j in range(nb)]
                       for i in range(len(cands))]
            applied_fits = [ov_res[len(cands) * nb + j]["fitness"] for j in range(nb)]
            overlays = select_overlays([c["params"] for c in cands], per_sym,
                                       applied_fits, applied_params, [s for s, _ in basket])
            self.orch.update_symbol_overlays(overlays, traded=[s for s, _ in basket])
        except Exception as e:  # noqa: BLE001 — overlays are an optimization, never fatal
            log.warning("overlay pass failed: %s", e)

        # LIVE-EVIDENCE DEMOTION: backtests propose, live results dispose. A
        # champion with a real sample whose actual profit factor collapsed vs
        # its validation expectation stops trading NOW, whatever backtests say.
        act = self.orch.find_champion(self.orch.active_champion_id)
        if act is not None and not act.get("live_flag"):
            lv = self.orch.champion_live_stats(recent_n=LIVE_RECENT_N).get(act["id"])
            if (lv and lv["trades"] >= LIVE_MIN_SAMPLE and lv["pf"] < LIVE_PF_FLOOR
                    and lv["pf"] < 0.5 * max(act.get("profit_factor", 1.0), 1.0)):
                act["live_flag"] = {"pf": lv["pf"], "trades": lv["trades"],
                                    "ts": int(time.time() * 1000)}
                alt = max((c for c in self.orch.champions
                           if c.get("fitness", 0.0) > 0 and c["id"] != act["id"]
                           and not c.get("live_flag")),
                          key=lambda c: c.get("fitness", 0.0), default=None)
                fb = alt["params"] if alt else _default_params()
                self.orch.apply_params(fb)
                if alt is not None:
                    self.orch.mark_champion_used(alt["id"])
                else:
                    self.orch.save_champions()
                self.history.append({
                    "ts": int(time.time() * 1000),
                    "from_fitness": round(champ_fit, 3),
                    "to_fitness": round(alt.get("fitness", 0.0), 3) if alt else 0.0,
                    "valid_wr": round(lv["win_rate"], 3), "valid_pf": round(lv["pf"], 3),
                    "gen": self.de.generation,
                    "source": f"live-evidence demotion (real PF {lv['pf']:.2f} on {lv['trades']} trades)",
                    "params": {},
                })
                self.history = self.history[-25:]
                log.warning("LIVE-EVIDENCE DEMOTION: champion %s real PF %.2f over %d trades",
                            act["id"], lv["pf"], lv["trades"])

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

        # meta-labeling research: retrain the P(win) model on the basket's full
        # history every so often (walk-forward credentialed; persists only if
        # it beats the incumbent's held-out AUC). Runs on the research pool.
        if self.cycles % META_TRAIN_EVERY == 0:
            try:
                from ..ml.meta import train_from_candles
                cbs = {}
                for tsym in self._traded_symbols()[:4]:
                    tc = self._cache.get(tsym)
                    if tc and len(tc[0]) >= MIN_BARS:
                        cbs[tsym] = tc[0]
                if cbs:
                    res = (await self.orch.map_cpu(train_from_candles,
                                                   [(cbs, interval, strat, risk)],
                                                   research=True))[0]
                    self.last_meta = {**res, "ts": int(time.time() * 1000)}
                    log.info("meta-model training: %s", res)
            except Exception as e:  # noqa: BLE001 — ML must never break tuning
                log.warning("meta training failed: %s", e)

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
            "meta": self._meta_status(),
            "history": self.history[-12:][::-1],
        }

    def _meta_status(self) -> dict:
        try:
            from ..ml.meta import get_meta
            m = get_meta()
            if m is None:
                return {"model": None, "last_training": self.last_meta}
            return {"model": {"auc": round(m.auc, 4), "n": m.n, "ready": m.ready,
                              "weight": round(m.blend_weight, 3),
                              "age_h": round((time.time() - m.trained_ts) / 3600, 1)},
                    "last_training": self.last_meta}
        except Exception as e:  # noqa: BLE001
            return {"model": None, "error": str(e)}


class _NullJob:
    progress = 0.0
