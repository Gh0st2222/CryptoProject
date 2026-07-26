"""Diagnostic resume: one plain-text dump of everything needed to analyze the
system's state remotely — portfolio, risk, brains, 24h range context, gates,
tuner, meta-model, vault, radar, carry, journal analytics, runtime and the
effective config. Built for sharing (a support snapshot): human-readable
sections with JSON bodies, hard-capped list sizes, and NO secrets (API keys
never appear; only the has_keys flag does).

Every section builds independently — a failing section reports its error
instead of killing the report. A diagnostic tool that can crash is useless
exactly when it's needed.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict as dc_asdict

from ..config import config_public_dict
from ..data.feed import bars_overdue
from ..engine.autotuner import MIN_OOS_TRADED_BARS
from ..engine.tradability import symbol_economics
from ..util import interval_ms, now_ms

TRADES_N = 60          # recent closed trades included
JOURNAL_RAW_N = 40     # raw journal rows (with decision context)
VAULT_N = 15           # champions listed
RADAR_ROWS_N = 12
RECORD_DAYS_N = 30


def _dump(obj) -> str:
    return json.dumps(obj, indent=1, default=str)


def _fin(x, nd: int = 8):
    """Round for the report; non-finite (unfilled rolling window) -> None."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def build_report(orch) -> str:
    parts: list[str] = []

    def section(title: str, fn) -> None:
        try:
            body = fn()
        except Exception as e:  # noqa: BLE001 — a broken section must not kill the report
            body = f"ERROR building section: {e!r}"
        parts.append(f"\n{'=' * 72}\n## {title}\n{'=' * 72}\n{body}")

    eng = orch.engine

    def header():
        marks = {s: st.mark_price() for s, st in eng.feed.states.items()} if eng else {}
        eq = eng.portfolio.equity(marks) if eng else None
        start = eng.portfolio.starting_balance if eng else None
        return _dump({
            "generated_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "mode": orch.mode,
            "feed": type(eng.feed).__name__ if eng else None,
            "feed_healthy": eng.feed.healthy() if eng else None,
            "interval": orch.cfg.strategy.interval,
            "symbols": list(orch.cfg.symbols),
            "adopted": sorted(eng.adopted) if eng else [],
            "uptime_min": round((now_ms() - eng.started_ts) / 60_000, 1) if eng and eng.started_ts else 0,
            "equity": round(eq, 4) if eq is not None else None,
            "starting_balance": start,
            "session_return_pct": round((eq / start - 1) * 100, 3) if eq and start else None,
            "config_version": orch.cfg.version,
            "pending_entries": eng.pending_entries() if eng else 0,
        })

    def risk_health():
        return _dump(eng.risk.status()) if eng else "engine not running"

    def divergence():
        d = orch._divergence()
        return _dump(d) if d else "engine not running"

    def positions():
        if not eng:
            return "engine not running"
        marks = {s: st.mark_price() for s, st in eng.feed.states.items()}
        out = []
        for s, p in eng.portfolio.positions.items():
            d = dc_asdict(p)
            d["mark"] = marks.get(s, 0.0)
            d["upnl"] = round(p.unrealized(marks.get(s, 0.0)), 6) if marks.get(s) else 0.0
            d["held_min"] = round((now_ms() - p.opened_ts) / 60_000, 1)
            out.append(d)
        return _dump(out) if out else "no open positions"

    def trades():
        if not eng:
            return "engine not running"
        return _dump([dc_asdict(t) for t in eng.portfolio.trades[-TRADES_N:]])

    def journal():
        return _dump({
            "summary_all_modes": orch.journal.summary(),
            "recent_rows_with_context": orch.journal.recent(JOURNAL_RAW_N),
        })

    def brains():
        if not eng:
            return "engine not running"
        return _dump(eng.snapshot()["symbols"])

    def range24():
        """Each symbol's 24h landscape: where price sits in the day's range,
        how wide the day is, and how far the extremes are in risk units (ATR).
        The same numbers the brain's 24h features and the meta-model see."""
        if not eng:
            return "engine not running"
        marks = {s: st.mark_price() for s, st in eng.feed.states.items()}
        out = {}
        for sym, c in eng.ctx.items():
            r = c.last_row
            if not r:
                continue
            px = marks.get(sym, 0.0)
            hi, lo = _fin(r.get("hi_24h")), _fin(r.get("lo_24h"))
            width_pct = (round((hi - lo) / px * 100.0, 3)
                         if hi is not None and lo is not None and px > 0 else None)
            out[sym] = {
                "price": px,
                "hi_24h": hi,
                "lo_24h": lo,
                "range_pos": _fin(r.get("range_pos_24h"), 4),   # 0=at the low, 1=at the high
                "range_width_pct": width_pct,                   # (hi-lo)/price
                "vwap_24h": _fin(r.get("vwap_24h")),
                "vwap_dev_atr": _fin(r.get("vwap24_dev"), 3),   # (price-vwap)/ATR
                "dist_hi_atr": _fin(r.get("dist_hi_24h"), 3),   # ATRs below the day high
                "dist_lo_atr": _fin(r.get("dist_lo_24h"), 3),   # ATRs above the day low
                "atr": _fin(r.get("atr")),
            }
        return _dump(out) if out else "no evaluated rows yet (warming up)"

    def tuner():
        at = orch.autotuner
        if at is None:
            return "auto-tuner not running"
        snap = at.snapshot()
        snap["history_full"] = at.history
        return _dump(snap)

    def meta_model():
        """The learned P(win) blender straight from disk — works whether or
        not the tuner is running, and shows exactly which inputs it sees."""
        from ..ml.meta import FEATURE_NAMES, MIN_AUC, MIN_SAMPLES, get_meta
        info: dict = {
            "gates": {"min_auc": MIN_AUC, "min_samples": MIN_SAMPLES},
            "feature_count": len(FEATURE_NAMES),
            "features": list(FEATURE_NAMES),
        }
        m = get_meta()
        if m is None:
            info["model"] = None   # trains during tuner cycles once history suffices
        else:
            info["model"] = {
                "auc": round(m.auc, 4), "n_samples": m.n, "ready": m.ready,
                "blend_weight": round(m.blend_weight, 3),
                "trained_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(m.trained_ts)),
                "age_h": round((time.time() - m.trained_ts) / 3600, 1),
            }
        return _dump(info)

    def vault():
        live = orch.champion_live_stats()
        champs = sorted(orch.champions, key=lambda c: (c.get("id") == orch.active_champion_id,
                                                       c.get("fitness", 0.0)), reverse=True)[:VAULT_N]
        out = []
        for c in champs:
            e = dict(c)
            e["live"] = live.get(c.get("id"), {})
            e["active"] = c.get("id") == orch.active_champion_id
            out.append(e)
        return _dump(out) if out else "vault empty"

    def overlays():
        return _dump(orch.symbol_overlays) if orch.symbol_overlays else "no per-symbol overlays"

    def radar():
        sc = orch.scanner
        if sc is None:
            return "radar not running"
        snap = sc.snapshot()
        snap["rows"] = snap.get("rows", [])[:RADAR_ROWS_N]
        return _dump(snap)

    def carry():
        return _dump(orch.carry.snapshot()) if orch.carry is not None else "carry desk not running"

    def record():
        pf = eng.portfolio if eng else None
        snap = orch.record.snapshot(pf, pf.mode if pf else "paper")
        snap["rows"] = snap.get("rows", [])[-RECORD_DAYS_N:]
        return _dump(snap)

    def runtime():
        """Interpreter + acceleration stack, so remote analysis knows whether
        the compiled kernel and the ML head are actually in play here."""
        import platform

        import numpy
        info = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": numpy.__version__,
        }
        for mod in ("sklearn", "numba"):
            try:
                info[mod] = __import__(mod).__version__
            except Exception:  # noqa: BLE001 — optional accelerators
                info[mod] = None
        if os.getenv("BOT_NO_KERNEL", "") == "1":
            info["backtest_kernel"] = "disabled (BOT_NO_KERNEL=1) — python path"
        else:
            try:
                from ..engine.kernel import run_kernel  # noqa: F401
                info["backtest_kernel"] = "available — training folds run compiled"
            except Exception as e:  # noqa: BLE001
                info["backtest_kernel"] = f"unavailable ({type(e).__name__}: {e}) — python fallback"
        # bar-pipeline freshness — THE first thing to check when the terminal
        # looks frozen: a live-looking price with an old last bar means the
        # kline stream starved and the brain has stopped evaluating.
        eng = orch.engine
        if eng is not None:
            from ..util import now_ms as _nm
            ages = {}
            for sym in eng.ctx:
                st0 = eng.feed.states.get(sym)
                lt = st0.candles.last_ts if st0 is not None and len(st0.candles) else 0
                ages[sym] = round((_nm() - lt) / 1000.0, 1) if lt else None
            info["feed"] = {"healthy": eng.feed.healthy(),
                            "last_bar_age_s": ages,
                            "interval": orch.cfg.strategy.interval}
            rec = getattr(eng.feed, "recorder", None)
            info["tape_recorder"] = rec.stats() if rec is not None else None
        return _dump(info)

    def config():
        return _dump(config_public_dict(orch.cfg))

    def tradability():
        """What a round trip costs each symbol, in units of the risk taken.

        Everything else in this system works to raise P(win) and the payoff
        ratio. This is the other side of the same inequality and nothing in the
        signal stack can move it: fees are charged on notional, notional is
        (risk / stop distance), so the cost of trading is set by how wide the
        stop is — which is set by ATR, which is set by the bar clock and the
        symbol. A number here above ~0.4 means the symbol is asking for a win
        rate a directional system does not produce.
        """
        eng = orch.engine
        if eng is None:
            return "engine idle"
        r = orch.cfg.risk
        out = {}
        for sym in eng.ctx:
            st = eng.feed.states.get(sym)
            row = eng.ctx[sym].last_row or {}
            atr_pct = row.get("atr_pct")
            micro = st.micro_snapshot() if st is not None else {}
            spec = eng.specs.get(sym)
            fees_rt = ((spec.taker_fee if spec else orch.cfg.exchange.taker_fee)
                       + ((spec.maker_fee if spec else orch.cfg.exchange.maker_fee)
                          if orch.cfg.strategy.entry_mode == "maker"
                          else (spec.taker_fee if spec else orch.cfg.exchange.taker_fee)))
            e = symbol_economics(
                atr_pct if isinstance(atr_pct, (int, float)) else float("nan"),
                micro.get("spread_bps", float("nan")),
                fees_rt, orch.cfg.paper.slippage_bps, r.sl_atr_min,
                eng.risk.payoff_ratio("trend"))
            e["fees_roundtrip"] = round(fees_rt, 6)
            out[sym] = e
        return _dump({
            "note": ("cost_r = (fees + spread + 2 x slippage) / (sl_atr_min x atr_pct); "
                     "breakeven_win_rate = (1 + cost_r) / (1 + payoff_b). "
                     "Longer bar clocks and wider stops both raise the "
                     "denominator, which is the only free lever here."),
            "interval": orch.cfg.strategy.interval,
            "sl_atr_min": r.sl_atr_min,
            "entry_mode": orch.cfg.strategy.entry_mode,
            "symbols": out,
        })

    def self_check():
        """Contradictions and dead ends the machine can see in its own state.

        Everything else in this file is a readout: it tells you what a number
        is, not whether that number can be right. Reading a resume then means
        cross-referencing the config against the overlays against the gates by
        eye, and the things worth catching are exactly the ones that survive
        that — a symbol whose spread cap it can never clear, a brain holding a
        scalar that appears in neither the config nor the overlay ledger.
        """
        eng = orch.engine
        findings: list[dict] = []
        if eng is None:
            return "engine idle — nothing to check"

        # 1) brain scalars that match neither the global set nor the overlay
        for d in eng.param_divergence():
            findings.append({
                "level": "WARN", "check": "brain-params-diverged",
                "detail": (f"{d['symbol']} is deciding with {d['param']}="
                           f"{d['in_force']}, but the {d['source']} set says "
                           f"{d['expected']}. Nothing in the config or the "
                           f"overlay ledger shows the value it is using."),
            })

        # 2) a symbol whose book cannot clear its own spread cap is not being
        #    filtered, it is switched off — it holds a position slot, a brain
        #    and a share of the research pool while being unable to trade
        cap = orch.cfg.risk.max_spread_bps
        for sym in eng.ctx:
            st = eng.feed.states.get(sym)
            sp = st.micro_snapshot().get("spread_bps") if st is not None else None
            if sp is not None and math.isfinite(sp) and sp > cap:
                findings.append({
                    "level": "WARN", "check": "spread-cap-unreachable",
                    "detail": (f"{sym} spread is {sp:.1f}bps against a "
                               f"max_spread_bps of {cap}. Its risk gate cannot "
                               f"pass at this spread — it occupies a seat "
                               f"without being able to trade."),
                })

        # 3) bar pipeline, stated against the rule instead of raw seconds, so
        #    nobody has to remember that bars are open-stamped
        iv = interval_ms(orch.cfg.strategy.interval)
        for sym in eng.ctx:
            st = eng.feed.states.get(sym)
            lt = st.candles.last_ts if st is not None and len(st.candles) else 0
            if not lt:
                continue
            age = now_ms() - lt
            if bars_overdue(lt, now_ms(), iv):
                findings.append({
                    "level": "WARN", "check": "bar-pipeline-starved",
                    "detail": (f"{sym} last closed bar is {age/1000:.0f}s old, "
                               f"past the {3*iv/1000:.0f}s overdue line — the "
                               f"kline stream has missed at least one close."),
                })
                break

        # 4) the meta head's vote is credentialed by its own AUC, not by how
        #    much evidence the online calibrator has. On a cold brain that
        #    means the GBM decides P(win) essentially alone.
        try:
            from ..ml.meta import get_meta
            m = get_meta()
        except Exception:  # noqa: BLE001 — the ML stack is optional
            m = None
        if m is not None and m.ready and m.blend_weight >= 0.8:
            cold = [s for s, c in eng.ctx.items()
                    if getattr(c.brain.calibrator, "n", 0) == 0]
            if cold:
                findings.append({
                    "level": "INFO", "check": "meta-alone-on-cold-brains",
                    "detail": (f"meta blend weight is {m.blend_weight:.2f} (its "
                               f"cap is 0.85) while the calibrator has graded "
                               f"nothing on {', '.join(sorted(cold))}. Until "
                               f"those brains have outcomes, P(win) is close to "
                               f"the model's opinion alone."),
                })

        # 5) a research desk that has never promoted is either correctly
        #    conservative or aiming at a bar it cannot reach — either way the
        #    operator should be told, not left to notice an empty vault
        at = orch.autotuner
        lc: dict = {}
        if at is not None:
            snap = at.snapshot()
            cyc, imp = snap.get("cycles", 0), snap.get("improvements", 0)
            lc = snap.get("last_cycle") or {}
            if cyc >= 10 and not imp:
                findings.append({
                    "level": "INFO", "check": "tuner-never-promoted",
                    "detail": (f"{cyc} cycles, 0 promotions, vault holds "
                               f"{len(orch.champions)}. Last cycle judged "
                               f"{lc.get('cands_judged')} candidates and "
                               f"{lc.get('pf_passed')} cleared the portfolio "
                               f"gate against a bar of {lc.get('bar')}."),
                })

        # 6) THE BAR ITSELF. "No promotion again" reads identically whether the
        #    research desk is being appropriately strict or aiming at a number
        #    its own judge cannot produce — and the second is what actually
        #    happened here for 618 cycles. Two states are worth saying out loud:
        #    judged folds too short to carry a verdict, and a best challenger
        #    that is not merely below the bar but nowhere near it.
        traded = lc.get("oos_traded_bars")
        if isinstance(traded, (int, float)) and 0 < traded < MIN_OOS_TRADED_BARS:
            findings.append({
                "level": "WARN", "check": "judged-folds-too-short",
                "detail": (f"each out-of-sample fold trades {int(traded)} bars, "
                           f"under the {MIN_OOS_TRADED_BARS} where a fold's "
                           f"verdict starts predicting the next window. Below "
                           f"this the promotion score is noise and the bar is "
                           f"effectively unreachable — widen the lookback or "
                           f"expect no champions."),
            })
        bar, bf = lc.get("bar"), lc.get("best_fitness")
        if (isinstance(bar, (int, float)) and isinstance(bf, (int, float))
                and at is not None and at.cycles >= 25 and not at.improvements
                and bf < bar - abs(bar) - 0.5):
            findings.append({
                "level": "WARN", "check": "promotion-bar-unreachable",
                "detail": (f"the best challenger in {at.cycles} cycles scored "
                           f"{bf:+.2f} against a bar of {bar:+.2f} — not close. "
                           f"A bar nothing approaches is not selectivity, it is "
                           f"a closed gate; check the judged-fold length and the "
                           f"scale MIN_ABS_FITNESS is written on."),
            })

        if not findings:
            return "no contradictions found"
        return _dump(findings)

    parts.append("PULSE — diagnostic resume (no secrets; safe to share)")
    section("SELF-CHECK", self_check)
    section("TRADABILITY (what a round trip costs, in R)", tradability)
    section("HEADER / SESSION", header)
    section("RISK & HEALTH", risk_health)
    section("DIVERGENCE MONITOR", divergence)
    section("OPEN POSITIONS", positions)
    section(f"RECENT CLOSED TRADES (last {TRADES_N})", trades)
    section("JOURNAL ANALYTICS + RECENT DECISIONS", journal)
    section("PER-SYMBOL BRAINS (edge/gates/desks/alphas/ladder)", brains)
    section("24H RANGE CONTEXT (per symbol)", range24)
    section("AUTO-TUNER (state + full promotion history)", tuner)
    section("META-MODEL (learned P(win) blender)", meta_model)
    section(f"CHAMPION VAULT (top {VAULT_N})", vault)
    section("PER-SYMBOL OVERLAYS", overlays)
    def shadow():
        sh = getattr(orch, "shadow", None)
        if sh is None:
            return _dump({"running": False,
                          "clock_trial": orch.cfg.strategy.clock_trial,
                          "status": getattr(orch, "_shadow_status", "")})
        spf = sh.portfolio
        marks = {s: s0.mark_price() for s, s0 in sh.feed.states.items()}
        return _dump({
            "running": True,
            "clock": sh.cfg.strategy.interval,
            "champion_id": sh.active_champion_id,
            "equity": round(spf.equity(marks), 4),
            "starting_balance": spf.starting_balance,
            "stats": spf.stats(),
            "open_positions": {s: {"side": p.side, "qty": p.qty, "entry": p.entry_price}
                               for s, p in spf.positions.items()},
            "status": getattr(orch, "_shadow_status", ""),
        })

    def refusals():
        """What the entry gates turned away, graded on the brain's horizon.
        DIRECTIONAL outcome only — no exits, fees or fills are simulated — so
        read it as 'was the signal this gate discarded right or wrong?', not
        as PnL. A mean move near zero means the gate is filtering noise; a
        clearly positive mean over a real sample means it is filtering edge."""
        eng = orch.engine
        if eng is None or not hasattr(eng, "refusals"):
            return _dump({"available": False})
        return _dump(eng.refusals.snapshot())

    section("REFUSED SIGNALS (gate opportunity cost)", refusals)
    section("SHADOW CLOCK (trial-interval live paper race)", shadow)
    section("RADAR (universe + board)", radar)
    section("CARRY DESK", carry)
    section(f"TRACK RECORD (last {RECORD_DAYS_N} days)", record)
    section("RUNTIME / ACCELERATION", runtime)
    section("EFFECTIVE CONFIG (public — no keys)", config)
    return "\n".join(parts) + "\n"
