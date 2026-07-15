"""Adaptive exit engine.

The old fixed ATR stop / fixed ATR target gave a ~1.1:1 reward:risk, which
cannot survive taker fees — losers cost more than 1R and winners barely clear
costs, so expectancy is negative no matter the entry.

This replaces it with a trend-follower's asymmetric structure so a few large
winners pay for many small losers:

- **Initial stop** sits at recent structure (Donchian swing), clamped between
  `sl_atr_min` and `sl_atr_max` × ATR — meaningful, not arbitrary, bounded.
- **No fixed target by default** — winners are ridden with a volatility
  *chandelier* trailing stop that widens in clean trends (high Kaufman
  efficiency ratio) and tightens in chop.
- **Breakeven** once the trade is +`be_rr` R.
- **Give-back lock**: after a big move, exit if it retraces too much of its peak.
- **Edge-flip exit**: the brain re-scores every bar; if the fused edge turns
  against the position with conviction, we exit — *the algorithm decides whether
  the move continues or is done*, rather than waiting for a static stop.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import RiskConfig
from ..exchange.models import LONG, Position
from ..strategy.regime import REGIME_EXIT_MULT
from ..util import clamp


@dataclass
class Bracket:
    stop: float
    take_profit: float
    init_risk: float     # price distance to the initial stop = 1R


class AdaptiveExitManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg

    # ------------------------------------------------------------- entry

    def initial_bracket(self, entry: float, side: str, atr: float, row: dict, regime: str,
                        style: str = "trend") -> Bracket | None:
        if entry <= 0 or atr <= 0:
            return None
        cfg = self.cfg
        d = 1 if side == LONG else -1
        if style == "scalp":
            # tight, symmetric-ish: a passive maker target and a close stop.
            dist = cfg.scalp_sl_atr * atr
            stop = entry - d * dist
            tp = entry + d * cfg.scalp_tp_atr * atr
            return Bracket(stop=stop, take_profit=tp, init_risk=dist)
        # trend: structure stop, let it run (no fixed target)
        ex = REGIME_EXIT_MULT.get(regime, {"sl": 1.0, "tp": 1.0})
        lo, hi = cfg.sl_atr_min * ex["sl"] * atr, cfg.sl_atr_max * ex["sl"] * atr
        if side == LONG:
            swing = row.get("dc_lo", entry - lo)
            struct_dist = entry - swing
        else:
            swing = row.get("dc_hi", entry + lo)
            struct_dist = swing - entry
        dist = clamp(struct_dist, lo, hi)
        stop = entry - d * dist
        tp = entry + d * cfg.tp_atr_cap * atr if cfg.tp_atr_cap > 0 else 0.0
        return Bracket(stop=stop, take_profit=tp, init_risk=dist)

    def attach(self, pos: Position, atr: float, init_risk: float) -> None:
        pos.atr_ref = atr
        pos.init_risk = init_risk
        pos.peak_price = pos.entry_price

    # ------------------------------------------------------------- per-bar

    def manage(self, pos: Position, price: float, high: float, low: float, atr: float,
               row: dict, edge: float, threshold: float, regime: str, bars_held: int
               ) -> tuple[bool, str | None]:
        """Update the trailing stop and decide whether to exit on this close.
        Returns (stop_moved, exit_reason_or_None)."""
        cfg = self.cfg
        d = pos.direction()
        atr = atr if atr > 0 else pos.atr_ref
        risk = pos.init_risk if pos.init_risk > 0 else abs(pos.entry_price - pos.stop_price)
        if risk <= 0:
            return False, None

        # track best excursion (chandelier anchor)
        fav = high if d > 0 else low
        if pos.peak_price == 0:
            pos.peak_price = pos.entry_price
        if (fav - pos.peak_price) * d > 0:
            pos.peak_price = fav

        gain = (price - pos.entry_price) * d
        rr = gain / risk

        # 1) edge-flip exit — the brain says the move reversed
        if edge * d <= -cfg.hold_edge_frac * threshold and abs(edge) > 0.15:
            return False, f"edge reversed {edge:+.2f}"

        # scalp: the passive maker target is the primary exit (handled intrabar
        # by the fill model). Here we just run a tight stop, breakeven and a
        # short time-stop; no chandelier riding.
        if pos.style == "scalp":
            m2 = False
            if not pos.breakeven_moved and rr >= 0.6:
                be = pos.entry_price + d * cfg.be_offset_atr * atr
                if (be - pos.stop_price) * d > 0:
                    pos.stop_price = be
                    pos.breakeven_moved = True
                    m2 = True
            if bars_held >= cfg.scalp_time_stop:
                return m2, f"scalp time stop ({bars_held})"
            return m2, None

        moved = False
        # 2) breakeven once we're up be_rr
        if not pos.breakeven_moved and rr >= cfg.be_rr:
            be = pos.entry_price + d * cfg.be_offset_atr * atr
            if (be - pos.stop_price) * d > 0:
                pos.stop_price = be
                pos.breakeven_moved = True
                moved = True

        # 3) profit-scaled chandelier trail: wide early so a trend can breathe,
        #    ratcheting tighter as the trade gains so a runner banks its move
        #    instead of round-tripping back to breakeven.
        er = row.get("eff_ratio", 0.3)
        ex = REGIME_EXIT_MULT.get(regime, {"trail": 1.0})
        k_base = cfg.trail_atr_min + (cfg.trail_atr_max - cfg.trail_atr_min) * clamp(er / 0.5, 0, 1)
        tighten = 1.0 - cfg.trail_tighten * clamp((rr - cfg.be_rr) / 3.0, 0.0, 1.0)
        k = k_base * ex["trail"] * tighten
        chand = pos.peak_price - d * k * atr
        if pos.breakeven_moved and (chand - pos.stop_price) * d > 0:
            pos.stop_price = chand
            moved = True

        # 4) give-back lock: protect a large open profit
        if rr >= cfg.giveback_rr:
            peak_gain = (pos.peak_price - pos.entry_price) * d
            retrace = (pos.peak_price - price) * d
            if peak_gain > 0 and retrace / peak_gain >= cfg.giveback_frac:
                return False, f"giveback {rr:.1f}R"

        # 5) long time backstop
        if bars_held >= cfg.time_stop_bars:
            return False, f"time stop ({bars_held})"

        return moved, None
