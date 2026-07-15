"""RiskManager: position sizing, exit geometry, and capital-protection gates.

Sizing is volatility-based: the quantity is chosen so that hitting the stop
loses exactly `risk_per_trade` of equity. Leverage is a *consequence* of that
size (capped), never the driver.
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field

from ..config import RiskConfig
from ..exchange.models import LONG, ContractSpec, Position, TradeRecord
from ..util import clamp, round_step

log = logging.getLogger("risk")

# Local import guard: regime tables live in strategy but risk shouldn't hard-
# depend on the strategy package import order.
from ..strategy.regime import REGIME_EXIT_MULT  # noqa: E402


@dataclass
class SizedOrder:
    qty: float
    notional: float
    leverage: int
    stop_price: float
    take_profit: float
    risk_amount: float
    size_mult: float = 1.0


@dataclass
class RiskState:
    day_key: str = ""
    day_start_equity: float = 0.0
    day_realized: float = 0.0
    consecutive_losses: int = 0
    cooldown_until: float = 0.0
    killed: bool = False
    kill_reason: str = ""
    trades_today: int = 0


class HealthGovernor:
    """Auto-correction: watches recent realized performance and drawdown and
    scales risk down when the strategy is cold, back up as it recovers. No one
    touches a setting — the machine throttles itself."""

    def __init__(self, window: int = 30):
        self.r_hist: deque[float] = deque(maxlen=window)
        self.scalar = 1.0
        self.peak_equity = 0.0
        self.drawdown = 0.0

    def on_trade(self, r_multiple: float, equity: float) -> None:
        self.r_hist.append(clamp(r_multiple, -3.0, 5.0))
        self.peak_equity = max(self.peak_equity, equity)
        self.drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        self._recompute()

    def mark_equity(self, equity: float) -> None:
        self.peak_equity = max(self.peak_equity, equity)
        if self.peak_equity > 0:
            self.drawdown = (self.peak_equity - equity) / self.peak_equity
            self._recompute()

    def _recompute(self) -> None:
        n = len(self.r_hist)
        if n < 8:
            base = 1.0
        else:
            expectancy = sum(self.r_hist) / n           # avg R over recent trades
            # map expectancy in [-0.4, +0.4] R to a risk scalar in [0.4, 1.3]
            base = clamp(1.0 + expectancy * 0.9, 0.4, 1.3)
        # drawdown brake: shrink hard as drawdown deepens
        dd_brake = clamp(1.0 - self.drawdown * 3.0, 0.3, 1.0)
        self.scalar = clamp(base * dd_brake, 0.3, 1.3)

    def snapshot(self) -> dict:
        n = len(self.r_hist)
        return {
            "scalar": round(self.scalar, 3),
            "drawdown": round(self.drawdown, 4),
            "recent_expectancy": round(sum(self.r_hist) / n, 3) if n else 0.0,
            "sample": n,
        }


class RiskManager:
    def __init__(self, cfg: RiskConfig, clock=time.time):
        self.cfg = cfg
        self.clock = clock  # injectable so backtests run on simulated time
        self.state = RiskState()
        self.health = HealthGovernor()

    # ------------------------------------------------------------- lifecycle

    def _roll_day(self, equity: float) -> None:
        day = time.strftime("%Y-%m-%d", time.gmtime(self.clock()))
        if day != self.state.day_key:
            self.state = RiskState(day_key=day, day_start_equity=equity)

    def on_trade_closed(self, trade: TradeRecord, equity: float) -> None:
        self._roll_day(equity)
        self.health.on_trade(trade.r_multiple, equity)
        st = self.state
        st.day_realized += trade.pnl
        st.trades_today += 1
        st.consecutive_losses = 0 if trade.pnl > 0 else st.consecutive_losses + 1
        if st.consecutive_losses >= self.cfg.max_consecutive_losses:
            st.cooldown_until = self.clock() + self.cfg.cooldown_minutes * 60
            st.consecutive_losses = 0
            log.warning("loss streak -> cooldown %d min", self.cfg.cooldown_minutes)
        if st.day_start_equity > 0:
            dd = -st.day_realized / st.day_start_equity
            if dd >= self.cfg.max_daily_loss_pct:
                st.killed = True
                st.kill_reason = f"daily loss limit hit ({dd*100:.1f}%)"
                log.error("KILL SWITCH: %s", st.kill_reason)

    def manual_kill(self, reason: str = "manual stop") -> None:
        self.state.killed = True
        self.state.kill_reason = reason

    def reset_kill(self) -> None:
        self.state.killed = False
        self.state.kill_reason = ""
        self.state.cooldown_until = 0.0

    # ------------------------------------------------------------- entry gate

    def can_enter(self, equity: float, open_positions: int, spread_bps: float) -> tuple[bool, str]:
        self._roll_day(equity)
        st = self.state
        if st.killed:
            return False, f"kill switch: {st.kill_reason}"
        if self.clock() < st.cooldown_until:
            return False, f"cooldown for {int(st.cooldown_until - self.clock())}s"
        if open_positions >= self.cfg.max_open_positions:
            return False, "max open positions"
        if spread_bps > self.cfg.max_spread_bps:
            return False, f"spread {spread_bps:.1f}bps > {self.cfg.max_spread_bps}bps"
        if equity <= 0:
            return False, "no equity"
        return True, "ok"

    # ------------------------------------------------------------- sizing

    def size_entry(self, equity: float, price: float, stop_dist: float, side: str,
                   spec: ContractSpec, size_mult: float = 1.0) -> SizedOrder | None:
        """Volatility-targeted leverage sizing.

        Risk-based sizing (lose `risk_per_trade` of equity if the initial stop
        hits) is expressed as a *leverage*, then clamped into the operating band
        [min_leverage, max_leverage]. Because the stop is ATR-based, this makes
        leverage rise in calm markets and fall in volatile ones (classic vol
        targeting) while per-trade risk stays ~constant. `size_mult` (Kelly x
        health) pushes conviction/health into where in the band we sit. A hard
        per-trade risk cap overrides the band in extreme volatility."""
        if price <= 0 or stop_dist <= 0 or equity <= 0:
            return None
        eff_mult = clamp(size_mult, 0.1, 2.0)
        risk_amount = equity * self.cfg.risk_per_trade * eff_mult
        lev_min, lev_max = self.cfg.min_leverage, self.cfg.max_leverage

        implied_lev = (risk_amount / stop_dist) * price / equity   # leverage risk-sizing wants
        lev = clamp(implied_lev, lev_min, lev_max)
        qty = lev * equity / price

        # hard safety cap: no single trade may risk more than max_risk_hard_pct,
        # even if that means dropping below the nominal min leverage in a storm.
        max_risk = equity * self.cfg.max_risk_hard_pct
        if qty * stop_dist > max_risk:
            qty = max_risk / stop_dist

        qty = round_step(qty, spec.qty_precision)
        if qty < spec.min_qty or qty * price < spec.min_notional_usdt:
            return None
        notional = qty * price
        leverage = int(clamp(round(notional / max(equity, 1e-9)), 1, lev_max))
        return SizedOrder(qty=qty, notional=notional, leverage=leverage,
                          stop_price=0.0, take_profit=0.0, risk_amount=qty * stop_dist,
                          size_mult=eff_mult)

    def payoff_ratio(self) -> float:
        """Assumed winner:loser ratio for Kelly, given the let-winners-run exit."""
        return self.cfg.expected_rr

    def time_stop_hit(self, bars_held: int) -> bool:
        return bars_held >= self.cfg.time_stop_bars

    # ------------------------------------------------------------- exposure

    def status(self) -> dict:
        st = self.state
        return {
            "killed": st.killed,
            "kill_reason": st.kill_reason,
            "cooldown_s": max(0, int(st.cooldown_until - self.clock())),
            "day_realized": round(st.day_realized, 4),
            "day_start_equity": round(st.day_start_equity, 2),
            "trades_today": st.trades_today,
            "consecutive_losses": st.consecutive_losses,
            "health": self.health.snapshot(),
        }
