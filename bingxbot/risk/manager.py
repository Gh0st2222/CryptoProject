"""RiskManager: position sizing, exit geometry, and capital-protection gates.

Sizing is volatility-based: the quantity is chosen so that hitting the stop
loses exactly `risk_per_trade` of equity. Leverage is a *consequence* of that
size (capped), never the driver.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

from ..config import RiskConfig
from ..exchange.models import LONG, ContractSpec, Position, TradeRecord
from ..strategy.regime import REGIME_EXIT_MULT
from ..util import round_step

log = logging.getLogger("risk")


@dataclass
class SizedOrder:
    qty: float
    notional: float
    leverage: int
    stop_price: float
    take_profit: float
    risk_amount: float


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


class RiskManager:
    def __init__(self, cfg: RiskConfig, clock=time.time):
        self.cfg = cfg
        self.clock = clock  # injectable so backtests run on simulated time
        self.state = RiskState()

    # ------------------------------------------------------------- lifecycle

    def _roll_day(self, equity: float) -> None:
        day = time.strftime("%Y-%m-%d", time.gmtime(self.clock()))
        if day != self.state.day_key:
            self.state = RiskState(day_key=day, day_start_equity=equity)

    def on_trade_closed(self, trade: TradeRecord, equity: float) -> None:
        self._roll_day(equity)
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

    def size_entry(self, equity: float, price: float, atr_val: float, side: str,
                   spec: ContractSpec, regime: str,
                   roundtrip_cost_pct: float = 0.0) -> SizedOrder | None:
        if price <= 0 or atr_val <= 0 or equity <= 0:
            return None
        ex = REGIME_EXIT_MULT.get(regime, {"sl": 1.0, "tp": 1.0, "trail": 1.0})
        sl_dist = self.cfg.atr_sl_mult * ex["sl"] * atr_val
        tp_dist = self.cfg.atr_tp_mult * ex["tp"] * atr_val
        if sl_dist <= 0:
            return None
        # Economic floor: a target that can't pay several times the round-trip
        # cost is not worth taking. Stretch the whole geometry, keep the R:R.
        tp_floor = self.cfg.cost_floor_mult * roundtrip_cost_pct * price
        if 0 < tp_dist < tp_floor:
            scale = tp_floor / tp_dist
            tp_dist *= scale
            sl_dist *= scale

        risk_amount = equity * self.cfg.risk_per_trade
        qty = risk_amount / sl_dist
        max_notional = equity * self.cfg.max_leverage * self.cfg.max_position_notional_pct
        qty = min(qty, max_notional / price)
        qty = round_step(qty, spec.qty_precision)
        if qty < spec.min_qty or qty * price < spec.min_notional_usdt:
            return None

        notional = qty * price
        leverage = max(1, min(self.cfg.max_leverage, math.ceil(notional / max(equity * 0.9, 1e-9))))
        d = 1 if side == LONG else -1
        stop = round_step(price - d * sl_dist, spec.price_precision)
        take = round_step(price + d * tp_dist, spec.price_precision)
        return SizedOrder(qty=qty, notional=notional, leverage=leverage,
                          stop_price=stop, take_profit=take, risk_amount=qty * sl_dist)

    # ------------------------------------------------------------- exits

    def update_trailing(self, pos: Position, price: float, atr_val: float, regime: str) -> bool:
        """Advance breakeven/trailing stop. Returns True if the stop moved."""
        if atr_val <= 0 or pos.stop_price <= 0:
            return False
        ex = REGIME_EXIT_MULT.get(regime, {"trail": 1.0})
        d = pos.direction()
        moved = False
        risk = abs(pos.entry_price - pos.stop_price)
        gain = (price - pos.entry_price) * d

        if not pos.breakeven_moved and risk > 0 and gain >= self.cfg.breakeven_rr * risk:
            be = pos.entry_price + d * 0.1 * atr_val   # entry + a hair, covers fees
            if (be - pos.stop_price) * d > 0:
                pos.stop_price = be
                pos.breakeven_moved = True
                moved = True

        watermark = pos.trail_price if pos.trail_price > 0 else pos.entry_price
        if (price - watermark) * d > 0:
            pos.trail_price = price
            trail_stop = price - d * self.cfg.trail_atr_mult * ex["trail"] * atr_val
            if pos.breakeven_moved and (trail_stop - pos.stop_price) * d > 0:
                pos.stop_price = trail_stop
                moved = True
        return moved

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
        }
