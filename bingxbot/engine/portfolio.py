"""Portfolio accounting shared by paper and live modes: positions, closed
trades, equity curve, and performance statistics."""
from __future__ import annotations

import math
from collections import deque

from ..exchange.models import Position, TradeRecord


class Portfolio:
    def __init__(self, starting_balance: float, mode: str = "paper"):
        self.mode = mode
        self.starting_balance = starting_balance
        self.cash = starting_balance          # realized balance (paper)
        self.funding_paid = 0.0               # cumulative funding transfers (+paid/-received)
        self.live_equity: float | None = None  # authoritative equity in live mode
        self.positions: dict[str, Position] = {}
        self.trades: list[TradeRecord] = []
        self.equity_curve: deque[tuple[int, float]] = deque(maxlen=6000)
        self._last_curve_ts = 0

    # ------------------------------------------------------------- equity

    def equity(self, marks: dict[str, float] | None = None) -> float:
        if self.mode == "live" and self.live_equity is not None:
            return self.live_equity
        eq = self.cash
        if marks:
            for sym, pos in self.positions.items():
                m = marks.get(sym, 0.0)
                if m > 0:
                    eq += pos.unrealized(m)
        return eq

    def record_equity(self, ts: int, marks: dict[str, float] | None = None,
                      min_gap_ms: int = 5_000) -> None:
        if ts - self._last_curve_ts >= min_gap_ms:
            self.equity_curve.append((ts, round(self.equity(marks), 6)))
            self._last_curve_ts = ts

    # ------------------------------------------------------------- trades

    def open_position(self, pos: Position, entry_fee: float) -> None:
        self.positions[pos.symbol] = pos
        self.cash -= entry_fee

    def charge_funding(self, amount: float) -> None:
        """Perp funding transfer while holding (+ = paid out, - = received).
        Backtests charge an assumed rate at every 8h boundary; the carry desk
        credits real received funding here."""
        self.cash -= amount
        self.funding_paid += amount

    def close_position(self, symbol: str, exit_price: float, exit_ts: int,
                       exit_fee: float, reason: str, planned_risk: float = 0.0) -> TradeRecord | None:
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return None
        gross = (exit_price - pos.entry_price) * pos.qty * pos.direction()
        fees = pos.entry_fee + exit_fee
        pnl = gross - exit_fee  # entry fee already deducted from cash at open
        self.cash += pnl
        net = gross - fees
        # excursions in R: how far it went against us (MAE) and the best it
        # ever looked (MFE) — the raw material for tuning stops and exits.
        d = pos.direction()
        risk = pos.init_risk if pos.init_risk > 0 else planned_risk
        mae_r = mfe_r = 0.0
        if risk > 0:
            if pos.trough_price > 0:
                mae_r = max(0.0, (pos.entry_price - pos.trough_price) * d / risk)
            if pos.peak_price > 0:
                mfe_r = max(0.0, (pos.peak_price - pos.entry_price) * d / risk)
        tr = TradeRecord(
            symbol=symbol, side=pos.side, qty=pos.qty,
            entry_price=pos.entry_price, exit_price=exit_price,
            entry_ts=pos.opened_ts, exit_ts=exit_ts,
            pnl=round(net, 8), fees=round(fees, 8),
            reason_open=pos.entry_reason, reason_close=reason,
            r_multiple=round(net / planned_risk, 3) if planned_risk > 0 else 0.0,
            mode=self.mode,
            mae_r=round(mae_r, 3), mfe_r=round(mfe_r, 3),
        )
        self.trades.append(tr)
        return tr

    # ------------------------------------------------------------- stats

    def stats(self) -> dict:
        trades = self.trades
        n = len(trades)
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        gross_win = sum(t.pnl for t in wins)
        gross_loss = -sum(t.pnl for t in losses)
        curve = [e for _, e in self.equity_curve] or [self.starting_balance]
        peak, max_dd = curve[0], 0.0
        for e in curve:
            peak = max(peak, e)
            if peak > 0:
                max_dd = max(max_dd, (peak - e) / peak)
        rets = [t.pnl for t in trades]
        mean = sum(rets) / n if n else 0.0
        var = sum((r - mean) ** 2 for r in rets) / (n - 1) if n > 1 else 0.0
        sharpe_like = mean / math.sqrt(var) * math.sqrt(min(n, 252)) if var > 0 else 0.0
        return {
            "trades": n,
            "win_rate": round(len(wins) / n, 4) if n else 0.0,
            "profit_factor": round(gross_win / gross_loss, 3) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0),
            "expectancy": round(mean, 6),
            "avg_win": round(gross_win / len(wins), 6) if wins else 0.0,
            "avg_loss": round(-gross_loss / len(losses), 6) if losses else 0.0,
            "avg_r": round(sum(t.r_multiple for t in trades) / n, 3) if n else 0.0,
            "total_pnl": round(sum(rets), 6),
            "fees_paid": round(sum(t.fees for t in trades), 6),
            "funding_paid": round(self.funding_paid, 6),
            "max_drawdown": round(max_dd, 4),
            "sharpe_like": round(sharpe_like, 3),
            "equity": round(curve[-1], 6),
            "total_return": round(curve[-1] / self.starting_balance - 1.0, 6)
            if self.starting_balance > 0 else 0.0,
        }

    def to_dict(self, marks: dict[str, float] | None = None) -> dict:
        return {
            "mode": self.mode,
            "starting_balance": self.starting_balance,
            "equity": round(self.equity(marks), 6),
            "cash": round(self.cash, 6),
            "open_positions": {
                s: {
                    "side": p.side, "qty": p.qty, "entry": p.entry_price,
                    "stop": p.stop_price, "tp": p.take_profit,
                    "opened_ts": p.opened_ts, "leverage": p.leverage,
                    "upnl": round(p.unrealized(marks.get(s, 0.0)), 6) if marks and marks.get(s) else 0.0,
                    "reason": p.entry_reason,
                }
                for s, p in self.positions.items()
            },
            "stats": self.stats(),
        }
