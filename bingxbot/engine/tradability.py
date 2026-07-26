"""What a round trip costs, expressed in the only unit that matters: R.

Every part of this system works on the numerator of the same fraction. The 19
alphas, the meta head, the calibrator and the tuner all exist to raise P(win)
and the payoff ratio. Nothing works on the denominator — and the denominator
is where the leverage is.

A trade risks a fixed fraction of equity down to its stop, so the position's
NOTIONAL is (risk / stop distance). Fees are charged on notional. Therefore

    cost_R = (fees + spread + 2 x slippage) / (sl_atr x atr_pct)

is what one round trip costs measured in the trade's own risk unit, and the
win rate needed to break even at a payoff of b is (1 + cost_R) / (1 + b).

That is a hard floor under profitability that no signal quality can move. On a
15m clock a major with a 0.09% ATR carries cost_R ~0.68, i.e. it needs to win
55% of the time at 2.2:1 just to stand still — a bar a directional system does
not clear. The same symbol on a 4h clock carries ~0.14 and needs 38%. The
strategy did not change; the denominator did.

These are pure functions of numbers the engine already has, so the terminal
and the resume can show what each symbol actually demands instead of leaving
it to be discovered in the P&L.
"""
from __future__ import annotations

import math


def cost_in_r(atr_pct: float, spread_bps: float, fees_roundtrip: float,
              slippage_bps: float, sl_atr: float) -> float | None:
    """Round-trip cost as a multiple of the trade's own risk (1R).

    Returns None when the inputs cannot support the question — an unknown
    volatility means an unknown stop, which means an unknown risk unit.
    """
    vals = (atr_pct, spread_bps, fees_roundtrip, slippage_bps, sl_atr)
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
        return None
    stop_pct = sl_atr * atr_pct
    if stop_pct <= 0:
        return None
    cost = fees_roundtrip + (spread_bps + 2.0 * slippage_bps) / 10_000.0
    return cost / stop_pct


def breakeven_win_rate(cost_r: float, payoff_b: float) -> float | None:
    """P(win) at which expected value is exactly zero, given the cost.

    Same identity `gate_ev` enforces per trade — kept here without its safety
    margin so a readout shows the true break-even, not the gate's bar.
    """
    if not all(isinstance(v, (int, float)) and math.isfinite(v)
               for v in (cost_r, payoff_b)):
        return None
    return (1.0 + cost_r) / (1.0 + max(payoff_b, 0.1))


def symbol_economics(atr_pct: float, spread_bps: float, fees_roundtrip: float,
                     slippage_bps: float, sl_atr: float,
                     payoff_b: float) -> dict:
    """The tradability of one symbol on the clock it is currently trading."""
    cr = cost_in_r(atr_pct, spread_bps, fees_roundtrip, slippage_bps, sl_atr)
    be = breakeven_win_rate(cr, payoff_b) if cr is not None else None
    return {
        "atr_pct": round(atr_pct, 6) if isinstance(atr_pct, (int, float))
                   and math.isfinite(atr_pct) else None,
        "stop_pct": round(sl_atr * atr_pct, 6) if cr is not None else None,
        "spread_bps": round(spread_bps, 3) if isinstance(spread_bps, (int, float))
                      and math.isfinite(spread_bps) else None,
        "cost_r": round(cr, 4) if cr is not None else None,
        "breakeven_win_rate": round(be, 4) if be is not None else None,
        "payoff_b": round(payoff_b, 3) if isinstance(payoff_b, (int, float))
                    and math.isfinite(payoff_b) else None,
    }
