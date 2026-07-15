from bingxbot.config import RiskConfig
from bingxbot.exchange.models import LONG, SHORT, ContractSpec, TradeRecord
from bingxbot.risk.manager import RiskManager


def _spec():
    return ContractSpec("BTC-USDT", qty_precision=4, price_precision=1, min_qty=0.0001)


def test_sizing_respects_risk_budget():
    cfg = RiskConfig(risk_per_trade=0.01, atr_sl_mult=2.0, max_leverage=20,
                     max_position_notional_pct=1.0)
    rm = RiskManager(cfg)
    equity, price, atr = 10_000.0, 60_000.0, 150.0
    sized = rm.size_entry(equity, price, atr, LONG, _spec(), "RANGE")
    assert sized is not None
    # loss if stop hits ~= equity * risk_per_trade (regime RANGE shrinks sl by 0.9)
    loss_at_stop = sized.qty * abs(price - sized.stop_price)
    assert abs(loss_at_stop - 100.0) / 100.0 < 0.08
    assert sized.stop_price < price < sized.take_profit
    assert 1 <= sized.leverage <= 20


def test_sizing_short_geometry():
    rm = RiskManager(RiskConfig())
    sized = rm.size_entry(10_000, 60_000, 150, SHORT, _spec(), "TREND_DOWN")
    assert sized is not None
    assert sized.take_profit < 60_000 < sized.stop_price


def test_sizing_rejects_dust():
    rm = RiskManager(RiskConfig(risk_per_trade=0.0001))
    spec = ContractSpec("BTC-USDT", qty_precision=4, min_qty=0.01, min_notional_usdt=100)
    assert rm.size_entry(100.0, 60_000.0, 150.0, LONG, spec, "RANGE") is None


def _trade(pnl: float) -> TradeRecord:
    return TradeRecord(symbol="BTC-USDT", side=LONG, qty=1, entry_price=100, exit_price=100 + pnl,
                       entry_ts=0, exit_ts=1, pnl=pnl, fees=0.1, reason_open="t", reason_close="t")


def test_daily_loss_kill_switch():
    t = {"v": 1_700_000_000.0}
    rm = RiskManager(RiskConfig(max_daily_loss_pct=0.05), clock=lambda: t["v"])
    equity = 10_000.0
    rm.can_enter(equity, 0, 1.0)          # initializes day baseline
    rm.on_trade_closed(_trade(-600.0), equity - 600)
    assert rm.state.killed
    ok, why = rm.can_enter(equity - 600, 0, 1.0)
    assert not ok and "kill" in why


def test_loss_streak_cooldown_and_day_roll():
    t = {"v": 1_700_000_000.0}
    rm = RiskManager(RiskConfig(max_consecutive_losses=3, cooldown_minutes=10), clock=lambda: t["v"])
    rm.can_enter(10_000, 0, 1.0)
    for _ in range(3):
        rm.on_trade_closed(_trade(-1.0), 10_000)
    ok, why = rm.can_enter(10_000, 0, 1.0)
    assert not ok and "cooldown" in why
    t["v"] += 11 * 60
    ok, _ = rm.can_enter(10_000, 0, 1.0)
    assert ok
    t["v"] += 86_400
    rm.can_enter(10_000, 0, 1.0)
    assert rm.state.day_realized == 0.0


def test_trailing_stop_advances_only_forward():
    from bingxbot.exchange.models import Position
    rm = RiskManager(RiskConfig(breakeven_rr=0.5, trail_atr_mult=1.0))
    pos = Position(symbol="X", side=LONG, qty=1, entry_price=100.0, opened_ts=0,
                   stop_price=98.0, take_profit=110.0)
    assert rm.update_trailing(pos, 101.5, 1.0, "TREND_UP")   # breakeven move
    assert pos.stop_price >= 100.0
    s1 = pos.stop_price
    rm.update_trailing(pos, 105.0, 1.0, "TREND_UP")
    assert pos.stop_price >= s1
    s2 = pos.stop_price
    rm.update_trailing(pos, 103.0, 1.0, "TREND_UP")          # pullback: no retreat
    assert pos.stop_price == s2
