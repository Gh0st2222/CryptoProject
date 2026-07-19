from bingxbot.config import RiskConfig
from bingxbot.exchange.models import LONG, SHORT, ContractSpec, Position, TradeRecord
from bingxbot.risk.manager import RiskManager
from bingxbot.strategy.exits import AdaptiveExitManager


def _spec():
    return ContractSpec("BTC-USDT", qty_precision=4, price_precision=1, min_qty=0.0001)


def test_sizing_respects_risk_budget():
    cfg = RiskConfig(risk_per_trade=0.01, max_leverage=20, max_position_notional_pct=1.0)
    rm = RiskManager(cfg)
    equity, price, stop_dist = 10_000.0, 60_000.0, 300.0
    sized = rm.size_entry(equity, price, stop_dist, LONG, _spec())
    assert sized is not None
    # loss if the initial stop hits == equity * risk_per_trade
    loss_at_stop = sized.qty * stop_dist
    assert abs(loss_at_stop - 100.0) / 100.0 < 0.05
    assert 1 <= sized.leverage <= 20


def test_bracket_geometry_long_and_short():
    ex = AdaptiveExitManager(RiskConfig())
    row = {"dc_lo": 59_700.0, "dc_hi": 60_300.0}
    lb = ex.initial_bracket(60_000.0, LONG, 150.0, row, "TREND_UP")
    assert lb is not None and lb.stop < 60_000 and lb.init_risk > 0
    sb = ex.initial_bracket(60_000.0, SHORT, 150.0, row, "TREND_DOWN")
    assert sb is not None and sb.stop > 60_000


def test_sizing_rejects_dust():
    rm = RiskManager(RiskConfig(risk_per_trade=0.0001))
    spec = ContractSpec("BTC-USDT", qty_precision=4, min_qty=0.01, min_notional_usdt=100)
    assert rm.size_entry(100.0, 60_000.0, 150.0, LONG, spec) is None


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


def test_adaptive_trail_advances_only_forward():
    ex = AdaptiveExitManager(RiskConfig(be_rr=0.5))
    pos = Position(symbol="X", side=LONG, qty=1, entry_price=100.0, opened_ts=0, stop_price=98.0)
    pos.init_risk = 2.0
    pos.peak_price = 100.0
    row = {"eff_ratio": 0.5}
    # push into profit: breakeven then chandelier trail must ratchet up, never down
    ex.manage(pos, 103.0, 103.0, 102.0, 1.0, row, 0.4, 0.3, "TREND_UP", 5)
    assert pos.stop_price >= 100.0
    s1 = pos.stop_price
    ex.manage(pos, 108.0, 108.0, 107.0, 1.0, row, 0.4, 0.3, "TREND_UP", 8)
    assert pos.stop_price >= s1
    s2 = pos.stop_price
    ex.manage(pos, 105.0, 105.0, 104.0, 1.0, row, 0.4, 0.3, "TREND_UP", 9)  # pullback
    assert pos.stop_price == s2


def test_adaptive_exit_on_edge_reversal():
    """A reversed edge with NO supporting higher-TF backdrop exits — but only
    after persisting EDGE_FLIP_BARS consecutive closes (one noisy close is not
    a reversal)."""
    ex = AdaptiveExitManager(RiskConfig(hold_edge_frac=0.7))
    pos = Position(symbol="X", side=LONG, qty=1, entry_price=100.0, opened_ts=0, stop_price=98.0)
    pos.init_risk = 2.0
    pos.peak_price = 101.0
    row = {"eff_ratio": 0.4}          # mtf_bias absent -> backdrop not supporting
    _, r1 = ex.manage(pos, 100.5, 101.0, 100.0, 1.0, row,
                      edge=-0.5, threshold=0.3, regime="TREND_UP", bars_held=3)
    assert r1 is None, "a single reversed close must not exit"
    _, r2 = ex.manage(pos, 100.5, 101.0, 100.0, 1.0, row,
                      edge=-0.5, threshold=0.3, regime="TREND_UP", bars_held=4)
    assert r2 and "reversed" in r2
