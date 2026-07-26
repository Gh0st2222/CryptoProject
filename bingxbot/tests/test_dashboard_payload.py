"""The dashboard's live views read fields off the wire, so the wire is the
contract. Two of them were added for the main-grid views and are easy to drop
in a future refactor of the snapshot builders:

  * `micro` on the HOT per-symbol payload. Order-flow state used to travel only
    on the full snapshot, which is gated behind bar/trade events — on a 15m
    signal clock that made the terminal's book reading up to fifteen minutes
    stale while the tape underneath it moved every tick.

  * `init_risk` on an open position. It is the stop distance AT ENTRY, i.e. the
    trade's 1R. The current stop is not a substitute: a trailing stop ratchets
    toward and then through entry, so |entry - stop| collapses toward zero and
    any R computed from it runs away with price standing still.
"""
import math

import pytest

from bingxbot.config import BotConfig
from bingxbot.engine.portfolio import Portfolio
from bingxbot.exchange.models import LONG, SHORT, Position

MICRO_KEYS = {"obi", "flow", "spread_bps", "cvd_slope", "ticks_per_s"}


# ------------------------------------------------------- the hot micro payload

def _engine():
    """A trader wired to the synthetic feed — enough to produce a hot frame.
    No start(): the payload shape must be right from the first push, before a
    single bar has closed, because that is when the browser first connects."""
    from bingxbot.data.feed import SyntheticFeed
    from bingxbot.engine.brokers import PaperBroker
    from bingxbot.engine.trader import TraderEngine
    from bingxbot.exchange.models import ContractSpec
    from bingxbot.risk.manager import RiskManager

    cfg = BotConfig()
    cfg.feed = "synthetic"
    cfg.symbols = ["BTC-USDT", "ETH-USDT"]
    cfg.strategy.interval = "1m"
    specs = {s: ContractSpec(s) for s in cfg.symbols}
    feed = SyntheticFeed(cfg.symbols, "1m", warmup_bars=60, speed=600.0, seed=7)
    pf = Portfolio(cfg.paper.starting_balance, mode="paper")
    broker = PaperBroker(pf, feed.states, specs, taker_fee=cfg.exchange.taker_fee,
                         slippage_bps=0.0)
    return TraderEngine(cfg, feed, broker, pf, RiskManager(cfg.risk), specs)


def test_hot_carries_order_flow_for_every_symbol():
    eng = _engine()
    hot = eng.hot()
    assert hot["symbols"], "no symbols in the hot frame"
    for sym, s in hot["symbols"].items():
        assert "micro" in s, f"{sym}: hot frame dropped micro"
        assert MICRO_KEYS <= set(s["micro"]), f"{sym}: micro keys shrank"
        for k, v in s["micro"].items():
            assert isinstance(v, (int, float)), f"{sym}.{k} is not a number"


def test_hot_micro_matches_the_full_snapshot():
    """The two channels must not disagree — the UI patches one over the other."""
    eng = _engine()
    hot, full = eng.hot(), eng.snapshot()
    for sym in hot["symbols"]:
        assert hot["symbols"][sym]["micro"] == full["symbols"][sym]["micro"]


# ------------------------------------------------------ init_risk on positions

def _pf_with(side, entry, stop, init_risk):
    pf = Portfolio(1000.0, mode="paper")
    pf.positions["X-USDT"] = Position(
        symbol="X-USDT", side=side, qty=1.0, entry_price=entry, opened_ts=0,
        stop_price=stop, init_risk=init_risk)
    return pf


def test_open_positions_expose_the_trades_1R():
    pf = _pf_with(LONG, 100.0, 98.0, 2.0)
    d = pf.to_dict({"X-USDT": 101.0})["open_positions"]["X-USDT"]
    assert d["init_risk"] == pytest.approx(2.0)
    assert math.isfinite(d["init_risk"])


def test_1R_survives_a_trail_ratcheting_past_entry():
    """The exact case the runway drew wrong: a winner whose stop has moved above
    entry. |entry - stop| says 1R is 0.5 and the trade is +8R; the truth is a
    1R of 2.0 and +2R, with the downside already locked out."""
    pf = _pf_with(LONG, 100.0, 100.5, 2.0)          # stop ABOVE entry
    d = pf.to_dict({"X-USDT": 104.0})["open_positions"]["X-USDT"]
    naive = (104.0 - d["entry"]) / abs(d["entry"] - d["stop"])
    honest = (104.0 - d["entry"]) / d["init_risk"]
    assert naive == pytest.approx(8.0)
    assert honest == pytest.approx(2.0)
    assert d["stop"] > d["entry"], "the position is risk-free and must read so"


def test_a_short_reports_the_same_way():
    pf = _pf_with(SHORT, 100.0, 99.5, 2.0)          # stop BELOW entry: locked in
    d = pf.to_dict({"X-USDT": 97.0})["open_positions"]["X-USDT"]
    assert d["init_risk"] == pytest.approx(2.0)
    assert (97.0 - d["entry"]) * -1 / d["init_risk"] == pytest.approx(1.5)


def test_an_adopted_position_without_1R_is_still_serializable():
    """Positions adopted off the exchange have no init_risk. The field must be
    present and numeric so the client's fallback is a choice, not a crash."""
    pf = _pf_with(LONG, 100.0, 98.0, 0.0)
    d = pf.to_dict({"X-USDT": 101.0})["open_positions"]["X-USDT"]
    assert d["init_risk"] == 0.0
