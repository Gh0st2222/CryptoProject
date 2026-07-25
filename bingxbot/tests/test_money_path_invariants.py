"""Property fuzz of the money path: position sizing and the exit bracket.

Every invariant here is money, and each is asserted against hostile input rather
than trusted:

  S1  sizing never risks more than max_risk_hard_pct of equity
  S2  sizing never returns a non-finite or non-positive quantity
  S3  the stop always sits on the LOSING side of entry
  S4  a trail may only ever TIGHTEN — a stop drifting away from price turns a
      small loss into a large one, silently
  S5  no bracket is ever handed out with a non-finite price

The bugs this found were all the same shape: a guard written as a COMPARISON.
`x <= 0` is False when x is NaN, so `if atr <= 0: return None` and
`if price <= 0 or stop_dist <= 0 or equity <= 0: return None` both failed open
on exactly the input they exist to stop.
"""
import itertools
import math
import random

import pytest

from bingxbot.config import RiskConfig
from bingxbot.exchange.models import LONG, SHORT, ContractSpec, Position
from bingxbot.risk.manager import RiskManager
from bingxbot.strategy.exits import AdaptiveExitManager

NAN, INF = float("nan"), float("inf")


def _spec():
    s = ContractSpec("X-USDT")
    s.qty_precision, s.min_qty, s.min_notional_usdt = 6, 1e-6, 1.0
    return s


PRICES = [1e-6, 1.0, 100.0, 65_000.0, 1e9, NAN, INF, -1.0, 0.0]
STOPS = [1e-9, 0.001, 0.01, 0.5, 1.5, NAN, INF, -0.01, 0.0]
EQS = [0.0, 1.0, 10_000.0, 1e7, NAN, INF, -5.0]
MULTS = [0.0, 1.0, 2.0, 100.0, NAN, INF, -1.0]


def test_sizing_refuses_bad_input_and_never_raises():
    """A sizer must REFUSE, never throw. Before the finiteness check this
    reached math.ceil(nan) and raised ValueError from inside the entry path."""
    cfg, rm = RiskConfig(), RiskManager(RiskConfig())
    n = 0
    for px, sf, eq, mult in itertools.product(PRICES, STOPS, EQS, MULTS):
        for side in (LONG, SHORT):
            n += 1
            so = rm.size_entry(eq, px, px * sf, side, _spec(), size_mult=mult)   # must not raise
            if so is None:
                continue
            assert math.isfinite(so.qty) and so.qty > 0, f"S2 qty={so.qty}"
            assert math.isfinite(so.notional) and so.notional >= 0, "S2 notional"
            assert isinstance(so.leverage, int) and so.leverage >= 1, "S2 leverage"
            if math.isfinite(eq) and eq > 0:
                risked = so.qty * px * sf
                cap = eq * cfg.max_risk_hard_pct
                assert risked <= cap * 1.001 + 1e-9, \
                    f"S1 risked {risked} > cap {cap} (px={px} sf={sf} eq={eq})"
    assert n > 3000, "the sweep must actually be wide"


def test_a_non_finite_input_never_produces_an_order():
    rm = RiskManager(RiskConfig())
    for bad in ({"equity": NAN}, {"price": NAN}, {"stop_dist": NAN},
                {"equity": INF}, {"price": INF}, {"size_mult": NAN}):
        kw = {"equity": 10_000.0, "price": 100.0, "stop_dist": 1.0, "size_mult": 1.0}
        kw.update(bad)
        assert rm.size_entry(kw["equity"], kw["price"], kw["stop_dist"], LONG,
                             _spec(), size_mult=kw["size_mult"]) is None, bad


@pytest.mark.parametrize("style", ["trend", "scalp"])
def test_a_bracket_is_never_handed_out_unprotective(style):
    """S3 + S5. A non-finite stop is worse than no trade: the tick watcher tests
    `pos.stop_price > 0` and manage() tests `risk <= 0`, and BOTH are False for
    NaN — so the position would run with no working stop at all."""
    ex = AdaptiveExitManager(RiskConfig())
    rng = random.Random(3)
    for _ in range(1500):
        side = rng.choice((LONG, SHORT))
        d = 1 if side == LONG else -1
        entry = rng.choice([1e-4, 1.0, 100.0, 65_000.0, NAN, INF, 0.0, -1.0])
        atr = rng.choice([0.0, 1e-9, 0.05, 5.0, NAN, INF, -1.0])
        row = {"dc_hi": rng.choice([entry, NAN, INF, 0.0]),
               "dc_lo": rng.choice([entry, NAN, INF, 0.0])}
        regime = rng.choice(["TREND_UP", "RANGE", "VOLATILE", "NONSENSE"])
        br = ex.initial_bracket(entry, side, atr, row, regime, style)
        if br is None:
            continue
        assert math.isfinite(br.stop) and math.isfinite(br.init_risk), "S5"
        assert math.isfinite(br.take_profit), "S5 take_profit"
        assert br.init_risk > 0, "S3 1R must be a real distance"
        assert (br.stop - entry) * d < 0, \
            f"S3 stop {br.stop} on the wrong side of {entry} ({side})"


def test_a_trail_only_ever_tightens():
    """S4 — the invariant that decides whether a loss stays small."""
    ex = AdaptiveExitManager(RiskConfig())
    rng = random.Random(5)
    checked = 0
    for _ in range(400):
        side = rng.choice((LONG, SHORT))
        d = 1 if side == LONG else -1
        entry, atr = 100.0, 1.0
        row = {"dc_hi": 102.0, "dc_lo": 98.0, "atr": atr}
        br = ex.initial_bracket(entry, side, atr, row, "TREND_UP",
                                rng.choice(["trend", "scalp"]))
        assert br is not None
        pos = Position(symbol="X-USDT", side=side, qty=1.0, entry_price=entry,
                       opened_ts=0, stop_price=br.stop, take_profit=br.take_profit,
                       init_risk=br.init_risk, atr_ref=atr)
        prev = pos.stop_price
        for _ in range(15):
            px = entry * rng.uniform(0.93, 1.07)
            ex.manage(pos, px, px * 1.002, px * 0.998, atr, row,
                      rng.uniform(-1, 1), 0.3, "TREND_UP", rng.randint(1, 200))
            assert math.isfinite(pos.stop_price), "S5"
            assert (pos.stop_price - prev) * d >= -1e-9, \
                f"S4 stop LOOSENED {prev} -> {pos.stop_price} ({side})"
            prev = pos.stop_price
            checked += 1
    assert checked > 1000


def test_manage_ignores_a_position_with_non_finite_risk():
    ex = AdaptiveExitManager(RiskConfig())
    pos = Position(symbol="X-USDT", side=LONG, qty=1.0, entry_price=100.0,
                   opened_ts=0, stop_price=NAN, take_profit=0.0, init_risk=NAN)
    moved, reason = ex.manage(pos, 110.0, 110.0, 100.0, 1.0, {}, 0.4, 0.3, "TREND_UP", 5)
    assert (moved, reason) == (False, None), "abstain rather than act on garbage"


def test_a_healthy_bracket_is_unaffected():
    """The guards must not have broken normal operation."""
    ex = AdaptiveExitManager(RiskConfig())
    br = ex.initial_bracket(100.0, LONG, 1.0, {"dc_lo": 97.0, "dc_hi": 103.0},
                            "TREND_UP", "trend")
    assert br is not None and br.stop < 100.0 and br.init_risk > 0
    rm = RiskManager(RiskConfig())
    so = rm.size_entry(10_000.0, 100.0, 2.0, LONG, _spec())
    assert so is not None and so.qty > 0
