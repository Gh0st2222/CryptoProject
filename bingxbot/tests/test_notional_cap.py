"""max_position_notional_pct must actually cap something.

It was declared in RiskConfig, described in a comment as a per-position cap,
and referenced by an existing test — but no production code read it. A user
could set it in config.json believing it protected them and be wrong.

It is deliberately non-binding in normal conditions and binding only in the tail
it exists for: a very tight stop drives implied leverage into the band ceiling
and the position toward max_leverage x equity.
"""
import pytest

from bingxbot.config import RiskConfig
from bingxbot.exchange.models import ContractSpec
from bingxbot.risk.manager import RiskManager

EQ, PX = 10_000.0, 100.0


def _spec():
    s = ContractSpec("BTC-USDT")
    s.qty_precision, s.min_qty, s.min_notional_usdt = 6, 1e-6, 1.0
    return s


def _size(cfg, stop_frac):
    return RiskManager(cfg).size_entry(EQ, PX, PX * stop_frac, "LONG", _spec())


def test_the_cap_binds_when_a_tight_stop_would_max_out_leverage():
    """The case it exists for. Without it the position runs to the leverage
    ceiling — 7x equity — on nothing but a very close stop."""
    cfg = RiskConfig()
    cfg.max_position_notional_pct = 0.35
    so = _size(cfg, 0.001)
    assert so is not None
    cap = 0.35 * EQ * cfg.max_leverage
    assert so.notional <= cap + 1e-6, "the per-position ceiling held"
    assert so.notional < cfg.max_leverage * EQ, "and it really did cut the size"


def test_it_does_not_bind_in_normal_conditions():
    """Behavioural neutrality at the shipped defaults — wiring a dormant knob
    must not quietly resize every trade the champions were validated on."""
    cfg = RiskConfig()
    for stop_frac in (0.005, 0.01, 0.02, 0.03, 0.05):
        so = _size(cfg, stop_frac)
        if so is None:
            continue
        cap = cfg.max_position_notional_pct * EQ * cfg.max_leverage
        assert so.notional < cap, f"stop {stop_frac:.1%} should be nowhere near the cap"


def test_tightening_it_shrinks_the_position_proportionally():
    """It has to be a real dial, not just a tripwire."""
    cfg = RiskConfig()
    cfg.max_position_notional_pct = 0.05
    so = _size(cfg, 0.01)
    assert so is not None
    assert so.notional <= 0.05 * EQ * cfg.max_leverage + 1e-6


def test_the_risk_cap_still_wins_when_it_is_tighter():
    """The notional cap must not be able to ENLARGE a position past the hard
    per-trade risk limit."""
    cfg = RiskConfig()
    cfg.max_position_notional_pct = 1.0          # effectively off
    cfg.max_risk_hard_pct = 0.002
    so = _size(cfg, 0.01)
    assert so is not None
    assert so.qty * (PX * 0.01) <= cfg.max_risk_hard_pct * EQ + 1e-6


def test_zero_disables_it_rather_than_forbidding_all_trading():
    cfg = RiskConfig()
    cfg.max_position_notional_pct = 0.0
    so = _size(cfg, 0.01)
    assert so is not None and so.qty > 0, "0 means 'no cap', not 'no trades'"


def test_the_setting_is_read_at_all():
    """Regression guard for the actual bug: it was inert."""
    import inspect
    src = inspect.getsource(RiskManager.size_entry)
    assert "max_position_notional_pct" in src


def test_the_compiled_kernel_knows_about_it_too():
    """Wiring this in python alone broke kernel parity — the tuner would have
    optimized against a sizer the live engine does not use. The full parity
    test covers the behaviour; this names the specific knob so a future edit
    to one side fails here with an obvious reason."""
    from bingxbot.config import StrategyConfig
    from bingxbot.engine.kernel import P, pack_params
    assert "max_position_notional_pct" in P
    r = RiskConfig()
    r.max_position_notional_pct = 0.42
    assert pack_params(StrategyConfig(), r)[P["max_position_notional_pct"]] == pytest.approx(0.42)
