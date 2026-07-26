"""Strategy archetypes: a champion that is a KIND of desk, not just constants.

Every champion until now was the same strategy with different numbers. The desk
mix started uniform in every brain and was only ever learned online, so nothing
in a parameter set could say "I follow trends" or "I fade extremes" — and the
learning reset with every new brain, every fold and every restart.

`desk_tilt` is one integer the tuner owns, applied as a standing multiplier at
the same site as REGIME_DESK_MULT. Three layers compose there: what this
champion IS, what today's market rewards, and what is actually working.

The risk this carries is parity. The fused edge is computed twice — once in
Python, once in the compiled kernel — and the tuner ranks on one while the
account trades the other. A tilt implemented in one engine and not the other
would let the tuner validate a strategy that never trades. The default index is
uniform, so the standing parity suite would pass while proving nothing about
this feature; these tests exercise the tilts that are NOT the identity.
"""
import numpy as np
import pytest

from bingxbot.config import RiskConfig, StrategyConfig
from bingxbot.data.history import synthetic_candles
from bingxbot.engine.backtest import TUNABLES, _coerce, candles_to_arrays, run_backtest
from bingxbot.exchange.models import ContractSpec
from bingxbot.strategy.alphas import DESK_ORDER
from bingxbot.strategy.brain import TradingBrain
from bingxbot.strategy.features import FeatureFrame
from bingxbot.strategy.regime import DESK_TILT_NAMES, DESK_TILTS, desk_tilt_weights

numba = pytest.importorskip("numba")


@pytest.fixture(autouse=True)
def _no_meta(monkeypatch):
    import bingxbot.strategy.brain as brain_mod
    monkeypatch.setattr(brain_mod, "_get_meta", lambda *a, **k: None)


# ------------------------------------------------------------- the table

def test_index_zero_is_exactly_uniform():
    """Every champion promoted before this field existed carries no archetype,
    and must behave EXACTLY as it did — a silent re-weighting of the whole vault
    would be indistinguishable from the strategy quietly changing."""
    assert set(DESK_TILTS[0].values()) == {1.0}
    assert desk_tilt_weights(0) == DESK_TILTS[0]


def test_every_archetype_covers_every_desk():
    for i, t in enumerate(DESK_TILTS):
        assert set(t) == set(DESK_ORDER), f"archetype {i} misses a desk"


def test_the_names_and_the_table_stay_the_same_length():
    """The dashboard and the resume label an archetype by index."""
    assert len(DESK_TILT_NAMES) == len(DESK_TILTS)


def test_archetypes_only_tilt_desks_the_judge_can_see():
    """Historical klines carry no book, no tape and no funding, so the micro and
    carry desks are dormant in every backtest. An archetype leaning on them
    would be a knob out-of-sample validation cannot tell apart from any other —
    the tuner would pick it at random and the number would mean nothing."""
    for i, t in enumerate(DESK_TILTS):
        assert t["micro"] == 1.0, f"archetype {i} tilts a desk backtests cannot see"
        assert t["carry"] == 1.0, f"archetype {i} tilts a desk backtests cannot see"


def test_an_unknown_archetype_falls_back_to_balanced():
    """This integer arrives from a tuned parameter set and from champion JSON on
    disk. An index past the end must mean 'no opinion', never a crash inside the
    fusion loop."""
    for junk in (-1, 99, len(DESK_TILTS), None, "trend", float("nan"), float("inf")):
        assert desk_tilt_weights(junk) == DESK_TILTS[0], junk


def test_a_float_index_rounds_the_same_way_the_tuner_coerces_it():
    """_coerce rounds every int tunable, so a hand-edited 1.7 must not mean
    archetype 1 in the brain and archetype 2 to everything else."""
    assert desk_tilt_weights(1.0) == DESK_TILTS[1]
    assert desk_tilt_weights(1.7) == DESK_TILTS[_coerce("desk_tilt", 1.7)]
    assert desk_tilt_weights(1.2) == DESK_TILTS[_coerce("desk_tilt", 1.2)]


# ------------------------------------------------------------- the brain

def test_the_index_and_its_weights_can_never_disagree():
    """A hot swap that set the index and left stale weights behind would trade a
    different strategy than the one that was validated, invisibly."""
    b = TradingBrain(desk_tilt_idx=0)
    assert b.desk_tilt == DESK_TILTS[0]
    b.desk_tilt_idx = 2
    assert b.desk_tilt_idx == 2 and b.desk_tilt == DESK_TILTS[2]
    b.desk_tilt_idx = 999
    assert b.desk_tilt == DESK_TILTS[0], "an unknown index must fall back, not persist"


def test_a_junk_index_from_disk_does_not_raise():
    b = TradingBrain()
    b.desk_tilt_idx = "corrupted"
    assert b.desk_tilt_idx == 0 and b.desk_tilt == DESK_TILTS[0]


def test_the_archetype_actually_moves_the_fused_edge():
    """The whole point. If a trend-led and a meanrev-led brain produce identical
    edges on the same bar, the knob is decoration and the tuner is searching a
    dimension that does nothing."""
    candles = synthetic_candles("BTC-USDT", "5m", 900, seed=4)
    ff = FeatureFrame(candles_to_arrays(candles), interval="5m")
    from bingxbot.engine.backtest import NO_CTX, NO_MICRO
    seen = {}
    for idx in range(len(DESK_TILTS)):
        b = TradingBrain(desk_tilt_idx=idx)
        edges = [b.score(ff.row_cached(i), NO_MICRO, NO_CTX)["edge"]
                 for i in range(400, 900)]
        seen[idx] = edges
    for idx in range(1, len(DESK_TILTS)):
        assert seen[idx] != seen[0], f"archetype {idx} changes nothing"


def test_the_tilt_composes_with_the_regime_multiplier_rather_than_replacing_it():
    """Three factors, one product. A trend-led brain in a RANGE must still be
    damped by the regime — the archetype says what this champion is, not what
    today's market rewards."""
    from bingxbot.strategy.regime import (RANGE, REGIME_DESK_MULT, TREND_DOWN,
                                          TREND_UP)
    trend_led, meanrev_led = desk_tilt_weights(1), desk_tilt_weights(2)

    def mix(reg, tilt):
        return {d: REGIME_DESK_MULT[reg][d] * tilt[d] for d in DESK_ORDER}

    ranged = mix(RANGE, trend_led)
    assert ranged["trend"] < ranged["meanrev"], (
        "a trend-led champion must still be outvoted by mean-reversion in a "
        "range — the archetype is a preference, the regime is a fact")
    for up in (TREND_UP, TREND_DOWN):
        trending = mix(up, meanrev_led)
        assert trending["meanrev"] < trending["trend"], (
            "a mean-reversion champion must still not fade a decided trend — "
            "the exact failure REGIME_DESK_MULT was introduced to stop")


STRONG = 2.0    # a regime ratio this wide is a deliberate mute, not a nudge


def test_no_archetype_can_overturn_a_regime_that_has_decided():
    """Stated as the general rule rather than case by case, so a future
    archetype cannot quietly be given a louder voice than the market.

    The invariant is about ORDER, not magnitude: where a regime deliberately
    mutes one desk in favour of another, no archetype may invert that pair.
    Demanding the archetype be narrower than every regime would make the whole
    feature decoration — the regimes that express no strong view should leave
    the champion's own preference in charge, and they do."""
    from bingxbot.strategy.regime import REGIME_DESK_MULT, REGIMES
    checked = 0
    for reg in REGIMES:
        r = REGIME_DESK_MULT[reg]
        for a in DESK_ORDER:
            for b in DESK_ORDER:
                if r[a] < r[b] * STRONG:
                    continue          # the regime has no strong view on this pair
                for i, t in enumerate(DESK_TILTS):
                    checked += 1
                    assert r[a] * t[a] > r[b] * t[b], (
                        f"archetype {i} ({DESK_TILT_NAMES[i]}) inverts {reg}'s "
                        f"decision to favour {a} over {b}")
    assert checked, "the invariant matched no pair — the test proves nothing"


# ------------------------------------------------------------ the tunable

def test_the_archetype_is_a_tunable_integer_spanning_the_table():
    lo, hi, grp, kind = TUNABLES["desk_tilt"]
    assert kind == "int" and grp == "strategy"
    assert (lo, hi) == (0, len(DESK_TILTS) - 1), (
        "the search box must cover every archetype and nothing beyond it")


def test_every_box_value_coerces_to_a_real_archetype():
    lo, hi, _g, _k = TUNABLES["desk_tilt"]
    for v in np.linspace(lo, hi, 25):
        idx = _coerce("desk_tilt", float(v))
        assert desk_tilt_weights(idx) is DESK_TILTS[idx]


def test_the_config_default_is_the_uniform_archetype():
    assert StrategyConfig().desk_tilt == 0


# ------------------------------------------------------------- PARITY

@pytest.mark.parametrize("idx", list(range(1, len(DESK_TILTS))))
def test_the_kernel_reproduces_each_archetype_trade_for_trade(idx):
    """The standing parity suite runs at the default archetype, which is the
    identity — it would pass on a kernel that ignored this parameter entirely.
    Each NON-uniform archetype has to be checked on its own."""
    from bingxbot.engine.kernel import kernel_fitness
    strat, risk = StrategyConfig(desk_tilt=idx), RiskConfig()
    spec = ContractSpec("BTC-USDT")
    candles = synthetic_candles("BTC-USDT", "5m", 2500, seed=4)
    ff = FeatureFrame(candles_to_arrays(candles), interval="5m")
    py = run_backtest(candles, "BTC-USDT", "5m", strat, risk, spec, taker_fee=0.0005,
                      slippage_bps=1.0, collect_series=True, ff=ff)
    kr = kernel_fitness(ff, strat, risk, spec, 0.0005, 1.0, "5m")
    assert len(py["trades"]) == kr["stats"]["trades"], "trade count diverged"
    for a, ots, ets, pnl in zip(py["trades"], kr["trade_open_ts"],
                                kr["trade_ts"], kr["trade_pnl"]):
        assert a["entry_ts"] == ots and a["exit_ts"] == ets, "trade timing diverged"
        assert a["pnl"] == pytest.approx(pnl, abs=1e-9), "trade pnl diverged"
    assert py["stats"]["total_pnl"] == pytest.approx(kr["stats"]["total_pnl"], abs=1e-6)


def test_the_archetypes_are_distinguishable_to_the_kernel_too():
    """...and the kernel must not be quietly running the same strategy four
    times: if every archetype produced the same trades, parity above would hold
    while the tuner searched a dead dimension in the engine that ranks."""
    from bingxbot.engine.kernel import kernel_fitness
    spec = ContractSpec("BTC-USDT")
    candles = synthetic_candles("BTC-USDT", "5m", 2500, seed=4)
    ff = FeatureFrame(candles_to_arrays(candles), interval="5m")
    pnls = {i: kernel_fitness(ff, StrategyConfig(desk_tilt=i), RiskConfig(), spec,
                              0.0005, 1.0, "5m")["stats"]["total_pnl"]
            for i in range(len(DESK_TILTS))}
    assert len(set(pnls.values())) > 1, f"every archetype traded identically: {pnls}"


def test_the_kernel_survives_an_archetype_index_off_the_end():
    """pack_params copies the raw number into a nopython loop; an out-of-range
    index there is an out-of-bounds read, not an exception."""
    from bingxbot.engine.kernel import kernel_fitness
    spec = ContractSpec("BTC-USDT")
    candles = synthetic_candles("BTC-USDT", "5m", 800, seed=4)
    ff = FeatureFrame(candles_to_arrays(candles), interval="5m")
    base = kernel_fitness(ff, StrategyConfig(desk_tilt=0), RiskConfig(), spec,
                          0.0005, 1.0, "5m")["stats"]["total_pnl"]
    for junk in (-3, len(DESK_TILTS), 99):
        got = kernel_fitness(ff, StrategyConfig(desk_tilt=junk), RiskConfig(), spec,
                             0.0005, 1.0, "5m")["stats"]["total_pnl"]
        assert got == pytest.approx(base, abs=1e-9), (
            f"index {junk} must fall back to balanced, as the python side does")
