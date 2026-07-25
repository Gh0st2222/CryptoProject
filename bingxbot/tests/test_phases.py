"""Phase-shifted signal bars: decide every minute, never on partial data.

The property under test is the one the whole design rests on: a 15m window that
ends off the epoch grid is the SAME KIND OF OBJECT as one that ends on it, while
a still-forming bar is not. If that ever stops being true, phase entries become
the intra-bar scanner that already lost ten live trades, and these tests are the
tripwire.
"""
import json

import numpy as np
import pytest

from bingxbot.data.phases import minutes_needed, phase_arrays, phase_of


def _minutes(n: int, start_ms: int = 0, seed: int = 7):
    """Deterministic 1m bars on a clean minute grid."""
    rng = np.random.default_rng(seed)
    px = 100.0 + np.cumsum(rng.normal(0, 0.05, n))
    hi = px + np.abs(rng.normal(0, 0.04, n))
    lo = px - np.abs(rng.normal(0, 0.04, n))
    return {
        "ts": np.arange(n, dtype=np.int64) * 60_000 + start_ms,
        "open": px.copy(), "high": hi, "low": lo, "close": px.copy(),
        "volume": rng.uniform(1, 10, n),
    }


# ------------------------------------------------------- aggregation is honest

def test_phase_zero_reproduces_the_exchanges_own_bars():
    """Phase 0 must BE the epoch grid. If it drifted, switching phases on would
    silently move the bar the engine already trades."""
    m = _minutes(900)
    a = phase_arrays(m, factor=15, phase=0)
    assert a["ts"].size == 60
    assert np.all(np.asarray(a["ts"], dtype=np.int64) % (15 * 60_000) == 0)
    # first window = first 15 minutes, reduced the way a candle is reduced
    assert a["open"][0] == m["open"][0]
    assert a["close"][0] == m["close"][14]
    assert a["high"][0] == m["high"][:15].max()
    assert a["low"][0] == m["low"][:15].min()
    assert a["volume"][0] == pytest.approx(m["volume"][:15].sum())


def test_every_phase_opens_on_its_own_minute():
    m = _minutes(900)
    for p in range(15):
        ts = np.asarray(phase_arrays(m, 15, p)["ts"], dtype=np.int64)
        assert ts.size > 0
        assert np.all((ts // 60_000) % 15 == p)
        # plain all(), not np.all(): np.all() on a generator is always truthy
        assert all(phase_of(int(t), 15) == p for t in ts)


def test_an_incomplete_window_is_never_emitted():
    """The trailing in-progress window is exactly what we must not score."""
    m = _minutes(20)                       # one full window + 5 spare minutes
    a = phase_arrays(m, factor=15, phase=0)
    assert a["ts"].size == 1, "only the finished window"
    assert phase_arrays(_minutes(14), 15, 0)["ts"].size == 0


def test_a_gap_in_the_minute_feed_drops_the_window_instead_of_splicing_it():
    """A dropped minute must not be papered over — splicing across a gap invents
    a candle whose range spans a hole in time, and range is what sizes trades."""
    m = _minutes(900)
    keep = np.ones(900, dtype=bool)
    keep[20] = False                        # a minute inside the 2nd window
    gapped = {k: v[keep] for k, v in m.items()}
    a = phase_arrays(gapped, factor=15, phase=0)
    ts = np.asarray(a["ts"], dtype=np.int64)
    assert 15 * 60_000 not in set(ts.tolist()), "the window containing the gap is dropped"
    assert 0 in set(ts.tolist()) and 30 * 60_000 in set(ts.tolist()), "neighbours survive"


def test_max_windows_keeps_the_most_recent():
    m = _minutes(900)
    full = phase_arrays(m, 15, 0)
    cut = phase_arrays(m, 15, 0, max_windows=10)
    assert cut["ts"].size == 10
    assert np.array_equal(cut["ts"], full["ts"][-10:])


def test_minutes_needed_covers_the_last_phase():
    """Phase 14 starts 14 minutes late, so seeding must cover the offset."""
    need = minutes_needed(15, 40)
    m = _minutes(need)
    for p in range(15):
        assert phase_arrays(m, 15, p)["ts"].size >= 40, f"phase {p} short"


# ------------------------------- the property the whole design rests on

def _agg_positional(a, start, step=15, frac=1.0):
    """Aggregate by POSITION, so a partial (still-forming) bar can be built —
    something phase_arrays deliberately cannot express."""
    n = a["ts"].size
    take = max(1, int(round(step * frac)))
    out = {k: [] for k in a}
    for i in range(start, n - step + 1, step):
        g = slice(i, i + take)
        out["ts"].append(a["ts"][i])
        out["open"].append(a["open"][i])
        out["high"].append(a["high"][g].max())
        out["low"].append(a["low"][g].min())
        out["close"].append(a["close"][i + take - 1])
        out["volume"].append(a["volume"][g].sum())
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def _atr_pct(arrays):
    from bingxbot.strategy.features import FeatureFrame
    f = FeatureFrame(arrays, interval="15m")
    return float(np.nanmedian(np.asarray(f.f["atr_pct"], dtype=np.float64)[300:]))


@pytest.mark.parametrize("phase", [1, 7, 14])
def test_a_phase_window_does_not_distort_the_feature_that_sizes_the_trade(phase):
    """atr_pct on a phase window must match the epoch bar. It drives stop
    distance, and `qty = max_risk / stop_dist` — so a biased ATR mis-sizes every
    trade taken through this door. Tolerance is deliberately tight (2%): the
    measured spread across all 15 offsets was under 0.6%."""
    m = _minutes(30_000, seed=3)
    base = _atr_pct(phase_arrays(m, 15, 0))
    got = _atr_pct(phase_arrays(m, 15, phase))
    assert abs(got / base - 1) < 0.02, f"phase +{phase}m shifted atr_pct to {got:.6f} vs {base:.6f}"


def test_a_still_forming_bar_DOES_distort_it_which_is_why_we_do_not_score_one():
    """The counter-test, and the reason this module exists. If this ever stops
    failing the way it does, the partial-bar hazard has changed and the whole
    argument for phase windows needs re-deriving rather than assuming."""
    m = _minutes(30_000, seed=3)
    base = _atr_pct(phase_arrays(m, 15, 0))
    third = _atr_pct(_agg_positional(m, 0, frac=1 / 3))
    twothirds = _atr_pct(_agg_positional(m, 0, frac=2 / 3))
    assert third < base * 0.9, "a third-formed bar understates ATR badly"
    assert twothirds < base * 0.97, "even two-thirds formed still understates it"
    # and the direction is what mis-sizes: tighter stop => bigger position
    assert base / third > 1.1, "the sizing error is large enough to matter"


def test_the_mtf_veto_gate_sees_the_same_market_on_every_phase():
    """mtf_bias drives a HARD veto (never fight a decided 15m/1h trend). It is an
    epoch-anchored aggregation, so it is the feature most likely to shift when
    the base bar leaves the grid. Compared on its own [-1,1] scale — a ratio test
    would divide by a mean that sits at zero."""
    from bingxbot.config import StrategyConfig
    from bingxbot.strategy.features import FeatureFrame
    m = _minutes(30_000, seed=5)
    veto = StrategyConfig().mtf_veto

    def stat(phase):
        f = FeatureFrame(phase_arrays(m, 15, phase), interval="15m")
        b = np.asarray(f.f["mtf_bias"], dtype=np.float64)[300:]
        return float(np.nanstd(b)), float(np.mean(np.abs(b) >= veto))

    sd0, rate0 = stat(0)
    for phase in (1, 7, 14):
        sd, rate = stat(phase)
        assert abs(sd - sd0) < 0.05, f"phase +{phase}m changed mtf_bias spread"
        assert abs(rate - rate0) < 0.05, f"phase +{phase}m changed the veto rate"


# ------------------------------------------------------------------ the setting

def test_entry_phases_defaults_to_todays_behaviour():
    """Default 1 = decide only on the exchange's bar. Shipping this must not
    change how a single existing install trades until its owner opts in."""
    from bingxbot.config import StrategyConfig
    assert StrategyConfig().entry_phases == 1


def test_entry_phases_is_user_owned_and_survives_migration(tmp_path):
    from bingxbot.config import CONFIG_VERSION, load_config
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"version": CONFIG_VERSION - 1, "mode": "paper",
                             "strategy": {"entry_phases": 15}}))
    assert load_config(path=p).strategy.entry_phases == 15
