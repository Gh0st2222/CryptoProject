"""No alpha may hand the fusion a non-finite score.

An alpha's score goes straight into the weighted fuse, and `clamp` passes NaN
through, so ONE NaN alpha makes the whole edge NaN — and `abs(nan) >= threshold`
is False, so the bot silently stops trading.

Three alphas guarded their inputs only with comparisons (`if abs(flow) < 0.1`,
`if spread > 4.0`, `if tps <= 0`). Every comparison with NaN is False, so those
guards FAILED OPEN and fell through to the return. The other sixteen check
finiteness explicitly, which is why this was easy to miss.
"""
import math

import pytest

from bingxbot.strategy.alphas import ALPHAS

MICRO_KEYS = ("obi", "flow", "spread_bps", "cvd_slope", "ticks_per_s")
CTX_KEYS = ("funding_rate", "funding_z", "oi_change_pct", "tide_dir", "tide_er")
ROW_KEYS = ("ema_8", "ema_21", "ema_55", "atr", "atr_pct", "atr_pctile", "roc_3",
            "roc_12", "roc_accel", "macd_hist", "macd_rising", "adx", "rsi_14",
            "rsi_7", "rsi_3", "stoch_k", "stoch_d", "bb_pctb", "bb_width_pctile",
            "squeeze_on", "dc_hi", "dc_lo", "close", "open", "high", "low",
            "volume", "vwap", "vwap_dev", "vwap_z", "vol_z", "eff_ratio",
            "mtf_align", "mtf_bias", "linreg_slope", "ret_1", "ts")

NAN, INF = float("nan"), float("inf")


def _fill(keys, v):
    return {k: v for k in keys}


ROWS = {"zeros": _fill(ROW_KEYS, 0.0), "nan": _fill(ROW_KEYS, NAN),
        "inf": _fill(ROW_KEYS, INF), "neg_inf": _fill(ROW_KEYS, -INF), "empty": {}}
MICROS = {"empty": {}, "zeros": _fill(MICRO_KEYS, 0.0), "nan": _fill(MICRO_KEYS, NAN),
          "inf": _fill(MICRO_KEYS, INF)}
CTXS = {"empty": {}, "nan": _fill(CTX_KEYS, NAN)}


@pytest.mark.parametrize("alpha_name", sorted(ALPHAS))
def test_an_alpha_never_returns_a_non_finite_score(alpha_name):
    fn = ALPHAS[alpha_name]
    for rn, row in ROWS.items():
        for mn, micro in MICROS.items():
            for cn, ctx in CTXS.items():
                v = fn(row, micro, ctx)
                assert isinstance(v, (int, float)), f"{alpha_name} returned {v!r}"
                assert math.isfinite(v), \
                    f"{alpha_name} -> {v!r} on row={rn} micro={mn} ctx={cn}"
                assert -1.0 - 1e-9 <= v <= 1.0 + 1e-9, \
                    f"{alpha_name} -> {v} out of [-1,1] on row={rn} micro={mn} ctx={cn}"


def test_the_three_that_used_to_fail_open():
    """Named explicitly, because the failure was silent and specific."""
    nan_micro = _fill(MICRO_KEYS, NAN)
    for name in ("flow", "spread_pressure", "cvd_trend"):
        assert ALPHAS[name]({}, nan_micro, {}) == 0.0, f"{name} must abstain, not emit NaN"


def test_why_a_comparison_guard_is_not_enough():
    """The trap itself, so the reason is on the record."""
    nan = float("nan")
    assert (abs(nan) < 0.1) is False
    assert (nan > 4.0) is False
    assert (nan <= 0) is False
    assert math.isfinite(nan) is False, "only an explicit check catches it"


def test_a_healthy_row_still_produces_real_signal():
    """The guards must not have muted the desk."""
    row = {"ema_8": 101.0, "ema_21": 100.0, "ema_55": 99.0, "atr": 1.0,
           "atr_pct": 0.01, "roc_12": 0.02, "mtf_align": 0.6, "eff_ratio": 0.5,
           "close": 101.0, "dc_hi": 101.0, "dc_lo": 95.0, "vol_z": 1.0}
    micro = {"obi": 0.5, "flow": 0.5, "spread_bps": 1.0, "cvd_slope": 0.4,
             "ticks_per_s": 5.0}
    scores = {n: ALPHAS[n](row, micro, {}) for n in ALPHAS}
    assert all(math.isfinite(v) for v in scores.values())
    assert any(abs(v) > 0.1 for v in scores.values()), "something must still speak"
    for n in ("flow", "spread_pressure", "cvd_trend"):
        assert abs(scores[n]) > 0.0, f"{n} should still fire on good micro data"
