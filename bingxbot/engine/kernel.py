"""Compiled backtest kernel (numba) for the tuner's TRAINING folds.

The DE search's binding constraint is backtest throughput: every candidate
re-runs the Python bar loop. This kernel is a faithful nopython port of that
loop — brain fusion/learning, gates, adaptive exits, fills, risk and
accounting — operating on candidate-invariant arrays prepared once per fold
(features, alpha-score matrix, regimes).

Scope is deliberate and honest:

- **Training folds only** (DE member/trial ranking). The kernel runs WITHOUT
  the meta-labeling head — sklearn cannot execute inside nopython code — so
  out-of-sample validation and promotion stay on the full Python engine,
  meta included. The search proposes fast; the judge stays full-fidelity.
  (Until a meta model exists, the two brains are identical anyway.)
- **Parity-gated.** test_kernel asserts trade-for-trade agreement (count,
  timestamps, pnl) with the Python engine on synthetic data across seeds and
  parameter sets. If the kernel or numba is unavailable, score_fold falls
  back to the Python path transparently.

Everything here mirrors the Python semantics operation-for-operation —
iteration orders, clamps, roundings, tie-breaks — because "fast but slightly
different" is worse than slow.
"""
from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:  # noqa: BLE001 — kernel is an optimization, never a dependency
    HAVE_NUMBA = False

    def njit(*a, **k):  # type: ignore
        def deco(fn):
            return fn
        return deco if not (len(a) == 1 and callable(a[0])) else a[0]

from ..strategy.alphas import ALPHAS, ALPHA_META, DESK_ORDER
from ..strategy.regime import REGIME_DESK_MULT, REGIME_EXIT_MULT, detect_regime

ALPHA_NAMES = list(ALPHAS)
N_ALPHA = len(ALPHA_NAMES)
N_DESK = len(DESK_ORDER)
DESK_IDX = {d: i for i, d in enumerate(DESK_ORDER)}
ALPHA_DESK = np.array([DESK_IDX[ALPHA_META[nm][0]] for nm in ALPHA_NAMES], dtype=np.int64)
REGIMES = ("TREND_UP", "TREND_DOWN", "RANGE", "VOLATILE")
REG_IDX = {r: i for i, r in enumerate(REGIMES)}
REG_DESK_MULT = np.array([[REGIME_DESK_MULT[r].get(d, 1.0) for d in DESK_ORDER]
                          for r in REGIMES], dtype=np.float64)
REG_EXIT = np.array([[REGIME_EXIT_MULT[r]["sl"], REGIME_EXIT_MULT[r]["tp"],
                      REGIME_EXIT_MULT[r]["trail"]] for r in REGIMES], dtype=np.float64)

# exits.py structural constants (must track that module)
MTF_EXIT_GUARD = 0.35
EDGE_EXIT_OVERRIDE = 0.55
EDGE_FLIP_BARS = 2
# backtest.py constants (mirrored — test_kernel asserts they stay equal)
ASSUMED_SPREAD_BPS = 1.0
FUNDING_MS = 8 * 3600 * 1000
ASSUMED_FUNDING_8H = 0.0001
EV_MARGIN = 0.02
FILL_THROUGH_BPS = 1.0
STOP_SLIP_MULT = 2.0
# risk manager constants
MAINT_MARGIN_RATE = 0.005
LIQ_STOP_HEADROOM = 0.8

# feature-column layout for the kernel's 2D feature matrix
FEAT_COLS = ("ts", "open", "high", "low", "close", "atr", "atr_pct", "atr_pctile",
             "eff_ratio", "mtf_align", "mtf_bias", "bb_pctb", "dc_hi", "dc_lo")
F_TS, F_O, F_H, F_L, F_C, F_ATR, F_ATRP, F_ATRPC, F_ER, F_ALIGN, F_BIAS, F_PCTB, F_DCHI, F_DCLO = range(14)

# parameter-vector layout (strategy + risk scalars the loop reads)
PARAM_NAMES = (
    "hedge_eta", "weight_floor", "horizon_bars", "base_threshold", "threshold_adapt",
    "target_trades_per_hour", "bars_per_hour", "cost_multiple", "min_p_win",
    "kelly_fraction", "use_kelly", "min_efficiency", "mtf_veto", "trade_range",
    "range_band_edge", "trade_volatile", "discipline", "trend_align_gate",
    "is_maker", "maker_offset_bps", "maker_wait_bars", "entry_pullback_atr",
    "risk_per_trade", "min_leverage", "max_leverage", "max_risk_hard_pct",
    "sl_atr_min", "sl_atr_max", "tp_atr_cap", "trail_atr_min", "trail_atr_max",
    "trail_tighten", "be_rr", "be_offset_atr", "giveback_rr", "giveback_frac",
    "hold_edge_frac", "expected_rr", "time_stop_bars", "scalp_tp_atr",
    "scalp_sl_atr", "scalp_time_stop", "scalp_expected_rr", "maker_adverse_bps",
    "max_open_positions", "max_daily_loss_pct", "max_consecutive_losses",
    "cooldown_minutes", "max_spread_bps", "scaleout_rr", "scaleout_frac",
    "trail_scale_trend", "trail_scale_chop",
)
P = {n: i for i, n in enumerate(PARAM_NAMES)}


def pack_params(strat, risk) -> np.ndarray:
    from dataclasses import asdict
    v = np.zeros(len(PARAM_NAMES), dtype=np.float64)
    src = {**asdict(strat), **asdict(risk)}
    for n, i in P.items():
        if n == "is_maker":
            v[i] = 1.0 if strat.entry_mode == "maker" else 0.0
        elif n == "bars_per_hour":
            v[i] = 0.0  # filled by caller (needs the interval)
        else:
            val = src[n]
            v[i] = (1.0 if val else 0.0) if isinstance(val, bool) else float(val)
    return v


def prep_fold(ff) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Candidate-invariant kernel inputs, built once per FeatureFrame and cached
    on it: the feature matrix, the (n x N_ALPHA) alpha-score matrix, and the
    per-bar regime codes (computed by the REAL detect_regime, so regime
    semantics can never fork between engines)."""
    if ff._kmats is not None:
        return ff._kmats
    from .backtest import _alpha_cache
    alpha_rows = _alpha_cache(ff)         # list[dict] — also used by the Python path
    n = ff.n
    feats = np.empty((n, len(FEAT_COLS)), dtype=np.float64)
    for j, k in enumerate(FEAT_COLS):
        feats[:, j] = np.asarray(ff.f[k], dtype=np.float64)
    amat = np.empty((n, N_ALPHA), dtype=np.float64)
    for i in range(n):
        r = alpha_rows[i]
        for a, nm in enumerate(ALPHA_NAMES):
            amat[i, a] = r[nm]
    regs = np.empty(n, dtype=np.int64)
    for i in range(n):
        regs[i] = REG_IDX[detect_regime(ff.row_cached(i))[0]]
    ff._kmats = (feats, amat, regs)
    return ff._kmats


@njit(cache=True)
def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


@njit(cache=True)
def _round_step(value, precision):
    if precision <= 0:
        return math.floor(value)
    factor = 10.0 ** precision
    return math.floor(value * factor + 1e-9) / factor


@njit(cache=True)
def _sigmoid(z):
    if z >= 0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


@njit(cache=True)
def run_kernel(feats, amat, regs, p, warmup, taker, maker_fee, slip_bps,
               qty_prec, min_qty, min_notional, starting_balance):  # noqa: C901
    """The whole event loop, nopython. Returns (n_trades, trade_ts, trade_pnl,
    final_equity, max_dd, gross_win, gross_loss, funding_paid)."""
    n = feats.shape[0]
    horizon = max(1, int(p[P_HOR]))
    eta = p[P_ETA]
    floor_w = p[P_FLOOR]
    slip = slip_bps / 10_000.0
    maker_adv = p[P_MADV] / 10_000.0
    maker_off = p[P_MOFF] / 10_000.0
    is_maker = p[P_ISMAKER] > 0.5
    fees_rt = taker + (maker_fee if is_maker else taker)

    # ---- brain state
    alpha_w = np.empty(N_ALPHA)
    desk_count = np.zeros(N_DESK)
    for a in range(N_ALPHA):
        desk_count[ALPHA_DESK[a]] += 1.0
    for a in range(N_ALPHA):
        alpha_w[a] = 1.0 / desk_count[ALPHA_DESK[a]]
    log_w = np.zeros(N_DESK)
    ew_pay = np.zeros(N_DESK)
    ew_win = np.zeros(N_DESK)
    ew_var = np.ones(N_DESK)
    d_graded = np.zeros(N_DESK)
    d_disabled = np.zeros(N_DESK)
    alloc = np.full(N_DESK, 1.0 / N_DESK)
    cal_w = np.array([1.2, 0.0, 0.15, 0.0])
    cal_b = 0.0
    cal_n = 0
    beta = 1.0
    threshold = p[P_BTHR]
    sh_buf = np.zeros(720)
    sh_n = 0
    sh_head = 0
    # pending-grade ring (one entry per bar): bar index, edge, desk sigs
    pg_cap = horizon + 2
    pg_idx = np.full(pg_cap, -1, dtype=np.int64)
    pg_gi = np.zeros(pg_cap, dtype=np.int64)   # GLOBAL bar index for amat lookups
    pg_edge = np.zeros(pg_cap)
    pg_close = np.zeros(pg_cap)
    pg_atr = np.zeros(pg_cap)
    pg_atrpc = np.zeros(pg_cap)
    pg_reg = np.zeros(pg_cap, dtype=np.int64)
    pg_dsig = np.zeros((pg_cap, N_DESK))
    pg_head = 0
    pg_n = 0
    brain_idx = 0

    # ---- risk state
    day_key = np.int64(-1)
    day_start_eq = 0.0
    day_realized = 0.0
    consec_losses = 0
    cooldown_until = 0.0
    killed = False
    r_hist = np.zeros(30)
    rh_n = 0
    rh_head = 0
    health_scalar = 1.0
    health_peak = 0.0
    health_dd = 0.0

    # ---- portfolio
    cash = starting_balance
    funding_paid = 0.0
    peak_eq = 0.0
    max_dd = 0.0
    max_trades = 2 * n + 8    # scale-outs can produce two records per position
    tr_ts = np.zeros(max_trades, dtype=np.int64)
    tr_open = np.zeros(max_trades, dtype=np.int64)
    tr_pnl = np.zeros(max_trades)
    tr_qty = np.zeros(max_trades)
    tr_dist = np.zeros(max_trades)
    n_tr = 0
    pos_dist_dbg = 0.0
    dbg_edge = np.zeros(n)
    dbg_pwin = np.zeros(n)
    dbg_thr = np.zeros(n)
    pos_open_ts = np.int64(0)
    gross_w = 0.0
    gross_l = 0.0

    # ---- position state (single symbol)
    has_pos = False
    pos_long = True
    pos_qty = 0.0
    pos_entry = 0.0
    pos_entry_fee = 0.0
    pos_stop = 0.0
    pos_tp = 0.0
    pos_be_moved = False
    pos_peak = 0.0
    pos_trough = 0.0
    pos_atr_ref = 0.0
    pos_init_risk = 0.0
    pos_scaled = False
    pos_flip_bars = 0
    pos_scalp = False
    planned_risk = 0.0
    bars_held = 0

    # ---- pending entry
    pend = False
    pend_long = True
    pend_taker = False
    pend_limit = 0.0
    pend_atr = 0.0
    pend_reg = 0
    pend_scalp = False
    pend_mult = 1.0
    pend_expires = 0

    last_eq = starting_balance

    for i in range(warmup, n):
        ts = np.int64(feats[i, F_TS])
        o = feats[i, F_O]; hi = feats[i, F_H]; lo = feats[i, F_L]; c = feats[i, F_C]
        atr = feats[i, F_ATR]; atr_pct = feats[i, F_ATRP]; atr_pctile = feats[i, F_ATRPC]
        reg = regs[i]
        clock = ts / 1000.0

        # (the risk day rolls lazily at each accounting site, matching Python's
        # _roll_day-at-call-time semantics — never at bar top)

        # ---------- 1) resolve pending entry
        if pend and not has_pos:
            fill = False
            fpx = 0.0
            fmaker = False
            if pend_taker:
                fill = True; fpx = o; fmaker = False
            else:
                thru = FILL_THROUGH_BPS / 10_000.0
                hitp = (lo <= pend_limit * (1.0 - thru)) if pend_long else (hi >= pend_limit * (1.0 + thru))
                if hitp:
                    fill = True; fpx = pend_limit; fmaker = True
                elif i >= pend_expires:
                    pend = False
            if fill:
                pend = False
                d = 1.0 if pend_long else -1.0
                eff = fpx * (1.0 + d * maker_adv) if fmaker else fpx * (1.0 + d * slip)
                # bracket at fill bar (dc), decision atr, decision regime
                if eff > 0 and pend_atr > 0:
                    if pend_scalp:
                        dist = p[P_SSL] * pend_atr
                        stop = eff - d * dist
                        tp = eff + d * p[P_STP] * pend_atr
                    else:
                        exsl = REG_EXIT[pend_reg, 0]
                        blo = p[P_SLMIN] * exsl * pend_atr
                        bhi = p[P_SLMAX] * exsl * pend_atr
                        if pend_long:
                            swing = feats[i, F_DCLO]
                            struct = eff - swing
                        else:
                            swing = feats[i, F_DCHI]
                            struct = swing - eff
                        dist = _clamp(struct, blo, bhi)
                        stop = eff - d * dist
                        tp = eff + d * p[P_TPCAP] * pend_atr if p[P_TPCAP] > 0 else 0.0
                    # sizing
                    eqs = cash
                    if eqs > 0 and dist > 0:
                        eff_mult = _clamp(pend_mult, 0.1, 2.0)
                        risk_amount = eqs * p[P_RPT] * eff_mult
                        implied = (risk_amount / dist) * eff / eqs
                        stop_frac = dist / eff
                        liq_cap = 1.0 / (stop_frac / LIQ_STOP_HEADROOM + MAINT_MARGIN_RATE)
                        lev = min(min(implied, p[P_LEVMAX]), liq_cap)
                        if lev > 0:
                            qty = lev * eqs / eff
                            max_risk = eqs * p[P_HARD]
                            if qty * dist > max_risk:
                                qty = max_risk / dist
                            qty = _round_step(qty, qty_prec)
                            if qty >= min_qty and qty * eff >= min_notional:
                                fee = qty * eff * (maker_fee if fmaker else taker)
                                cash -= fee
                                has_pos = True
                                pos_long = pend_long
                                pos_qty = qty
                                pos_entry = eff
                                pos_entry_fee = fee
                                pos_stop = stop
                                pos_tp = tp
                                pos_be_moved = False
                                pos_peak = eff
                                pos_trough = eff
                                pos_atr_ref = pend_atr
                                pos_init_risk = dist
                                pos_scaled = False
                                pos_flip_bars = 0
                                pos_scalp = pend_scalp
                                pos_open_ts = ts
                                planned_risk = qty * dist
                                pos_dist_dbg = dist
                                bars_held = 0

        # ---------- funding at 8h boundaries
        if has_pos and i > 0:
            if (ts // FUNDING_MS) != (np.int64(feats[i - 1, F_TS]) // FUNDING_MS):
                amt = pos_qty * c * ASSUMED_FUNDING_8H
                cash -= amt
                funding_paid += amt

        # ---------- 2) intrabar stop / tp
        if has_pos:
            d = 1.0 if pos_long else -1.0
            if pos_stop > 0 and ((lo <= pos_stop) if pos_long else (hi >= pos_stop)):
                gap = (o < pos_stop) if pos_long else (o > pos_stop)
                px = o if gap else pos_stop
                s2 = slip * STOP_SLIP_MULT   # stops fire into momentum
                px = px * (1.0 - s2) if pos_long else px * (1.0 + s2)
                fee = pos_qty * px * taker
                gross = (px - pos_entry) * pos_qty * d
                cash += gross - fee
                net = gross - (pos_entry_fee + fee)
                pnl = np.round(net * 1e8) / 1e8
                tr_ts[n_tr] = ts; tr_open[n_tr] = pos_open_ts; tr_pnl[n_tr] = pnl; tr_qty[n_tr] = pos_qty; tr_dist[n_tr] = pos_dist_dbg; n_tr += 1
                if pnl > 0:
                    gross_w += pnl
                else:
                    gross_l += -pnl
                rmul = np.round((net / planned_risk) * 1e3) / 1e3 if planned_risk > 0 else 0.0
                # risk.on_trade_closed
                eq_after = cash
                day2 = ts // 86_400_000
                if day2 != day_key:
                    day_key = day2; day_start_eq = eq_after; day_realized = 0.0
                    consec_losses = 0; cooldown_until = 0.0; killed = False
                r_hist[rh_head] = _clamp(rmul, -3.0, 5.0); rh_head = (rh_head + 1) % 30
                if rh_n < 30:
                    rh_n += 1
                if eq_after > health_peak:
                    health_peak = eq_after
                health_dd = (health_peak - eq_after) / health_peak if health_peak > 0 else 0.0
                if rh_n < 8:
                    hbase = 1.0
                else:
                    s = 0.0
                    for q in range(rh_n):
                        s += r_hist[(rh_head - rh_n + q) % 30]
                    hbase = _clamp(1.0 + (s / rh_n) * 0.9, 0.4, 1.3)
                health_scalar = _clamp(hbase * _clamp(1.0 - health_dd * 3.0, 0.3, 1.0), 0.3, 1.3)
                day_realized += pnl
                consec_losses = 0 if pnl > 0 else consec_losses + 1
                if consec_losses >= int(p[P_MAXLOSS]):
                    cooldown_until = clock + p[P_COOL] * 60.0
                    consec_losses = 0
                if day_start_eq > 0 and (-day_realized / day_start_eq) >= p[P_DAYLOSS]:
                    killed = True
                has_pos = False
                bars_held = 0
            elif pos_tp > 0 and ((hi >= pos_tp) if pos_long else (lo <= pos_tp)):
                px = pos_tp
                fee = pos_qty * px * maker_fee
                gross = (px - pos_entry) * pos_qty * d
                cash += gross - fee
                net = gross - (pos_entry_fee + fee)
                pnl = np.round(net * 1e8) / 1e8
                tr_ts[n_tr] = ts; tr_open[n_tr] = pos_open_ts; tr_pnl[n_tr] = pnl; tr_qty[n_tr] = pos_qty; tr_dist[n_tr] = pos_dist_dbg; n_tr += 1
                if pnl > 0:
                    gross_w += pnl
                else:
                    gross_l += -pnl
                rmul = np.round((net / planned_risk) * 1e3) / 1e3 if planned_risk > 0 else 0.0
                eq_after = cash
                day2 = ts // 86_400_000
                if day2 != day_key:
                    day_key = day2; day_start_eq = eq_after; day_realized = 0.0
                    consec_losses = 0; cooldown_until = 0.0; killed = False
                r_hist[rh_head] = _clamp(rmul, -3.0, 5.0); rh_head = (rh_head + 1) % 30
                if rh_n < 30:
                    rh_n += 1
                if eq_after > health_peak:
                    health_peak = eq_after
                health_dd = (health_peak - eq_after) / health_peak if health_peak > 0 else 0.0
                if rh_n < 8:
                    hbase = 1.0
                else:
                    s = 0.0
                    for q in range(rh_n):
                        s += r_hist[(rh_head - rh_n + q) % 30]
                    hbase = _clamp(1.0 + (s / rh_n) * 0.9, 0.4, 1.3)
                health_scalar = _clamp(hbase * _clamp(1.0 - health_dd * 3.0, 0.3, 1.0), 0.3, 1.3)
                day_realized += pnl
                consec_losses = 0 if pnl > 0 else consec_losses + 1
                if consec_losses >= int(p[P_MAXLOSS]):
                    cooldown_until = clock + p[P_COOL] * 60.0
                    consec_losses = 0
                if day_start_eq > 0 and (-day_realized / day_start_eq) >= p[P_DAYLOSS]:
                    killed = True
                has_pos = False
                bars_held = 0

        # ---------- 3) brain: fuse + learn (score+observe)
        # desk fusion from the alpha matrix
        d_num = np.zeros(N_DESK)
        d_den = np.zeros(N_DESK)
        d_active = np.zeros(N_DESK, dtype=np.int64)
        for a in range(N_ALPHA):
            s = amat[i, a]
            dk = ALPHA_DESK[a]
            w = alpha_w[a]
            d_num[dk] += w * s
            d_den[dk] += w
            if abs(s) > 0.05:
                d_active[dk] += 1
        desk_sig = np.zeros(N_DESK)
        for dk in range(N_DESK):
            desk_sig[dk] = _clamp(d_num[dk] / d_den[dk] if d_den[dk] > 0 else 0.0, -1.0, 1.0)
        fnum = 0.0
        fden = 0.0
        for dk in range(N_DESK):
            if d_active[dk] == 0:
                continue
            w = alloc[dk] * REG_DESK_MULT[reg, dk]
            fnum += w * desk_sig[dk]
            fden += w
        edge = _clamp(fnum / fden if fden > 0 else 0.0, -1.0, 1.0)
        # calibrator predict
        a_e = abs(_clamp(edge, -1.0, 1.0))
        is_trend = 1.0 if reg <= 1 else 0.0
        z = cal_b + cal_w[0] * a_e + cal_w[1] * a_e * a_e + cal_w[2] * is_trend + cal_w[3] * _clamp(atr_pctile, 0.0, 1.0)
        p_model = _sigmoid(z)
        prior = 0.5 + 0.22 * a_e
        blend = _clamp(cal_n / 200.0, 0.0, 1.0)
        p_win = _clamp(blend * p_model + (1.0 - blend) * prior, 0.05, 0.95)
        dbg_edge[i] = edge
        dbg_pwin[i] = p_win
        dbg_thr[i] = threshold

        # observe: grade matured pendings
        while pg_n > 0:
            tail = (pg_head - pg_n) % pg_cap
            if brain_idx - pg_idx[tail] < horizon:
                break
            pcl = pg_close[tail]
            if pcl > 0 and c > 0:
                ret = (c - pcl) / pcl
                norm = _clamp(ret / max(pg_atr[tail] / pcl, 1e-9), -2.5, 2.5)
                gi = pg_gi[tail]
                for a in range(N_ALPHA):
                    s = amat[gi, a]
                    if abs(s) < 0.10:
                        continue
                    payoff = (min(abs(s), 1.0) * (1.0 if s > 0 else -1.0)) * norm
                    alpha_w[a] *= math.exp(eta * _clamp(payoff, -2.0, 2.0) * 0.25 / horizon)
                # renormalize desks
                for dk in range(N_DESK):
                    k = desk_count[dk]
                    uni = 1.0 / k
                    tot = 0.0
                    for a in range(N_ALPHA):
                        if ALPHA_DESK[a] == dk:
                            tot += alpha_w[a]
                    if tot == 0.0:
                        tot = 1.0
                    lam = 0.004
                    for a in range(N_ALPHA):
                        if ALPHA_DESK[a] == dk:
                            alpha_w[a] = (1 - lam) * (alpha_w[a] / tot) + lam * uni
                    exc_tot = 0.0
                    for a in range(N_ALPHA):
                        if ALPHA_DESK[a] == dk:
                            e = alpha_w[a] - floor_w
                            if e > 0:
                                exc_tot += e
                    free = 1.0 - k * floor_w
                    if exc_tot <= 1e-12 or free <= 0:
                        for a in range(N_ALPHA):
                            if ALPHA_DESK[a] == dk:
                                alpha_w[a] = uni
                    else:
                        for a in range(N_ALPHA):
                            if ALPHA_DESK[a] == dk:
                                e = alpha_w[a] - floor_w
                                if e < 0:
                                    e = 0.0
                                alpha_w[a] = floor_w + e * (free / exc_tot)
                # allocator update per desk
                for dk in range(N_DESK):
                    sig = pg_dsig[tail, dk]
                    if abs(sig) < 0.10:
                        continue
                    payoff = (min(abs(sig), 1.0) * (1.0 if sig > 0 else -1.0)) * norm
                    aa = 0.05
                    ew_pay[dk] = (1 - aa) * ew_pay[dk] + aa * payoff
                    ew_win[dk] = (1 - aa) * ew_win[dk] + aa * (1.0 if sig * ret > 0 else 0.0)
                    dv = payoff - ew_pay[dk]
                    ew_var[dk] = (1 - aa) * ew_var[dk] + aa * dv * dv
                    d_graded[dk] += 1
                    risk_adj = ew_pay[dk] / math.sqrt(max(ew_var[dk], 1e-6))
                    log_w[dk] += 0.6 * _clamp(risk_adj, -3.0, 3.0) * 0.02
                    d_disabled[dk] = 1.0 if (d_graded[dk] > 120 and ew_pay[dk] < -0.05 and ew_win[dk] < 0.42) else 0.0
                    # recompute alloc (softmax + floor + ceiling passes + decay)
                    m = log_w[0]
                    for q in range(1, N_DESK):
                        if log_w[q] > m:
                            m = log_w[q]
                    raw = np.empty(N_DESK)
                    tot2 = 0.0
                    for q in range(N_DESK):
                        raw[q] = math.exp(log_w[q] - m)
                        if d_disabled[q] > 0.5:
                            raw[q] *= 0.15
                        tot2 += raw[q]
                    if tot2 == 0.0:
                        tot2 = 1.0
                    free2 = 1.0 - N_DESK * 0.05
                    for q in range(N_DESK):
                        alloc[q] = 0.05 + free2 * raw[q] / tot2
                    ceil = max(0.40, 1.0 / N_DESK)
                    for _pass in range(3):
                        spill = 0.0
                        n_over = 0
                        for q in range(N_DESK):
                            if alloc[q] > ceil + 1e-9:
                                spill += alloc[q] - ceil
                                n_over += 1
                        if n_over == 0:
                            break
                        base2 = 0.0
                        for q in range(N_DESK):
                            if alloc[q] < ceil - 1e-9:
                                base2 += alloc[q]
                        if base2 == 0.0:
                            base2 = 1.0
                        for q in range(N_DESK):
                            if alloc[q] > ceil + 1e-9:
                                alloc[q] = ceil
                        for q in range(N_DESK):
                            if alloc[q] < ceil - 1e-9:
                                alloc[q] += spill * alloc[q] / base2
                    for q in range(N_DESK):
                        log_w[q] *= 0.995
                # calibrator update
                pe = pg_edge[tail]
                if abs(pe) > 0.10:
                    a2 = abs(_clamp(pe, -1.0, 1.0))
                    it2 = 1.0 if pg_reg[tail] <= 1 else 0.0
                    x3 = _clamp(pg_atrpc[tail], 0.0, 1.0)
                    z2 = cal_b + cal_w[0] * a2 + cal_w[1] * a2 * a2 + cal_w[2] * it2 + cal_w[3] * x3
                    pm = _sigmoid(z2)
                    y = 1.0 if pe * ret > 0 else 0.0
                    g = pm - y
                    lr = 0.03
                    l2 = 1e-4
                    cal_b -= lr * g
                    cal_w[0] -= lr * (g * a2 + l2 * cal_w[0])
                    cal_w[1] -= lr * (g * a2 * a2 + l2 * cal_w[1])
                    cal_w[2] -= lr * (g * it2 + l2 * cal_w[2])
                    cal_w[3] -= lr * (g * x3 + l2 * cal_w[3])
                    cal_n += 1
                # beta
                pred = abs(pe) * (pg_atr[tail] / pcl) * math.sqrt(horizon)
                if pred > 1e-9 and abs(pe) > 0.15:
                    ratio = _clamp(abs(ret) / pred, 0.1, 4.0)
                    beta = _clamp(beta + 0.03 * (ratio - beta), 0.3, 3.0)
            pg_n -= 1
        # record this bar for future grading
        pg_idx[pg_head] = brain_idx
        pg_gi[pg_head] = i
        pg_edge[pg_head] = edge
        pg_close[pg_head] = c
        pg_atr[pg_head] = max(atr, 1e-12)
        pg_atrpc[pg_head] = atr_pctile
        pg_reg[pg_head] = reg
        for dk in range(N_DESK):
            pg_dsig[pg_head, dk] = desk_sig[dk]
        pg_head = (pg_head + 1) % pg_cap
        if pg_n < pg_cap:
            pg_n += 1
        sh_buf[sh_head] = abs(edge)
        sh_head = (sh_head + 1) % 720
        if sh_n < 720:
            sh_n += 1
        brain_idx += 1
        # threshold adapt — python's ev["threshold"] snapshot (used by exit
        # management) is the PRE-adapt value from score(); the entry gate reads
        # brain.threshold AFTER observe() adapted it. Preserve both.
        thr_prev = threshold
        if p[P_TADAPT] < 0.5 or sh_n < 120:
            threshold = p[P_BTHR]
        else:
            rate = min(p[P_TRATE], 0.5 * p[P_BPH])
            pq = _clamp(1.0 - rate / p[P_BPH], 0.5, 0.995)
            tmp = np.sort(sh_buf[:sh_n].copy())
            qv = tmp[min(int(pq * sh_n), sh_n - 1)]
            threshold = _clamp(0.5 * qv + 0.5 * p[P_BTHR], p[P_BTHR], 0.92)

        # ---------- 4) manage / 5) entry
        if has_pos:
            bars_held += 1
            d = 1.0 if pos_long else -1.0
            matr = atr if atr > 0 else pos_atr_ref
            risk_d = pos_init_risk if pos_init_risk > 0 else abs(pos_entry - pos_stop)
            exit_code = 0  # 0 none, 1 full close, 2 scale-out
            moved = False
            if risk_d > 0:
                fav = hi if pos_long else lo
                adv = lo if pos_long else hi
                if pos_peak == 0.0:
                    pos_peak = pos_entry
                if (fav - pos_peak) * d > 0:
                    pos_peak = fav
                if pos_trough == 0.0:
                    pos_trough = pos_entry
                if (adv - pos_trough) * d < 0:
                    pos_trough = adv
                gain = (c - pos_entry) * d
                rr = gain / risk_d
                flip = (edge * d <= -p[P_HEDGEF] * thr_prev) and abs(edge) > 0.15
                pos_flip_bars = pos_flip_bars + 1 if flip else 0
                if flip and abs(edge) >= EDGE_EXIT_OVERRIDE:
                    exit_code = 1
                elif flip:
                    bias = feats[i, F_BIAS]
                    supported = bias * d >= MTF_EXIT_GUARD
                    if (not supported) and pos_flip_bars >= EDGE_FLIP_BARS:
                        exit_code = 1
                if exit_code == 0 and pos_scalp:
                    if not pos_be_moved and rr >= 0.6:
                        be = pos_entry + d * p[P_BEOFF] * matr
                        if (be - pos_stop) * d > 0:
                            pos_stop = be
                            pos_be_moved = True
                    if bars_held >= int(p[P_STIME]):
                        exit_code = 1
                elif exit_code == 0:
                    if not pos_be_moved and rr >= p[P_BERR]:
                        be = pos_entry + d * p[P_BEOFF] * matr
                        if (be - pos_stop) * d > 0:
                            pos_stop = be
                            pos_be_moved = True
                            moved = True
                    er = feats[i, F_ER]
                    rscale = p[P_TRSTREND] if reg <= 1 else p[P_TRSCHOP]
                    k_base = p[P_TRMIN] + (p[P_TRMAX] - p[P_TRMIN]) * _clamp(er / 0.5, 0.0, 1.0)
                    tighten = 1.0 - p[P_TTIGHT] * _clamp((rr - p[P_BERR]) / 3.0, 0.0, 1.0)
                    kk = k_base * REG_EXIT[reg, 2] * tighten * rscale
                    chand = pos_peak - d * kk * matr
                    if pos_be_moved and (chand - pos_stop) * d > 0:
                        pos_stop = chand
                        moved = True
                    so = p[P_SORR]
                    if so > 0 and not pos_scaled and rr >= so:
                        exit_code = 2
                    if exit_code == 0 and rr >= p[P_GBRR]:
                        peak_gain = (pos_peak - pos_entry) * d
                        retr = (pos_peak - c) * d
                        if peak_gain > 0 and retr / peak_gain >= p[P_GBFRAC]:
                            exit_code = 1
                    if exit_code == 0 and bars_held >= int(p[P_TSTOP]):
                        exit_code = 1
            if exit_code == 2:
                frac = p[P_SOFRAC]
                qty_out = pos_qty * frac
                if qty_out < min_qty or (pos_qty - qty_out) < min_qty:
                    exit_code = 1     # dust -> full close ("scale out (full)")
                else:
                    eff = c * (1.0 - slip) if pos_long else c * (1.0 + slip)
                    fee = qty_out * eff * taker
                    gross = (eff - pos_entry) * qty_out * d
                    fee_part = pos_entry_fee * frac
                    cash += gross - fee
                    net = gross - fee - fee_part
                    pnl = np.round(net * 1e8) / 1e8
                    tr_ts[n_tr] = ts; tr_open[n_tr] = pos_open_ts; tr_pnl[n_tr] = pnl; tr_qty[n_tr] = pos_qty; tr_dist[n_tr] = pos_dist_dbg; n_tr += 1
                    if pnl > 0:
                        gross_w += pnl
                    else:
                        gross_l += -pnl
                    riskm = pos_init_risk * qty_out
                    rmul = np.round((net / riskm) * 1e3) / 1e3 if riskm > 0 else 0.0
                    pos_qty -= qty_out
                    pos_entry_fee -= fee_part
                    pos_scaled = True
                    eq_after = cash + (c - pos_entry) * pos_qty * d
                    day2 = ts // 86_400_000
                    if day2 != day_key:
                        day_key = day2; day_start_eq = eq_after; day_realized = 0.0
                        consec_losses = 0; cooldown_until = 0.0; killed = False
                    r_hist[rh_head] = _clamp(rmul, -3.0, 5.0); rh_head = (rh_head + 1) % 30
                    if rh_n < 30:
                        rh_n += 1
                    if eq_after > health_peak:
                        health_peak = eq_after
                    health_dd = (health_peak - eq_after) / health_peak if health_peak > 0 else 0.0
                    if rh_n < 8:
                        hbase = 1.0
                    else:
                        s2 = 0.0
                        for q in range(rh_n):
                            s2 += r_hist[(rh_head - rh_n + q) % 30]
                        hbase = _clamp(1.0 + (s2 / rh_n) * 0.9, 0.4, 1.3)
                    health_scalar = _clamp(hbase * _clamp(1.0 - health_dd * 3.0, 0.3, 1.0), 0.3, 1.3)
                    day_realized += pnl
                    consec_losses = 0 if pnl > 0 else consec_losses + 1
                    if consec_losses >= int(p[P_MAXLOSS]):
                        cooldown_until = clock + p[P_COOL] * 60.0
                        consec_losses = 0
                    if day_start_eq > 0 and (-day_realized / day_start_eq) >= p[P_DAYLOSS]:
                        killed = True
            if exit_code == 1:
                px = c * (1.0 - slip) if pos_long else c * (1.0 + slip)
                fee = pos_qty * px * taker
                gross = (px - pos_entry) * pos_qty * d
                cash += gross - fee
                net = gross - (pos_entry_fee + fee)
                pnl = np.round(net * 1e8) / 1e8
                tr_ts[n_tr] = ts; tr_open[n_tr] = pos_open_ts; tr_pnl[n_tr] = pnl; tr_qty[n_tr] = pos_qty; tr_dist[n_tr] = pos_dist_dbg; n_tr += 1
                if pnl > 0:
                    gross_w += pnl
                else:
                    gross_l += -pnl
                rmul = np.round((net / planned_risk) * 1e3) / 1e3 if planned_risk > 0 else 0.0
                eq_after = cash
                day2 = ts // 86_400_000
                if day2 != day_key:
                    day_key = day2; day_start_eq = eq_after; day_realized = 0.0
                    consec_losses = 0; cooldown_until = 0.0; killed = False
                r_hist[rh_head] = _clamp(rmul, -3.0, 5.0); rh_head = (rh_head + 1) % 30
                if rh_n < 30:
                    rh_n += 1
                if eq_after > health_peak:
                    health_peak = eq_after
                health_dd = (health_peak - eq_after) / health_peak if health_peak > 0 else 0.0
                if rh_n < 8:
                    hbase = 1.0
                else:
                    s2 = 0.0
                    for q in range(rh_n):
                        s2 += r_hist[(rh_head - rh_n + q) % 30]
                    hbase = _clamp(1.0 + (s2 / rh_n) * 0.9, 0.4, 1.3)
                health_scalar = _clamp(hbase * _clamp(1.0 - health_dd * 3.0, 0.3, 1.0), 0.3, 1.3)
                day_realized += pnl
                consec_losses = 0 if pnl > 0 else consec_losses + 1
                if consec_losses >= int(p[P_MAXLOSS]):
                    cooldown_until = clock + p[P_COOL] * 60.0
                    consec_losses = 0
                if day_start_eq > 0 and (-day_realized / day_start_eq) >= p[P_DAYLOSS]:
                    killed = True
                has_pos = False
                bars_held = 0
        elif not pend:
            # entry decision (can_enter -> signal chain -> queue pending)
            eq2 = cash
            day2 = ts // 86_400_000
            if day2 != day_key:
                day_key = day2; day_start_eq = eq2; day_realized = 0.0
                consec_losses = 0; cooldown_until = 0.0; killed = False
            ok = (not killed) and (clock >= cooldown_until) and (0 < int(p[P_MAXPOS])) \
                and (ASSUMED_SPREAD_BPS <= p[P_MAXSPR]) and eq2 > 0
            if ok and abs(edge) >= threshold and p_win >= p[P_MINPW] and atr_pct > 0:
                predicted = beta * abs(edge) * atr_pct * math.sqrt(horizon)
                cost = fees_rt + (ASSUMED_SPREAD_BPS + 2.0 * slip_bps) / 10_000.0
                if predicted >= cost * p[P_COSTM]:
                    bias = feats[i, F_BIAS]
                    veto = p[P_MTFV] > 0 and abs(bias) >= p[P_MTFV] and bias * edge < 0
                    if not veto:
                        # regime gate
                        allow = False
                        scalp = False
                        if p[P_DISC] < 0.5:
                            allow = True
                            if p[P_ALIGNG] > 0.5 and reg <= 1:
                                al = feats[i, F_ALIGN]
                                if al * edge < 0 and abs(al) > 0.15:
                                    allow = False
                        elif reg <= 1:
                            er2 = feats[i, F_ER]
                            al = feats[i, F_ALIGN]
                            allow = er2 >= p[P_MINEFF] and al * edge > 0
                        elif reg == 2:
                            if p[P_TRANGE] > 0.5:
                                pctb = feats[i, F_PCTB]
                                b2 = p[P_RBAND]
                                allow = (pctb < b2 and edge > 0) or (pctb > 1.0 - b2 and edge < 0)
                                scalp = True
                        else:
                            allow = p[P_TVOL] > 0.5
                        if allow:
                            scalp = scalp or reg == 2
                            # measured payoff blend — shared by the EV floor
                            # and Kelly (mirror of risk.payoff_ratio)
                            prior_b = p[P_SERR] if scalp else p[P_ERR]
                            nw = 0
                            nl = 0
                            sw = 0.0
                            sl2 = 0.0
                            for q in range(rh_n):
                                rv = r_hist[(rh_head - rh_n + q) % 30]
                                if rv > 0:
                                    nw += 1
                                    sw += rv
                                elif rv < 0:
                                    nl += 1
                                    sl2 += -rv
                            b3 = prior_b
                            if nw >= 8 and nl >= 8:
                                measured = _clamp((sw / nw) / max(sl2 / nl, 1e-9), 0.6, 4.0)
                                wgt = _clamp(rh_n / 30.0, 0.0, 1.0)
                                b3 = (1.0 - wgt) * prior_b + wgt * measured
                            # EV floor — exact mirror of backtest.gate_ev:
                            # p(win) must clear breakeven at the measured
                            # payoff with real costs in stop-distance units
                            stop_pct = max(p[P_SLMIN] * atr_pct, 1e-9)
                            cost_r = (fees_rt + (ASSUMED_SPREAD_BPS + 2.0 * slip_bps) / 10_000.0) / stop_pct
                            bb0 = b3 if b3 > 0.1 else 0.1
                            need = (1.0 + cost_r) / (1.0 + bb0) + EV_MARGIN
                            if need > 0.92:
                                need = 0.92
                            if p_win >= need:
                                # kelly sizing
                                if p[P_USEK] > 0.5:
                                    bb = max(b3, 0.1)
                                    fK = p_win - (1.0 - p_win) / bb
                                    kel = 0.0 if fK <= 0 else _clamp(p[P_KF] * fK * 4.0, 0.25, 1.75)
                                else:
                                    kel = 1.0
                                # NOTE: python does NOT gate on kelly > 0 — sizing
                                # clamps eff_mult to >= 0.1, so a zero-Kelly signal
                                # still trades at floor size. Mirror that exactly.
                                size_mult = kel * health_scalar
                                pend = True
                                pend_long = edge > 0
                                pend_atr = atr
                                pend_reg = reg
                                pend_scalp = scalp
                                pend_mult = size_mult
                                pull = p[P_PULL]
                                if pull > 0 and not scalp and atr > 0:
                                    pend_taker = False
                                    pend_limit = c - (1.0 if pend_long else -1.0) * pull * atr
                                    pend_expires = i + int(p[P_MWAIT])
                                elif is_maker:
                                    pend_taker = False
                                    pend_limit = c * (1.0 - maker_off) if pend_long else c * (1.0 + maker_off)
                                    pend_expires = i + int(p[P_MWAIT])
                                else:
                                    pend_taker = True

        # ---------- equity record (rounded like the Python curve)
        eq_rec = cash + (((c - pos_entry) * pos_qty * (1.0 if pos_long else -1.0)) if has_pos else 0.0)
        eq_rec = np.round(eq_rec * 1e6) / 1e6
        if eq_rec > peak_eq:
            peak_eq = eq_rec
        if peak_eq > 0:
            ddv = (peak_eq - eq_rec) / peak_eq
            if ddv > max_dd:
                max_dd = ddv
        last_eq = eq_rec

    # forced close at history end
    if has_pos:
        i = n - 1
        c = feats[i, F_C]
        d = 1.0 if pos_long else -1.0
        px = c * (1.0 - slip) if pos_long else c * (1.0 + slip)
        fee = pos_qty * px * taker
        gross = (px - pos_entry) * pos_qty * d
        cash += gross - fee
        net = gross - (pos_entry_fee + fee)
        pnl = np.round(net * 1e8) / 1e8
        tr_ts[n_tr] = np.int64(feats[i, F_TS]); tr_open[n_tr] = pos_open_ts; tr_pnl[n_tr] = pnl; tr_qty[n_tr] = pos_qty; tr_dist[n_tr] = pos_dist_dbg; n_tr += 1
        if pnl > 0:
            gross_w += pnl
        else:
            gross_l += -pnl
        has_pos = False
        eq_rec = np.round(cash * 1e6) / 1e6
        if eq_rec > peak_eq:
            peak_eq = eq_rec
        if peak_eq > 0:
            ddv = (peak_eq - eq_rec) / peak_eq
            if ddv > max_dd:
                max_dd = ddv
        last_eq = eq_rec

    return n_tr, tr_ts[:n_tr], tr_open[:n_tr], tr_pnl[:n_tr], tr_qty[:n_tr], tr_dist[:n_tr], last_eq, max_dd, gross_w, gross_l, funding_paid, dbg_edge, dbg_pwin, dbg_thr


# parameter indices as module constants (numba needs literals-by-closure)
P_ETA = P["hedge_eta"]; P_FLOOR = P["weight_floor"]; P_HOR = P["horizon_bars"]
P_BTHR = P["base_threshold"]; P_TADAPT = P["threshold_adapt"]
P_TRATE = P["target_trades_per_hour"]; P_BPH = P["bars_per_hour"]
P_COSTM = P["cost_multiple"]; P_MINPW = P["min_p_win"]; P_KF = P["kelly_fraction"]
P_USEK = P["use_kelly"]; P_MINEFF = P["min_efficiency"]; P_MTFV = P["mtf_veto"]
P_TRANGE = P["trade_range"]; P_RBAND = P["range_band_edge"]; P_TVOL = P["trade_volatile"]
P_DISC = P["discipline"]; P_ALIGNG = P["trend_align_gate"]; P_ISMAKER = P["is_maker"]
P_MOFF = P["maker_offset_bps"]; P_MWAIT = P["maker_wait_bars"]; P_PULL = P["entry_pullback_atr"]
P_RPT = P["risk_per_trade"]; P_LEVMIN = P["min_leverage"]; P_LEVMAX = P["max_leverage"]
P_HARD = P["max_risk_hard_pct"]; P_SLMIN = P["sl_atr_min"]; P_SLMAX = P["sl_atr_max"]
P_TPCAP = P["tp_atr_cap"]; P_TRMIN = P["trail_atr_min"]; P_TRMAX = P["trail_atr_max"]
P_TTIGHT = P["trail_tighten"]; P_BERR = P["be_rr"]; P_BEOFF = P["be_offset_atr"]
P_GBRR = P["giveback_rr"]; P_GBFRAC = P["giveback_frac"]; P_HEDGEF = P["hold_edge_frac"]
P_ERR = P["expected_rr"]; P_TSTOP = P["time_stop_bars"]; P_STP = P["scalp_tp_atr"]
P_SSL = P["scalp_sl_atr"]; P_STIME = P["scalp_time_stop"]; P_SERR = P["scalp_expected_rr"]
P_MADV = P["maker_adverse_bps"]; P_MAXPOS = P["max_open_positions"]
P_DAYLOSS = P["max_daily_loss_pct"]; P_MAXLOSS = P["max_consecutive_losses"]
P_COOL = P["cooldown_minutes"]; P_MAXSPR = P["max_spread_bps"]
P_SORR = P["scaleout_rr"]; P_SOFRAC = P["scaleout_frac"]
P_TRSTREND = P["trail_scale_trend"]; P_TRSCHOP = P["trail_scale_chop"]


def kernel_fitness(ff, strat, risk, spec, taker: float, slip_bps: float,
                   interval: str, warmup: int = 300,
                   starting_balance: float = 10_000.0) -> dict:
    """Run the kernel over a prepared FeatureFrame and return _fitness-compatible
    stats. Raises if numba is unavailable — callers fall back to Python."""
    if not HAVE_NUMBA:
        raise RuntimeError("numba unavailable")
    from ..util import interval_ms
    feats, amat, regs = prep_fold(ff)
    pv = pack_params(strat, risk)
    pv[P_BPH] = 3_600_000 / interval_ms(interval)
    n_tr, tr_ts, tr_open, tr_pnl, tr_qty, tr_dist, eq, dd, gw, gl, funding, dbg_edge, dbg_pwin, dbg_thr = run_kernel(
        feats, amat, regs, pv, warmup, taker, float(getattr(spec, "maker_fee", 0.0002)),
        float(slip_bps), int(spec.qty_precision), float(spec.min_qty),
        float(spec.min_notional_usdt), starting_balance)
    stats = {
        "trades": int(n_tr),
        "total_return": round(eq / starting_balance - 1.0, 6) if starting_balance > 0 else 0.0,
        "max_drawdown": round(float(dd), 4),
        "profit_factor": round(gw / gl, 3) if gl > 0 else (999.0 if gw > 0 else 0.0),
        "win_rate": round(float(np.sum(tr_pnl > 0)) / n_tr, 4) if n_tr else 0.0,
        "total_pnl": round(float(np.sum(tr_pnl)), 6),
        "funding_paid": round(float(funding), 6),
        "equity": round(float(eq), 6),
    }
    return {"stats": stats, "trade_ts": tr_ts, "trade_open_ts": tr_open, "trade_pnl": tr_pnl, "trade_qty": tr_qty, "trade_dist": tr_dist, "dbg_edge": dbg_edge, "dbg_pwin": dbg_pwin, "dbg_thr": dbg_thr}
