"""Phase-shifted signal bars: decide more often WITHOUT deciding on partial data.

The engine's signal timeframe (15m) has always doubled as its decision clock, so
a setup that becomes true at :01 waits until :15 to be acted on. Those are two
different things welded together, and only one of them has to be 15 minutes.

The obvious shortcut — score the bar while it is still forming — is the one that
already cost real money. A half-built candle has not had time to make its range
yet, so its ATR reads low: measured -20% at a third formed, -7.7% at two thirds.
ATR is what sizes the trade (`qty = max_risk / stop_dist`, and stop_dist is an
ATR multiple), so an understated ATR gives a stop that much too tight AND a
position ~22% too large, at once, on a signal built from incomplete data. Only
the volatility features break that way; the rest are fine. That is a mechanical
error, not bad luck.

A phase-shifted window has no partial candle anywhere in it. A 15m window ending
at 10:07 is fifteen FINISHED 1m bars, and it is the same kind of object as the
one ending at 10:00 — measured across all fifteen offsets: atr_pct within +0.6%,
mtf_bias distribution identical (sd 0.880 vs 0.880), veto-gate rate 98.4% vs
98.4%. `test_phases.py` freezes exactly that, because it is the property the
whole idea rests on.

So: build the signal bar from finished minutes, ending wherever we like. Same
15m signal quality, a decision every minute.
"""
from __future__ import annotations

import numpy as np

MINUTE_MS = 60_000
FIELDS = ("ts", "open", "high", "low", "close", "volume")


def phase_of(ts_ms: int, factor: int) -> int:
    """Which phase a window OPENING at ts belongs to. Phase 0 is the epoch grid
    (`ts % interval == 0`), i.e. exactly the bars the exchange itself publishes."""
    return int((ts_ms // MINUTE_MS) % factor)


def phase_arrays(minute: dict[str, np.ndarray], factor: int, phase: int,
                 max_windows: int | None = None) -> dict[str, np.ndarray]:
    """Aggregate 1m bars into `factor`-minute windows opening on `phase`.

    Windows are selected by TIMESTAMP, never by array position, so a gap in the
    minute feed drops the windows it touches instead of silently splicing two
    sides of the gap into one bogus candle. Only COMPLETE windows are returned —
    that is the whole point, and it is why the trailing in-progress window is
    always discarded rather than emitted early.

    `phase=0` reproduces the exchange's own epoch-aligned bars (asserted in
    tests against native klines), so turning phases on cannot move the bar the
    engine already trades.
    """
    if factor < 1:
        raise ValueError("factor must be >= 1")
    phase %= factor
    ts = np.asarray(minute["ts"], dtype=np.int64)
    n = ts.size
    if n < factor:
        return {k: np.asarray([], dtype=np.float64) for k in FIELDS}

    span = factor * MINUTE_MS
    # a window may OPEN only on its phase's minutes
    opens = np.flatnonzero((ts % span) == (phase * MINUTE_MS))
    opens = opens[opens + factor <= n]
    if opens.size == 0:
        return {k: np.asarray([], dtype=np.float64) for k in FIELDS}
    # ...and only if all `factor` minutes are actually present and contiguous:
    # last ts must sit exactly one span minus one minute after the first.
    contiguous = ts[opens + factor - 1] - ts[opens] == span - MINUTE_MS
    opens = opens[contiguous]
    if opens.size == 0:
        return {k: np.asarray([], dtype=np.float64) for k in FIELDS}
    if max_windows is not None and opens.size > max_windows:
        opens = opens[-max_windows:]

    o = np.asarray(minute["open"], dtype=np.float64)
    h = np.asarray(minute["high"], dtype=np.float64)
    lo = np.asarray(minute["low"], dtype=np.float64)
    c = np.asarray(minute["close"], dtype=np.float64)
    v = np.asarray(minute["volume"], dtype=np.float64)
    # one row per window, `factor` columns: reduce along the columns
    idx = opens[:, None] + np.arange(factor)[None, :]
    return {
        "ts": ts[opens].astype(np.float64),
        "open": o[opens],
        "high": h[idx].max(axis=1),
        "low": lo[idx].min(axis=1),
        "close": c[opens + factor - 1],
        "volume": v[idx].sum(axis=1),
    }


def minutes_needed(factor: int, windows: int) -> int:
    """1m bars to seed `windows` complete signal bars for every phase. One extra
    span covers the offset: phase 14's first window starts 14 minutes after
    phase 0's."""
    return factor * (windows + 1)
