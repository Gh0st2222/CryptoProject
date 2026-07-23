"""Binance Vision historical klines: the free multi-year archive that lets
champions be stress-tested across regimes our 90-day BingX window never saw.

Why Binance data for a BingX bot: BTC on Binance and BTC on BingX are the same
asset arbitraged to within basis points, so for SIGNAL-level validation ("do
these parameters make money on this price action?") the venue doesn't matter —
and Binance publishes complete, curated history (data.binance.vision) while
BingX publishes none. Execution truth (fees, funding, spreads) stays BingX:
the gauntlet runs Binance PRICES through the same portfolio simulator with
BingX fee assumptions, exactly like every other backtest here.

Everything is disk-cached and immutable: a finished month never changes, so a
window is downloaded once, forever. Every failure path returns None/{} — the
gauntlet is evidence, never a dependency; no internet, no gauntlet, no harm.
"""
from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

from ..config import ROOT
from ..exchange.models import Candle

log = logging.getLogger("binance_hist")

BINANCE_VISION = "https://data.binance.vision/data/futures/um/monthly/klines"
CACHE_DIR = ROOT / "data_cache" / "binance"
FETCH_TIMEOUT_S = 30

# Regime eras for the champion gauntlet: five distinct market personalities,
# three complete months each — long enough for a real trading sample, small
# enough (~200KB/symbol/month) that the whole gauntlet is a few MB on disk.
# Immutable by construction: only finished calendar months, never the present.
GAUNTLET_WINDOWS: list[tuple[str, list[str]]] = [
    ("2021Q4 blow-off top", ["2021-10", "2021-11", "2021-12"]),
    ("2022 crash",          ["2022-05", "2022-06", "2022-07"]),
    ("2023 chop",           ["2023-06", "2023-07", "2023-08"]),
    ("2024 bull",           ["2024-03", "2024-04", "2024-05"]),
    ("2025 range",          ["2025-04", "2025-05", "2025-06"]),
    ("2026 recent",         ["2026-01", "2026-02", "2026-03"]),
]


def bx_to_binance(symbol: str) -> str:
    """BingX 'BTC-USDT' -> Binance 'BTCUSDT'."""
    return symbol.replace("-", "").upper()


def _norm_ts(v: float) -> int:
    """Binance archives have shipped seconds, milliseconds AND microseconds
    (spot moved to µs in 2025). Normalize by magnitude — robust to the next
    silent format change instead of trusting an announcement."""
    v = float(v)
    if v > 1e15:        # microseconds
        return int(v // 1000)
    if v > 1e12:        # milliseconds
        return int(v)
    return int(v * 1000)  # seconds


def parse_kline_csv(text: str) -> list[Candle]:
    """Binance kline CSV -> closed Candles. Tolerates the header row newer
    files carry and skips any malformed line — an archive parser must never
    crash the tuner over one bad row."""
    out: list[Candle] = []
    for line in text.splitlines():
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        try:
            ts = _norm_ts(float(parts[0]))
            out.append(Candle(ts=ts, open=float(parts[1]), high=float(parts[2]),
                              low=float(parts[3]), close=float(parts[4]),
                              volume=float(parts[5]), closed=True))
        except ValueError:
            continue   # header row or junk
    out.sort(key=lambda c: c.ts)
    return out


def month_cache_path(symbol_bx: str, interval: str, ym: str, cache_dir: Path = CACHE_DIR) -> Path:
    return cache_dir / f"{bx_to_binance(symbol_bx)}-{interval}-{ym}.csv"


async def fetch_month(symbol_bx: str, interval: str, ym: str,
                      cache_dir: Path = CACHE_DIR) -> list[Candle] | None:
    """One symbol-month of klines: disk cache first (immutable — a finished
    month never changes), else download the Vision ZIP, extract, cache, parse.
    Returns None on ANY failure — callers treat missing months as 'window not
    available', never as an error."""
    path = month_cache_path(symbol_bx, interval, ym, cache_dir)
    try:
        if path.exists():
            return parse_kline_csv(path.read_text())
    except OSError:
        pass
    bsym = bx_to_binance(symbol_bx)
    url = f"{BINANCE_VISION}/{bsym}/{interval}/{bsym}-{interval}-{ym}.zip"
    try:
        import aiohttp
        async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=FETCH_TIMEOUT_S)) as s:
            async with s.get(url) as r:
                if r.status != 200:
                    log.debug("binance archive %s: HTTP %d", url, r.status)
                    return None
                blob = await r.read()
        zf = zipfile.ZipFile(io.BytesIO(blob))
        names = zf.namelist()
        if not names:
            return None
        text = zf.read(names[0]).decode("utf-8", errors="replace")
        candles = parse_kline_csv(text)
        if not candles:
            return None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        except OSError:
            pass   # cache is an optimization; the parse already succeeded
        return candles
    except Exception as e:  # noqa: BLE001 — the archive is evidence, never a dependency
        log.debug("binance archive fetch %s %s %s failed: %s", symbol_bx, interval, ym, e)
        return None


async def load_window(symbols_bx: list[str], interval: str, months: list[str],
                      min_bars: int = 900,
                      cache_dir: Path = CACHE_DIR) -> dict[str, list[Candle]]:
    """A gauntlet window: each symbol's months concatenated in order, keyed by
    the BINGX symbol name (so specs/fees resolve downstream). Symbols with
    missing or thin data are dropped; an empty dict means 'window unavailable'
    and the caller simply skips it."""
    out: dict[str, list[Candle]] = {}
    for sym in symbols_bx:
        series: list[Candle] = []
        ok = True
        for ym in months:
            got = await fetch_month(sym, interval, ym, cache_dir)
            if not got:
                ok = False
                break
            series.extend(got)
        if ok and len(series) >= min_bars:
            series.sort(key=lambda c: c.ts)
            out[sym] = series
    return out
