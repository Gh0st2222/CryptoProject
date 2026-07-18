"""Market Radar: a universe scanner over the whole BingX perp board.

Two symbols is a tiny opportunity surface — the radar widens it without
loosening a single standard. Every cycle it pulls funding + 24h stats for
EVERY perp in two cheap calls, ranks the board by the two edges a
retail-latency bot can actually collect:

  * CARRY  — |funding| annualized. Extreme funding is a mechanical, public
             payment to whoever takes the other side; no prediction needed.
  * TREND  — 4h trend quality (Kaufman ER + EMA-stack direction) on the top
             candidates, fetched sparingly (one klines call each).

The ranked board feeds the UI Radar tab and the funding-carry desk. With a
synthetic feed it fabricates a plausible demo board so the whole pipeline
works offline.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time

import numpy as np

from ..util import clamp

log = logging.getLogger("radar")

SCAN_EVERY_S = 240          # full board refresh cadence
MIN_QVOL_USDT = 5_000_000   # display-board floor (can't exit = not worth showing)
CARRY_MIN_QVOL = 25_000_000  # HARVESTABLE carry needs real liquidity, not a 6M meme
TREND_MIN_QVOL = 25_000_000  # ...and so does an adoptable trend
UNIVERSE_MIN_QVOL = 50_000_000  # the tuner's research universe: majors only
APR_SANITY_CAP = 5.0        # |funding| above 500% APR = degenerate listing, not an edge
TOP_FUNDING = 12            # deep-dive this many by |funding|...
TOP_VOLUME = 8              # ...plus this many by volume
KLINES_4H = 120             # bars for the 4h trend read (~20 days)
FUNDING_WINDOWS_PER_YEAR = 3 * 365


def clean_perp(symbol: str) -> bool:
    """A 'real' crypto perp we'd let the machine trade or study: USDT-quoted,
    short all-letter base — which excludes BingX's tokenized stock/commodity
    index products (NCSINASDAQ100..., NCCOXAG2USD...), leverage-multiplied
    listings (1000PEPE) and the exotic long-name micro-caps."""
    if not symbol.endswith("-USDT"):
        return False
    base = symbol[:-5]
    return base.isalpha() and 2 <= len(base) <= 6


def annualize_funding(rate_8h: float) -> float:
    """A funding print is per-8h; three windows a day, compounding ignored."""
    return rate_8h * FUNDING_WINDOWS_PER_YEAR


def trend_read_4h(closes: np.ndarray) -> dict:
    """Compact 4h trend snapshot from raw closes: Kaufman ER (quality),
    EMA-stack direction, and ATR-ish volatility for stop geometry."""
    n = len(closes)
    if n < 40:
        return {"er": 0.0, "dir": 0, "atr_pct": 0.0}
    c = closes
    net = abs(float(c[-1] - c[-21]))
    path = float(np.sum(np.abs(np.diff(c[-21:])))) or 1e-12
    er = clamp(net / path, 0.0, 1.0)
    # EMA 21 vs 55 by simple exponential smoothing
    a21, a55 = 2 / 22, 2 / 56
    e21 = e55 = float(c[0])
    for x in c:
        e21 += a21 * (float(x) - e21)
        e55 += a55 * (float(x) - e55)
    d = 1 if e21 > e55 * 1.0005 else (-1 if e21 < e55 * 0.9995 else 0)
    rets = np.abs(np.diff(c[-31:]) / np.maximum(c[-31:-1], 1e-12))
    return {"er": round(er, 3), "dir": d, "atr_pct": round(float(np.mean(rets)) * 1.6, 5)}


def rank_universe(premium: list[dict], tickers: list[dict],
                  trend: dict[str, dict] | None = None,
                  min_qvol: float = MIN_QVOL_USDT) -> list[dict]:
    """Join funding + volume (+ optional 4h trend) into one ranked board.
    Pure and synchronous — unit-testable without a network."""
    vol = {t["symbol"]: t for t in tickers}
    trend = trend or {}
    rows = []
    for p in premium:
        sym = p["symbol"]
        t = vol.get(sym)
        if t is None or not clean_perp(sym):
            continue
        qv = t.get("quote_volume", 0.0)
        if qv < min_qvol:
            continue
        apr = annualize_funding(p.get("funding_rate", 0.0))
        tr = trend.get(sym, {})
        er, tdir = tr.get("er", 0.0), tr.get("dir", 0)
        # receiving side of the funding payment (longs pay shorts when +)
        side = "SHORT" if apr > 0 else "LONG"
        trend_score = clamp(er / 0.5, 0.0, 1.0) if tdir != 0 else 0.0
        # HARVESTABLE carry only: real liquidity and a sane rate. A 6M-volume
        # meme at -1000% APR is a rug in progress, not an edge — you cannot
        # size into it, cannot exit it, and the print exists BECAUSE it's junk.
        harvestable = qv >= CARRY_MIN_QVOL and abs(apr) <= APR_SANITY_CAP
        if abs(apr) >= 0.20 and harvestable:
            kind = "carry"
        elif trend_score >= 0.5 and qv >= TREND_MIN_QVOL:
            kind = "trend"
        else:
            kind = "watch"
        # score reflects what the desk can actually COLLECT — junk funding no
        # longer floats to the top of a board titled "harvestable edge".
        carry_score = clamp(abs(apr) / 0.60, 0.0, 1.0) if harvestable else 0.0
        rows.append({
            "symbol": sym,
            "mark": p.get("mark", t.get("last", 0.0)),
            "funding_rate": p.get("funding_rate", 0.0),
            "funding_apr": round(apr, 4),
            "next_funding_time": p.get("next_funding_time", 0),
            "quote_volume": qv,
            "change_24h": t.get("change_pct", 0.0),
            "er_4h": er, "dir_4h": tdir, "atr_pct_4h": tr.get("atr_pct", 0.0),
            "carry_side": side,
            "kind": kind,
            "score": round(0.65 * carry_score + 0.35 * trend_score, 4),
        })
    rows.sort(key=lambda r: (r["score"], abs(r["funding_apr"])), reverse=True)
    return rows


def top_volume_universe(tickers: list[dict], n: int = 10) -> list[str]:
    """The tuner's research universe: the ACTUAL top-N BingX perps by 24h USDT
    volume — clean majors only (no index products, no long-tail memes). Falls
    back to relaxing the floor rather than returning nothing."""
    clean = [t for t in tickers if clean_perp(t["symbol"])]
    clean.sort(key=lambda t: t.get("quote_volume", 0.0), reverse=True)
    majors = [t["symbol"] for t in clean if t.get("quote_volume", 0.0) >= UNIVERSE_MIN_QVOL]
    return majors[:n] if len(majors) >= 4 else [t["symbol"] for t in clean[:n]]


def demo_universe(seed: int | None = None) -> tuple[list[dict], list[dict]]:
    """Fabricated but plausible board for the synthetic feed — lets the Radar
    tab and carry desk run end-to-end with no exchange access."""
    rng = random.Random(seed)
    names = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "DOGE-USDT", "XRP-USDT", "PEPE-USDT",
             "WIF-USDT", "AVAX-USDT", "LINK-USDT", "ARB-USDT", "OP-USDT", "SUI-USDT"]
    premium, tickers = [], []
    for i, s in enumerate(names):
        hot = i in (5, 6)  # a couple of squeezed mid-caps with juicy funding
        fr = rng.uniform(0.0008, 0.0025) * rng.choice([1, -1]) if hot else rng.gauss(0.0001, 0.00012)
        px = rng.uniform(0.5, 60000)
        premium.append({"symbol": s, "mark": px, "funding_rate": round(fr, 6),
                        "next_funding_time": int(time.time() * 1000) + rng.randint(1, 8) * 3_600_000})
        tickers.append({"symbol": s, "last": px,
                        "quote_volume": rng.uniform(8e6, 9e8),
                        "change_pct": rng.gauss(0, 3.5)})
    return premium, tickers


class MarketScanner:
    def __init__(self, orch):
        self.orch = orch
        self._task: asyncio.Task | None = None
        self.rows: list[dict] = []
        self.top_volume: list[str] = []   # the REAL top perps by 24h volume (tuner universe)
        self.ts = 0.0
        self.demo = False
        self.scans = 0
        self.error = ""

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(), name="radar")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def _loop(self) -> None:
        await asyncio.sleep(6)
        while True:
            try:
                await self.scan()
            except Exception as e:  # noqa: BLE001 — the radar must never kill the app
                self.error = str(e)
                log.warning("radar scan failed: %s", e)
            await asyncio.sleep(SCAN_EVERY_S)

    async def scan(self) -> list[dict]:
        rest = getattr(self.orch, "rest", None)
        if rest is None:
            premium, tickers = demo_universe(seed=int(time.time() // SCAN_EVERY_S))
            self.demo = True
            trend = {p["symbol"]: {"er": random.random() * 0.6,
                                   "dir": random.choice([-1, 0, 1]),
                                   "atr_pct": random.uniform(0.004, 0.02)}
                     for p in premium}
        else:
            self.demo = False
            premium = await rest.premium_index_all()
            tickers = await rest.tickers_24h()
            # deep-dive 4h trend only on the interesting CLEAN few (one klines
            # call each): harvestable-liquidity funding movers + the majors.
            by_f = sorted(premium, key=lambda p: abs(p.get("funding_rate", 0.0)), reverse=True)
            vol_ok = {t["symbol"] for t in tickers
                      if t.get("quote_volume", 0) >= CARRY_MIN_QVOL and clean_perp(t["symbol"])}
            focus = [p["symbol"] for p in by_f if p["symbol"] in vol_ok][:TOP_FUNDING]
            focus += [s for s in top_volume_universe(tickers, TOP_VOLUME) if s not in focus]
            trend = {}
            for sym in focus:
                try:
                    kl = await rest.klines(sym, "4h", limit=KLINES_4H)
                    if kl:
                        trend[sym] = trend_read_4h(np.array([c.close for c in kl]))
                except Exception as e:  # noqa: BLE001
                    log.debug("radar 4h read %s: %s", sym, e)
        self.top_volume = top_volume_universe(tickers, 10)
        self.rows = rank_universe(premium, tickers, trend)[:24]
        self.ts = time.time()
        self.scans += 1
        self.error = ""
        maybe_adopt = getattr(self.orch, "maybe_adopt", None)
        if maybe_adopt is not None:
            try:
                await maybe_adopt()   # radar picks feed the trend engine
            except Exception as e:  # noqa: BLE001
                log.warning("adoption pass failed: %s", e)
        if self.orch._notify:
            await self.orch._notify("radar")
        return self.rows

    def snapshot(self) -> dict:
        return {"ts": int(self.ts * 1000), "demo": self.demo, "scans": self.scans,
                "error": self.error, "rows": self.rows, "top_volume": self.top_volume}
