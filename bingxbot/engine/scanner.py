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
import json
import logging
import os
import random
import time

import numpy as np

from ..config import ROOT
from ..util import clamp

log = logging.getLogger("radar")

UNIVERSE_PATH = ROOT / "data_cache" / "radar_universe.json"
UNIVERSE_TTL_S = 6 * 3600      # market caps drift slowly; refresh a few times a day
UNIVERSE_RETRY_S = 900         # min gap between fetch attempts (success or fail)
UNIVERSE_MIN_TOKENS = 30       # a smaller result is a broken response, not a universe
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

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
    """Format sanity: USDT-quoted, short all-letter base — excludes BingX's
    tokenized stock/commodity index products (NCSINASDAQ100..., NCCOXAG2USD...),
    leverage-multiplied listings (1000PEPE) and exotic long-name micro-caps.
    Format alone does NOT make a token reasonable — see MAJORS below."""
    if not symbol.endswith("-USDT"):
        return False
    base = symbol[:-5]
    return base.isalpha() and 2 <= len(base) <= 6


# The radar's eligibility universe: popular tokens only. Volume floors cannot
# express this — a pumping micro-cap out-trades ATOM every time, and extreme
# funding (the carry column) is exactly where squeezed junk lives — so
# eligibility is an explicit allowlist, not a statistic.
#
# The LIVE list is the CoinGecko top-100 by market cap, taken AS-IS (whatever
# is big enough to be top-100 is admitted — the only rows dropped are ones
# whose ticker can't match a clean BingX perp symbol anyway). Refreshed every
# few hours and cached to disk (see DynamicUniverse). MAJORS below is the
# OFFLINE FALLBACK used until the first successful fetch or when CoinGecko is
# unreachable. The user's configured symbols and the `radar_extra` setting are
# always admitted on top, so any deliberate choice extends the list.
MAJORS: frozenset[str] = frozenset({
    # L1 / L2 / payments
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "DOT", "ATOM", "NEAR",
    "APT", "SUI", "TON", "TRX", "LTC", "BCH", "XLM", "ALGO", "HBAR", "ICP",
    "FIL", "ETC", "VET", "EGLD", "FLOW", "MINA", "KAS", "SEI", "TIA", "INJ",
    "STX", "ROSE", "KAVA", "XTZ", "EOS", "KSM", "ZEC", "DASH", "NEO", "QNT",
    "IOTA", "CELO", "ARB", "OP", "POL", "MATIC", "STRK", "IMX", "MNT", "ZK",
    "TAO", "HYPE",
    # DeFi / infra / data
    "LINK", "UNI", "AAVE", "MKR", "CRV", "COMP", "SNX", "LDO", "RUNE", "GRT",
    "ENS", "DYDX", "GMX", "CAKE", "JUP", "PYTH", "JTO", "ENA", "ONDO", "WLD",
    "RNDR", "RENDER", "FET", "AR", "ZRO", "EIGEN", "PENDLE",
    # gaming / consumer (established mid-caps, not memes)
    "THETA", "AXS", "SAND", "MANA", "GALA", "CHZ", "ENJ",
})


def _base(symbol: str) -> str:
    return symbol[:-5] if symbol.endswith("-USDT") else symbol


def reasonable_perp(symbol: str, allowed: set[str] | None = None) -> bool:
    """Eligible for the radar: clean format AND in the allowed base set.
    `allowed` is the FULL admitted universe (dynamic CoinGecko list plus the
    user's own bases), already assembled by the caller; None falls back to the
    built-in MAJORS."""
    if not clean_perp(symbol):
        return False
    b = _base(symbol)
    return b in (allowed if allowed else MAJORS)


def parse_coingecko_universe(top_rows: list[dict]) -> set[str]:
    """Pure: CoinGecko /coins/markets payload -> admitted base-ticker set.
    The top-100 by market cap, AS-IS — everything CoinGecko gives us. Rows are
    only dropped when their ticker can't possibly match a clean BingX perp
    symbol (non-alpha or outside 2-6 letters, same as clean_perp). Unit-testable
    without a network."""
    out: set[str] = set()
    for r in top_rows or []:
        if not isinstance(r, dict):
            continue
        b = str(r.get("symbol", "")).strip().upper()
        if b and b.isalpha() and 2 <= len(b) <= 6:
            out.add(b)
    return out


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
                  min_qvol: float = MIN_QVOL_USDT,
                  allowed: set[str] | None = None) -> list[dict]:
    """Join funding + volume (+ optional 4h trend) into one ranked board.
    Only bases in `allowed` (the dynamic reasonable-token universe + the
    user's own; None -> built-in MAJORS) are eligible — memecoins never reach
    the board however hard they pump. Pure and synchronous — unit-testable
    without a network."""
    vol = {t["symbol"]: t for t in tickers}
    trend = trend or {}
    rows = []
    for p in premium:
        sym = p["symbol"]
        t = vol.get(sym)
        if t is None or not reasonable_perp(sym, allowed):
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


def top_volume_universe(tickers: list[dict], n: int = 10,
                        allowed: set[str] | None = None) -> list[str]:
    """The tuner's research universe: the ACTUAL top-N BingX perps by 24h USDT
    volume among admitted tokens (`allowed`; None -> built-in MAJORS — never a
    long-tail micro-cap, no matter its volume). Falls back to relaxing the
    VOLUME floor, never the eligibility list."""
    ok = [t for t in tickers if reasonable_perp(t["symbol"], allowed)]
    ok.sort(key=lambda t: t.get("quote_volume", 0.0), reverse=True)
    liquid = [t["symbol"] for t in ok if t.get("quote_volume", 0.0) >= UNIVERSE_MIN_QVOL]
    return liquid[:n] if len(liquid) >= 4 else [t["symbol"] for t in ok[:n]]


def demo_universe(seed: int | None = None) -> tuple[list[dict], list[dict]]:
    """Fabricated but plausible board for the synthetic feed — lets the Radar
    tab and carry desk run end-to-end with no exchange access. Majors only,
    matching the live policy; a couple of squeezed mid-cap MAJORS carry the
    juicy funding."""
    rng = random.Random(seed)
    names = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "AVAX-USDT", "SUI-USDT",
             "SEI-USDT", "LINK-USDT", "ARB-USDT", "OP-USDT", "TON-USDT", "NEAR-USDT"]
    premium, tickers = [], []
    for i, s in enumerate(names):
        hot = i in (5, 6)  # squeezed mid-cap majors with juicy funding
        fr = rng.uniform(0.0008, 0.0025) * rng.choice([1, -1]) if hot else rng.gauss(0.0001, 0.00012)
        px = rng.uniform(0.5, 60000)
        premium.append({"symbol": s, "mark": px, "funding_rate": round(fr, 6),
                        "next_funding_time": int(time.time() * 1000) + rng.randint(1, 8) * 3_600_000})
        tickers.append({"symbol": s, "last": px,
                        "quote_volume": rng.uniform(5e7, 9e8) if hot else rng.uniform(8e6, 9e8),
                        "change_pct": rng.gauss(0, 3.5)})
    return premium, tickers


class DynamicUniverse:
    """The radar's live eligibility list: the CoinGecko top-100 by market cap,
    as-is, cached to disk so a restart (or a CoinGecko outage) never leaves the
    radar blind — it falls back to the last good fetch, and to the built-in
    MAJORS before the first one ever."""

    def __init__(self, path=UNIVERSE_PATH):
        self.path = path
        self.bases: set[str] = set()
        self.fetched_ts = 0.0
        self._next_try = 0.0
        self.source = "builtin majors"
        self._load()

    def _load(self) -> None:
        try:
            d = json.loads(self.path.read_text())
            bases = {str(b).upper() for b in d.get("bases", [])}
            if len(bases) >= UNIVERSE_MIN_TOKENS:
                self.bases = bases
                self.fetched_ts = float(d.get("ts", 0.0))
                self.source = "coingecko (cached)"
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            pass

    def allowed(self) -> set[str]:
        return self.bases if self.bases else set(MAJORS)

    def age_s(self) -> float:
        return time.time() - self.fetched_ts if self.fetched_ts else -1.0

    async def maybe_refresh(self) -> bool:
        now = time.time()
        if now < self._next_try or (self.bases and now - self.fetched_ts < UNIVERSE_TTL_S):
            return False
        self._next_try = now + UNIVERSE_RETRY_S
        try:
            import aiohttp
            headers = {"User-Agent": "bingxbot/1.0"}
            key = os.getenv("COINGECKO_API_KEY", "").strip()
            if key:
                headers["x-cg-demo-api-key"] = key
            async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=15), headers=headers) as s:
                async with s.get(f"{COINGECKO_BASE}/coins/markets",
                                 params={"vs_currency": "usd", "order": "market_cap_desc",
                                         "per_page": "100", "page": "1",
                                         "sparkline": "false"}) as r1:
                    top = await r1.json() if r1.status == 200 else None
        except Exception as e:  # noqa: BLE001 — the radar must survive CoinGecko
            log.debug("universe fetch failed: %s", e)
            return False
        if not isinstance(top, list) or len(top) < 50:
            log.debug("universe fetch rejected (top=%s)", type(top).__name__)
            return False
        bases = parse_coingecko_universe(top)
        if len(bases) < UNIVERSE_MIN_TOKENS:
            return False
        self.bases = bases
        self.fetched_ts = time.time()
        self.source = "coingecko"
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"ts": self.fetched_ts, "bases": sorted(bases)}))
        except OSError:
            pass
        log.info("radar universe refreshed: %d tokens (CoinGecko top-100 by market cap)",
                 len(bases))
        return True


class MarketScanner:
    def __init__(self, orch):
        self.orch = orch
        self._task: asyncio.Task | None = None
        self.rows: list[dict] = []
        self.top_volume: list[str] = []   # the REAL top perps by 24h volume (tuner universe)
        self.universe = DynamicUniverse()
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

    def _extra_allowed(self) -> set[str]:
        """Base tickers admitted beyond the built-in MAJORS: the user's own
        traded symbols plus the user-owned radar_extra setting — a deliberate
        choice always overrides the allowlist."""
        cfg = getattr(self.orch, "cfg", None)
        raw = list(getattr(cfg, "symbols", []) or []) + list(getattr(cfg, "radar_extra", []) or [])
        return {str(s).strip().upper().removesuffix("-USDT") for s in raw if str(s).strip()}

    async def scan(self) -> list[dict]:
        rest = getattr(self.orch, "rest", None)
        if rest is not None:   # offline/synthetic runs stay fully offline
            await self.universe.maybe_refresh()
        # the admitted universe: the CoinGecko top-100 as-is (or built-in
        # majors before the first fetch), plus the user's deliberate choices.
        allowed = self.universe.allowed() | self._extra_allowed()
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
            # deep-dive 4h trend only on the interesting ADMITTED few (one
            # klines call each): harvestable-liquidity funding movers + majors —
            # never spend a call probing a token the board won't show anyway.
            by_f = sorted(premium, key=lambda p: abs(p.get("funding_rate", 0.0)), reverse=True)
            vol_ok = {t["symbol"] for t in tickers
                      if t.get("quote_volume", 0) >= CARRY_MIN_QVOL
                      and reasonable_perp(t["symbol"], allowed)}
            focus = [p["symbol"] for p in by_f if p["symbol"] in vol_ok][:TOP_FUNDING]
            focus += [s for s in top_volume_universe(tickers, TOP_VOLUME, allowed) if s not in focus]
            trend = {}
            for sym in focus:
                try:
                    kl = await rest.klines(sym, "4h", limit=KLINES_4H)
                    if kl:
                        trend[sym] = trend_read_4h(np.array([c.close for c in kl]))
                except Exception as e:  # noqa: BLE001
                    log.debug("radar 4h read %s: %s", sym, e)
        self.top_volume = top_volume_universe(tickers, 10, allowed)
        self.rows = rank_universe(premium, tickers, trend, allowed=allowed)[:24]
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
        age = self.universe.age_s()
        return {"ts": int(self.ts * 1000), "demo": self.demo, "scans": self.scans,
                "error": self.error, "rows": self.rows, "top_volume": self.top_volume,
                "universe": {"source": self.universe.source,
                             "count": len(self.universe.allowed()),
                             "age_min": int(age / 60) if age >= 0 else None}}
