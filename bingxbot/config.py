"""Configuration: dataclass tree, JSON persistence, env-based secrets.

Secrets (API keys) come only from the environment / .env file and are never
written to config.json. Everything else is runtime-editable from the UI and
persisted to config.json in the project root.
"""
from __future__ import annotations

import copy
import json
import os
import threading
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"

MODE_IDLE = "idle"
MODE_PAPER = "paper"
MODE_LIVE = "live"

FEED_BINGX = "bingx"
FEED_SYNTHETIC = "synthetic"


@dataclass
class ExchangeConfig:
    base_url: str = "https://open-api.bingx.com"
    ws_url: str = "wss://open-api-swap.bingx.com/swap-market"
    recv_window_ms: int = 5000
    taker_fee: float = 0.0005   # VIP0 defaults; refreshed from the API when keys exist
    maker_fee: float = 0.0002


@dataclass
class StrategyConfig:
    interval: str = "15m"           # SIGNAL timeframe. 1m is execution-only: the
                                    # reactive intra-bar scanner already times entries
                                    # inside the forming bar; the signal itself needs a
                                    # timeframe where edge/cost > 1 (15m-4h).
    warmup_bars: int = 350          # bars required before the brain may trade
    horizon_bars: int = 8           # bars over which alpha/desk calls are graded
    hedge_eta: float = 0.35         # multiplicative-weights learning rate (alphas)
    weight_floor: float = 0.05      # no alpha is ever fully muted
    base_threshold: float = 0.30    # |fused edge| needed to consider a trade
    threshold_adapt: bool = True    # auto-tune threshold toward target trade rate
    target_trades_per_hour: float = 2.5
    cost_multiple: float = 1.85     # predicted move must exceed round-trip cost x this
    micro_confirm: bool = True      # order-flow agreement gate at entry (live/paper)
    # clock trial (manual research switch): the tuner alternates cycles between
    # the live interval and trial_interval, each with its own gene pool, and
    # records TAGGED champions for both — evidence for "which bar clock earns
    # more", gathered in parallel. Trial-clock champions are never traded; to
    # act on the evidence the user switches `interval` deliberately.
    clock_trial: bool = False
    trial_interval: str = "5m"
    trend_align_gate: bool = True   # in trends, only trade with multi-TF alignment
    discipline: bool = True         # regime-appropriate entries (the big anti-bleed fix)
    min_efficiency: float = 0.35    # trend entries need this Kaufman efficiency ratio
    mtf_veto: float = 0.35          # HARD gate: never trade against a decided 15m/1h trend
    trade_range: bool = False       # opt-in: scalp range extremes (maker in/out). Off by
    range_band_edge: float = 0.15   # default — range mean-reversion is a weak edge; the
    trade_volatile: bool = False    # sit out VOLATILE/chop (pure fee bleed)
    min_p_win: float = 0.48         # refuse trades below this calibrated win prob
                                    # (trend systems profit at <50% via asymmetric R)
    use_kelly: bool = True          # size by fractional Kelly from P(win)
    kelly_fraction: float = 0.30    # fraction of full Kelly (conservative)
    entry_mode: str = "maker"       # maker (post-only, pays maker fee) | taker
    maker_offset_bps: float = 1.0   # how far inside the touch to rest the limit
    maker_wait_bars: int = 2        # bars to wait for a maker fill before cancelling
    entry_pullback_atr: float = 0.0  # trend entries: rest the limit this many ATRs
                                    # BEHIND price and let the pullback come to us
                                    # (0 = enter at the touch as before). Cheaper
                                    # fills on entries that retrace, missed trades
                                    # when the move runs — tuner-owned, so the
                                    # optimizer decides with data whether it pays.
    auto_tune: bool = True          # background walk-forward self-tuning
                                    # (cadence is self-managed: fast after a
                                    # promotion, duty-cycled on slow hosts)
    adopt_symbols: int = 3          # radar may adopt this many extra trending perps
                                    # (2 user symbols + 3 adopted = up to 5 brains
                                    # hunting, so the 3-position cap has real
                                    # candidates even while brains are in trades)


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.008       # target equity fraction at risk if the stop hits
    min_leverage: int = 2               # floor for the EXCHANGE margin setting only —
                                        # it never inflates position size beyond what
                                        # risk_per_trade allows (sizing is pure risk-based)
    max_leverage: int = 7               # hard ceiling on size AND the margin setting
    max_risk_hard_pct: float = 0.035    # hard cap on any single trade's loss-at-stop
    margin_mode: str = "ISOLATED"
    # --- adaptive exit geometry (let winners run, cut losers) ---
    sl_atr_min: float = 1.5             # initial stop: no tighter than this x ATR (room to breathe)
    sl_atr_max: float = 2.8             # ...and no wider than this (structure-clamped between)
    tp_atr_cap: float = 0.0             # 0 = no fixed target; exit only via trail/edge
    trail_atr_min: float = 1.6          # chandelier trail width in chop
    trail_atr_max: float = 3.8          # ...widened in clean trends (Kaufman ER)
    trail_tighten: float = 0.55         # ratchet trail this much tighter as profit grows
    be_rr: float = 0.8                  # move stop to breakeven at this R
    be_offset_atr: float = 0.15         # breakeven sits entry + this x ATR (covers fees)
    giveback_rr: float = 2.5            # once past this R, protect the open profit
    giveback_frac: float = 0.5          # ...exit if price retraces this fraction of MFE
    scaleout_rr: float = 0.0            # bank scaleout_frac of a trend trade at this R
                                        # (0 = off; tuner-owned — variance reducer)
    scaleout_frac: float = 0.5          # fraction banked when the scale-out fires
    trail_scale_trend: float = 1.0      # regime-conditional trail width scales
    trail_scale_chop: float = 1.0       # (tuner-owned: wide in trends, tight in chop)
    hold_edge_frac: float = 0.7         # exit if brain edge flips past this x threshold
    expected_rr: float = 2.2            # assumed winner:loser ratio for Kelly sizing (trend)
    time_stop_bars: int = 120           # long backstop; trends need room to develop
    # --- scalp style (range/mean-reversion): passive maker target, quick ---
    scalp_tp_atr: float = 1.1           # passive limit target distance (x ATR)
    scalp_sl_atr: float = 1.0           # tight stop for scalps (x ATR)
    scalp_time_stop: int = 24           # scalps are quick; bail if it stalls
    scalp_expected_rr: float = 1.1      # ~1:1 target:stop for Kelly on scalps
    maker_adverse_bps: float = 0.4      # honest adverse-selection penalty on maker fills
    max_open_positions: int = 3              # up to 3 concurrent positions, each on a
                                             # DIFFERENT token (one position per symbol
                                             # is structural: portfolio + brokers refuse
                                             # a second open on a held symbol)
    correlation_haircut: float = 0.65        # shrink a same-direction add across symbols
    max_net_exposure: float = 2.5            # cap on summed same-direction notional / equity
    max_position_notional_pct: float = 0.35  # of equity x leverage, per position
    max_daily_loss_pct: float = 0.05         # kill switch: flatten + halt for the day
    max_consecutive_losses: int = 8          # cooldown trigger
    cooldown_minutes: int = 45
    max_spread_bps: float = 6.0              # refuse entries into a wide spread


@dataclass
class CarryConfig:
    """Funding-carry desk: harvest extreme perp funding as its own strategy.
    Enters the RECEIVING side of stretched funding when the 4h trend doesn't
    oppose it; small size, low leverage, always stopped. The one edge in this
    codebase that does not require out-predicting anyone — the funding print
    is public and the payment is mechanical."""
    enabled: bool = True
    max_positions: int = 1          # carry's own cap (also counts toward the global cap)
    min_apr: float = 0.35           # enter when |funding| annualized >= this (35% APR)
    exit_apr: float = 0.10          # exit once funding has normalized below this
    risk_frac: float = 0.5          # fraction of risk_per_trade to use per carry trade
    leverage: int = 2               # fixed low leverage — carry is a yield trade
    stop_atr_4h: float = 2.5        # stop distance in 4h ATRs
    max_hold_hours: int = 48        # ~6 funding windows: more prints per (maker)
                                    # fee paid — the stop + trend veto carry the risk
    trend_veto_er: float = 0.35     # skip if the 4h trend opposes the side this strongly


@dataclass
class PaperConfig:
    starting_balance: float = 10_000.0
    slippage_bps: float = 1.0  # also stands in for order latency cost


@dataclass
class ServerConfig:
    host: str = field(default_factory=lambda: os.getenv("BOT_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("BOT_PORT", "8420")))


# Bump when the auto-managed parameter defaults change in a way that should
# override values persisted by an older build. On load, a config written by an
# older version keeps only the user-owned settings; the tuner-owned params are
# reset to current defaults (then the auto-tuner evolves from there).
CONFIG_VERSION = 7

# Top-level settings the user owns — everything else is auto-managed by the
# tuner and reset to code defaults when migrating an older config.
USER_OWNED_TOP = {"symbols", "radar_extra", "mode", "feed", "allow_live", "data_dir", "log_level"}


@dataclass
class TapeConfig:
    """Recording of our OWN BingX market data (trades + best bid/ask). BingX
    publishes no historical tick data — this archive is the only way it ever
    exists. Live feed only; capped on disk; crash-safe (see data/tape.py)."""
    enabled: bool = True
    max_disk_mb: int = 2000
    book_ms: int = 250            # min gap between recorded book-top rows per symbol


@dataclass
class BotConfig:
    version: int = CONFIG_VERSION
    symbols: list[str] = field(default_factory=lambda: ["BTC-USDT", "ETH-USDT"])
    radar_extra: list[str] = field(default_factory=list)  # extra tokens the radar may
                                        # consider beyond the built-in majors allowlist
                                        # (e.g. ["DOGE"] to re-admit one deliberately)
    mode: str = MODE_PAPER              # idle | paper | live
    feed: str = FEED_BINGX              # bingx | synthetic (offline demo)
    allow_live: bool = False            # hard gate: live orders refused unless true
    data_dir: str = "data_cache"
    log_level: str = "INFO"
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    carry: CarryConfig = field(default_factory=CarryConfig)
    tape: TapeConfig = field(default_factory=TapeConfig)
    paper: PaperConfig = field(default_factory=PaperConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    @property
    def api_key(self) -> str:
        return os.getenv("BINGX_API_KEY", "")

    @property
    def api_secret(self) -> str:
        return os.getenv("BINGX_API_SECRET", "")

    def has_keys(self) -> bool:
        return bool(self.api_key and self.api_secret)


_lock = threading.Lock()


def _merge_into(dc: Any, data: dict) -> None:
    """Recursively apply known keys from `data` onto dataclass `dc`."""
    for f in fields(dc):
        if f.name not in data:
            continue
        val = data[f.name]
        cur = getattr(dc, f.name)
        if is_dataclass(cur) and isinstance(val, dict):
            _merge_into(cur, val)
        elif isinstance(cur, bool):
            setattr(dc, f.name, bool(val))
        elif isinstance(cur, int) and not isinstance(val, bool):
            setattr(dc, f.name, int(val))
        elif isinstance(cur, float):
            setattr(dc, f.name, float(val))
        elif isinstance(cur, list):
            if isinstance(val, list):
                upper = f.name in ("symbols", "radar_extra")
                setattr(dc, f.name, [str(x).strip().upper() if upper else x for x in val if str(x).strip()])
        elif isinstance(cur, str):
            setattr(dc, f.name, str(val))


_NESTED_USER_OWNED = {
    "exchange": {"base_url", "ws_url", "recv_window_ms", "taker_fee", "maker_fee"},
    "server": {"host", "port"},
    "paper": {"starting_balance", "slippage_bps"},
    # auto_tune is user-owned (Settings toggle) — a migration must not silently
    # re-enable a tuner the user turned off.
    "strategy": {"interval", "warmup_bars", "adopt_symbols", "auto_tune",
                 "clock_trial", "trial_interval"},
    "tape": {"enabled", "max_disk_mb", "book_ms"},
    # leverage band and max_open_positions intentionally omitted so migrating an
    # old config picks up the current defaults (2-7x band, 3 concurrent tokens);
    # both stay UI-editable afterwards. max_risk_hard_pct is UI-editable and kept.
    "risk": {"max_daily_loss_pct", "max_risk_hard_pct", "margin_mode",
             "max_spread_bps", "max_consecutive_losses", "cooldown_minutes"},
    "carry": {"enabled", "max_positions"},
}


def _filter_user_owned(data: dict) -> dict:
    """Keep only user-owned keys from a raw config dict (drop stale tuner params)."""
    out = {k: v for k, v in data.items() if k in USER_OWNED_TOP}
    for group, allowed in _NESTED_USER_OWNED.items():
        if isinstance(data.get(group), dict):
            sub = {k: v for k, v in data[group].items() if k in allowed}
            if sub:
                out[group] = sub
    return out


def load_config(path: Path = CONFIG_PATH) -> BotConfig:
    cfg = BotConfig()
    if path.exists():
        try:
            raw = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return cfg  # fall back to defaults rather than refuse to boot
        if raw.get("version") == CONFIG_VERSION:
            _merge_into(cfg, raw)
        else:
            # migrate: keep the user's settings, reset tuner-owned params to
            # current defaults, and persist the upgraded file.
            _merge_into(cfg, _filter_user_owned(raw))
            # v6 strategic migration: 1m is no longer a SIGNAL timeframe (it's
            # execution-only — the reactive scanner). A preserved 1m interval is
            # upgraded once to the new 15m default; still user-editable after.
            if raw.get("version", 0) < 6 and cfg.strategy.interval == "1m":
                cfg.strategy.interval = "15m"
            save_config(cfg, path)
    return cfg


def save_config(cfg: BotConfig, path: Path = CONFIG_PATH) -> None:
    with _lock:
        path.write_text(json.dumps(asdict(cfg), indent=2))


def update_config(cfg: BotConfig, patch: dict) -> BotConfig:
    """Apply a partial update (from the UI) and persist. Returns same object."""
    protected = {"mode"}  # mode changes go through the engine, not raw config writes
    clean = {k: v for k, v in patch.items() if k not in protected}
    _merge_into(cfg, clean)
    save_config(cfg)
    return cfg


def config_public_dict(cfg: BotConfig) -> dict:
    """Config as a dict safe to ship to the browser (no secrets)."""
    d = asdict(cfg)
    d["has_keys"] = cfg.has_keys()
    return d
