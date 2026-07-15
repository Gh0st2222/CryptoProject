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
    interval: str = "5m"            # 15m/5m recommended; 1m only with maker entries
    warmup_bars: int = 350          # bars required before the brain may trade
    horizon_bars: int = 8           # bars over which alpha/desk calls are graded
    hedge_eta: float = 0.35         # multiplicative-weights learning rate (alphas)
    weight_floor: float = 0.05      # no alpha is ever fully muted
    base_threshold: float = 0.30    # |fused edge| needed to consider a trade
    threshold_adapt: bool = True    # auto-tune threshold toward target trade rate
    target_trades_per_hour: float = 1.5
    cost_multiple: float = 2.0      # predicted move must exceed round-trip cost x this
    micro_confirm: bool = True      # order-flow agreement gate at entry (live/paper)
    trend_align_gate: bool = True   # in trends, only trade with multi-TF alignment
    discipline: bool = True         # regime-appropriate entries (the big anti-bleed fix)
    min_efficiency: float = 0.35    # trend entries need this Kaufman efficiency ratio
    trade_range: bool = False       # also fade range extremes (adds trades, lower quality)
    range_band_edge: float = 0.12   # range fades only within this far into a band tail
    trade_volatile: bool = False    # sit out VOLATILE/chop (pure fee bleed)
    min_p_win: float = 0.50         # refuse trades below this calibrated win prob
    use_kelly: bool = True          # size by fractional Kelly from P(win)
    kelly_fraction: float = 0.30    # fraction of full Kelly (conservative)
    entry_mode: str = "maker"       # maker (post-only, pays maker fee) | taker
    maker_offset_bps: float = 1.0   # how far inside the touch to rest the limit
    maker_wait_bars: int = 2        # bars to wait for a maker fill before cancelling
    auto_tune: bool = True          # background walk-forward self-tuning
    auto_tune_minutes: int = 90     # how often the auto-tuner re-evaluates


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.008       # target equity fraction at risk if the stop hits
    min_leverage: int = 2               # operating leverage band (auto-adapted within)
    max_leverage: int = 7
    max_risk_hard_pct: float = 0.035    # hard cap on any single trade's loss-at-stop
    margin_mode: str = "ISOLATED"
    # --- adaptive exit geometry (let winners run, cut losers) ---
    sl_atr_min: float = 1.2             # initial stop: no tighter than this x ATR
    sl_atr_max: float = 2.8             # ...and no wider than this (structure-clamped between)
    tp_atr_cap: float = 0.0             # 0 = no fixed target; exit only via trail/edge
    trail_atr_min: float = 1.6          # chandelier trail width in chop
    trail_atr_max: float = 3.8          # ...widened in clean trends (Kaufman ER)
    trail_tighten: float = 0.55         # ratchet trail this much tighter as profit grows
    be_rr: float = 0.8                  # move stop to breakeven at this R
    be_offset_atr: float = 0.15         # breakeven sits entry + this x ATR (covers fees)
    giveback_rr: float = 2.5            # once past this R, protect the open profit
    giveback_frac: float = 0.5          # ...exit if price retraces this fraction of MFE
    hold_edge_frac: float = 0.7         # exit if brain edge flips past this x threshold
    expected_rr: float = 2.2            # assumed winner:loser ratio for Kelly sizing
    time_stop_bars: int = 120           # long backstop; trends need room to develop
    max_open_positions: int = 2
    max_position_notional_pct: float = 0.35  # of equity x leverage, per position
    max_daily_loss_pct: float = 0.05         # kill switch: flatten + halt for the day
    max_consecutive_losses: int = 8          # cooldown trigger
    cooldown_minutes: int = 45
    max_spread_bps: float = 6.0              # refuse entries into a wide spread


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
CONFIG_VERSION = 2

# Top-level settings the user owns — everything else is auto-managed by the
# tuner and reset to code defaults when migrating an older config.
USER_OWNED_TOP = {"symbols", "mode", "feed", "allow_live", "data_dir", "log_level"}


@dataclass
class BotConfig:
    version: int = CONFIG_VERSION
    symbols: list[str] = field(default_factory=lambda: ["BTC-USDT", "ETH-USDT"])
    mode: str = MODE_PAPER              # idle | paper | live
    feed: str = FEED_BINGX              # bingx | synthetic (offline demo)
    allow_live: bool = False            # hard gate: live orders refused unless true
    data_dir: str = "data_cache"
    log_level: str = "INFO"
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
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
                setattr(dc, f.name, [str(x).strip().upper() if f.name == "symbols" else x for x in val if str(x).strip()])
        elif isinstance(cur, str):
            setattr(dc, f.name, str(val))


_NESTED_USER_OWNED = {
    "exchange": {"base_url", "ws_url", "recv_window_ms", "taker_fee", "maker_fee"},
    "server": {"host", "port"},
    "paper": {"starting_balance", "slippage_bps"},
    "strategy": {"interval", "warmup_bars"},
    # leverage band intentionally omitted so migrating an old config resets it
    # to the current 2-7x default; it stays UI-editable afterwards.
    "risk": {"max_open_positions", "max_daily_loss_pct", "margin_mode",
             "max_spread_bps", "max_consecutive_losses", "cooldown_minutes"},
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
