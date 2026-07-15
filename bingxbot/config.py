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
    interval: str = "5m"            # 1m allowed; cost floor keeps geometry honest
    warmup_bars: int = 350          # bars required before the ensemble may trade
    horizon_bars: int = 5           # bars over which alpha calls are graded
    hedge_eta: float = 0.35         # multiplicative-weights learning rate
    weight_floor: float = 0.04      # no alpha is ever fully muted
    base_threshold: float = 0.34    # |ensemble score| needed to consider a trade
    threshold_adapt: bool = True    # auto-tune threshold toward target trade rate
    target_trades_per_hour: float = 1.5
    cost_multiple: float = 1.4      # predicted move must exceed round-trip cost x this
    micro_confirm: bool = True      # order-flow agreement gate at entry (live/paper)


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.005       # equity fraction lost if the stop is hit
    max_leverage: int = 10
    margin_mode: str = "ISOLATED"
    atr_sl_mult: float = 1.7
    atr_tp_mult: float = 1.9
    trail_atr_mult: float = 1.3
    breakeven_rr: float = 0.75          # move stop to entry at this R multiple
    time_stop_bars: int = 40
    cost_floor_mult: float = 3.5        # take-profit distance >= this x round-trip cost
    max_open_positions: int = 2
    max_position_notional_pct: float = 0.35  # of equity x leverage, per position
    max_daily_loss_pct: float = 0.05         # kill switch: flatten + halt for the day
    max_consecutive_losses: int = 6          # cooldown trigger
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


@dataclass
class BotConfig:
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


def load_config(path: Path = CONFIG_PATH) -> BotConfig:
    cfg = BotConfig()
    if path.exists():
        try:
            _merge_into(cfg, json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError):
            pass  # fall back to defaults rather than refuse to boot
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
