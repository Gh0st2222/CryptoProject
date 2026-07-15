"""Plain dataclasses shared by feed, brokers and engines. Kept pydantic-free
because these travel through the hot path thousands of times per minute."""
from __future__ import annotations

from dataclasses import dataclass, field

LONG = "LONG"
SHORT = "SHORT"

BUY = "BUY"
SELL = "SELL"


@dataclass(slots=True)
class Candle:
    ts: int          # open time, ms
    open: float
    high: float
    low: float
    close: float
    volume: float
    closed: bool = True


@dataclass(slots=True)
class Tick:
    ts: int
    price: float
    qty: float
    is_buyer_maker: bool  # True => aggressive seller hit the bid


@dataclass(slots=True)
class BookTop:
    ts: int
    bid: float
    bid_qty: float
    ask: float
    ask_qty: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_bps(self) -> float:
        m = self.mid
        return (self.ask - self.bid) / m * 10_000 if m > 0 else 0.0


@dataclass(slots=True)
class DepthSnapshot:
    ts: int
    bids: list[tuple[float, float]]  # [(price, qty)] best-first
    asks: list[tuple[float, float]]

    def imbalance(self, levels: int = 10) -> float:
        """Order-book imbalance in [-1, 1]; positive = bid-heavy."""
        b = sum(q for _, q in self.bids[:levels])
        a = sum(q for _, q in self.asks[:levels])
        tot = a + b
        return (b - a) / tot if tot > 0 else 0.0


@dataclass(slots=True)
class ContractSpec:
    symbol: str
    qty_precision: int = 4
    price_precision: int = 2
    min_qty: float = 0.0001
    min_notional_usdt: float = 2.0
    max_long_leverage: int = 100
    max_short_leverage: int = 100
    maker_fee: float = 0.0002
    taker_fee: float = 0.0005


@dataclass(slots=True)
class Position:
    symbol: str
    side: str                 # LONG | SHORT
    qty: float                # in coins, always positive
    entry_price: float
    opened_ts: int
    leverage: float = 1.0
    stop_price: float = 0.0
    take_profit: float = 0.0
    entry_fee: float = 0.0
    entry_reason: str = ""
    entry_bar_ts: int = 0     # bar timestamp at entry (time stop)
    breakeven_moved: bool = False
    trail_price: float = 0.0  # high-water (LONG) / low-water (SHORT) for trailing
    exchange_position_id: str = ""

    def direction(self) -> int:
        return 1 if self.side == LONG else -1

    def unrealized(self, price: float) -> float:
        return (price - self.entry_price) * self.qty * self.direction()

    def notional(self, price: float) -> float:
        return abs(self.qty) * price


@dataclass(slots=True)
class TradeRecord:
    symbol: str
    side: str
    qty: float
    entry_price: float
    exit_price: float
    entry_ts: int
    exit_ts: int
    pnl: float               # net of fees
    fees: float
    reason_open: str
    reason_close: str
    r_multiple: float = 0.0  # pnl / planned risk
    mode: str = "paper"

    @property
    def won(self) -> bool:
        return self.pnl > 0


@dataclass(slots=True)
class OrderResult:
    ok: bool
    order_id: str = ""
    filled_price: float = 0.0
    filled_qty: float = 0.0
    fee: float = 0.0
    error: str = ""
    raw: dict = field(default_factory=dict)
