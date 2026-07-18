"""Funding-carry desk: harvest extreme perp funding as a first-class strategy.

The one edge in this codebase that does not require out-predicting anyone:
when funding is stretched, the exchange mechanically pays the unpopular side
every 8 hours, and the print is public. The desk takes the RECEIVING side of
extreme funding on liquid perps — but only when the 4h trend doesn't oppose
it (carry against a freight train gets steamrolled), small, low-leverage,
always stopped, and it leaves when the funding normalizes.

Positions live in the SHARED portfolio (equity, risk caps and the UI see
them), but management runs on its own slow loop with REST marks — a carry
trade is held for hours, it does not need tick handling. Symbols outside the
engine's feed get a lightweight pseudo-state so the paper broker can fill
them; live orders go to the exchange natively.
"""
from __future__ import annotations

import asyncio
import logging
import time

from ..exchange.models import LONG, SHORT, ContractSpec
from ..util import clamp, now_ms, round_step
from .scanner import annualize_funding

log = logging.getLogger("carry")

LOOP_S = 90                 # management cadence (stops / funding / exits)
FUNDING_MS = 8 * 3600 * 1000
ENTRY_WINDOW_MS = 3 * 3600 * 1000   # enter only within 3h of the next settlement:
                                    # collect the first print soon, carry less
                                    # price risk per print collected


# ------------------------------------------------------- pure decision logic

def receiving_side(funding_rate: float) -> str:
    """Longs pay shorts when funding is positive; shorts pay longs when negative."""
    return SHORT if funding_rate > 0 else LONG


def carry_entry_ok(apr: float, er_4h: float, dir_4h: int, cfg) -> tuple[bool, str]:
    """Enter only when the payment is worth it AND the 4h trend doesn't oppose
    the receiving side. cfg is CarryConfig."""
    if abs(apr) < cfg.min_apr:
        return False, f"apr {apr*100:.0f}% < {cfg.min_apr*100:.0f}%"
    side = receiving_side(apr)
    side_d = 1 if side == LONG else -1
    if dir_4h * side_d < 0 and er_4h >= cfg.trend_veto_er:
        return False, f"4h trend (ER {er_4h:.2f}) opposes {side}"
    return True, "ok"


def pick_carry_entry(rows: list[dict], held: set, cfg, now: int) -> tuple[dict | None, str]:
    """Choose the carry entry for this pass, if any: among qualifying harvestable
    rows, the one whose settlement lands SOONEST — and only inside the entry
    window (being long price-risk for 7 hours to collect one print is a worse
    trade than waiting). Pure and testable; cfg is CarryConfig."""
    qualifying, reason = [], ""
    for row in rows:
        sym = row.get("symbol", "")
        if row.get("kind") != "carry" or sym in held:
            continue
        ok, why = carry_entry_ok(row.get("funding_apr", 0.0), row.get("er_4h", 0.0),
                                 row.get("dir_4h", 0), cfg)
        reason = f"{sym}: {why}"
        if ok:
            qualifying.append(row)
    if not qualifying:
        return None, reason or "no harvestable funding on the board"
    qualifying.sort(key=lambda r: r.get("next_funding_time", 0) or (now + FUNDING_MS))
    soon = qualifying[0]
    wait_ms = (soon.get("next_funding_time", 0) or (now + FUNDING_MS)) - now
    if wait_ms > ENTRY_WINDOW_MS:
        return None, f"{soon['symbol']}: waiting funding window ({wait_ms/3_600_000:.1f}h out)"
    return soon, f"{soon['symbol']}: ok"


def carry_exit_reason(side: str, apr: float, er_4h: float, dir_4h: int,
                      held_hours: float, cfg) -> str | None:
    """Why an open carry position should close now, if at all. cfg is CarryConfig."""
    if held_hours >= cfg.max_hold_hours:
        return "carry max hold"
    if abs(apr) < cfg.exit_apr:
        return "funding normalized"
    side_d = 1 if side == LONG else -1
    # the payment flipped to the other side: we'd now be the one paying
    if apr != 0 and receiving_side(apr) != side:
        return "funding flipped"
    if dir_4h * side_d < 0 and er_4h >= cfg.trend_veto_er:
        return "4h trend turned"
    return None


# ----------------------------------------------------- REST-fed pseudo-state

class _StubCandles:
    def __init__(self):
        self.last_close = 0.0


class RestMarketState:
    """The minimal surface PaperBroker + portfolio marks need, fed by REST
    marks instead of a websocket — for carry symbols outside the engine feed."""

    def __init__(self, price: float = 0.0):
        self.book = None
        self.last_price = price
        self.candles = _StubCandles()
        self.candles.last_close = price

    def set_price(self, px: float) -> None:
        if px > 0:
            self.last_price = px
            self.candles.last_close = px

    def mark_price(self) -> float:
        return self.last_price


class CarryDesk:
    def __init__(self, orch):
        self.orch = orch
        self._task: asyncio.Task | None = None
        self.meta: dict[str, dict] = {}     # symbol -> {rate, apr, next_ft, opened_ts, stop, er, dir}
        self.funding_collected = 0.0
        self.entries = 0
        self.exits = 0
        self.last_reason = ""

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(), name="carry-desk")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    def positions(self) -> dict:
        eng = self.orch.engine
        if eng is None:
            return {}
        return {s: p for s, p in eng.portfolio.positions.items() if s in self.meta}

    # ----------------------------------------------------------------- loop

    def _adopt_orphans(self) -> None:
        """After a paper-state restore, positions on symbols outside the engine's
        brain contexts (i.e. old carry trades) have no manager — claim them so
        their stops and funding keep being handled."""
        eng = self.orch.engine
        if eng is None:
            return
        for sym, pos in eng.portfolio.positions.items():
            if sym not in eng.ctx and sym not in self.meta:
                self.meta[sym] = {"rate": 0.0, "apr": 0.0,
                                  "next_ft": now_ms() + FUNDING_MS,
                                  "opened_ts": pos.opened_ts, "er": 0.0, "dir": 0}
                log.info("carry re-adopted restored position %s", sym)

    async def _loop(self) -> None:
        await asyncio.sleep(20)
        self._adopt_orphans()
        while True:
            try:
                if self.orch.cfg.carry.enabled and self.orch.engine is not None:
                    await self._manage_open()
                    await self._maybe_enter()
            except Exception as e:  # noqa: BLE001 — the desk must never kill the app
                log.warning("carry loop failed: %s", e)
            await asyncio.sleep(LOOP_S)

    # ---------------------------------------------------------- entry logic

    async def _maybe_enter(self) -> None:
        orch, cfg = self.orch, self.orch.cfg
        eng, radar = orch.engine, getattr(orch, "scanner", None)
        if eng is None or radar is None or not radar.rows:
            return
        pf, risk = eng.portfolio, eng.risk
        if len(self.positions()) >= cfg.carry.max_positions:
            return
        marks = {s: st.mark_price() for s, st in eng.feed.states.items()}
        equity = pf.equity(marks)
        ok, why = risk.can_enter(equity, len(pf.positions), 1.0)
        if not ok:
            self.last_reason = f"risk: {why}"
            return
        row, self.last_reason = pick_carry_entry(radar.rows, set(pf.positions), cfg.carry, now_ms())
        if row is not None:
            await self._open(row, equity)   # one entry per loop — carry scales slowly by design

    async def _open(self, row: dict, equity: float) -> None:
        orch, cfg = self.orch, self.orch.cfg
        eng = orch.engine
        sym, price = row["symbol"], row.get("mark", 0.0)
        atr_pct = max(row.get("atr_pct_4h", 0.0), 0.002)
        if price <= 0:
            return
        side = receiving_side(row.get("funding_rate", 0.0))
        stop_dist = cfg.carry.stop_atr_4h * atr_pct * price
        spec = orch.specs.get(sym, ContractSpec(sym))
        # small, capped size: a fraction of normal per-trade risk at fixed low leverage
        risk_amt = equity * cfg.risk.risk_per_trade * cfg.carry.risk_frac
        qty = risk_amt / stop_dist
        qty = min(qty, equity * cfg.carry.leverage * 0.9 / price)
        qty = round_step(qty, spec.qty_precision)
        if qty < spec.min_qty or qty * price < spec.min_notional_usdt:
            self.last_reason = f"{sym}: size below exchange minimum"
            return
        # make the symbol fillable/markable outside the engine feed
        if sym not in eng.feed.states:
            eng.feed.states[sym] = RestMarketState(price)
        elif isinstance(eng.feed.states.get(sym), RestMarketState):
            eng.feed.states[sym].set_price(price)

        from ..risk.manager import SizedOrder
        stop = price - stop_dist if side == LONG else price + stop_dist
        sized = SizedOrder(qty=qty, notional=qty * price, leverage=cfg.carry.leverage,
                           stop_price=stop, take_profit=0.0, risk_amount=qty * stop_dist)
        apr = row.get("funding_apr", 0.0)
        reason = f"carry {apr*100:+.0f}% APR"
        res = await eng.broker.open_position(sym, side, sized, reason, bar_ts=now_ms())
        if res.ok:
            self.entries += 1
            self.meta[sym] = {
                "rate": row.get("funding_rate", 0.0), "apr": apr,
                "next_ft": row.get("next_funding_time", 0) or (now_ms() + FUNDING_MS),
                "opened_ts": now_ms(), "er": row.get("er_4h", 0.0), "dir": row.get("dir_4h", 0),
            }
            log.info("CARRY OPEN %s %s qty=%.6g @ %.6g (%s)", side, sym, qty, res.filled_price, reason)
            if orch._notify:
                await orch._notify("trade")

    # ------------------------------------------------------- manage / exits

    async def _manage_open(self) -> None:
        orch = self.orch
        eng, cfg = orch.engine, orch.cfg
        for sym, pos in list(self.positions().items()):
            m = self.meta.get(sym, {})
            # refresh mark + funding (REST; live data when available, radar-stale otherwise)
            rate, mark = m.get("rate", 0.0), 0.0
            if orch.rest is not None:
                try:
                    prem = await orch.rest.premium_index(sym)
                    rate, mark = prem.get("funding_rate", rate), prem.get("mark", 0.0)
                    m["next_ft"] = prem.get("next_funding_time", m.get("next_ft", 0)) or m.get("next_ft", 0)
                except Exception as e:  # noqa: BLE001
                    log.debug("carry premium %s: %s", sym, e)
            st = eng.feed.states.get(sym)
            if mark <= 0:
                mark = st.mark_price() if st is not None else pos.entry_price
            if isinstance(st, RestMarketState):
                st.set_price(mark)
            m["rate"] = rate
            m["apr"] = annualize_funding(rate)

            # paper funding accrual at each 8h boundary (live: the exchange settles it)
            if eng.portfolio.mode != "live" and m.get("next_ft", 0) and now_ms() >= m["next_ft"]:
                notional = pos.qty * mark
                recv = receiving_side(rate) == pos.side
                transfer = notional * abs(rate) * (-1.0 if recv else 1.0)   # negative = credit
                eng.portfolio.charge_funding(transfer)
                self.funding_collected -= transfer
                m["next_ft"] += FUNDING_MS
                log.info("CARRY funding %s: %s %.4f", sym, "received" if recv else "paid", -transfer)

            # protective stop first, then strategy exits
            d = pos.direction()
            if pos.stop_price > 0 and (mark - pos.stop_price) * d <= 0:
                await self._close(sym, "carry stop")
                continue
            held_h = (now_ms() - m.get("opened_ts", now_ms())) / 3_600_000
            why = carry_exit_reason(pos.side, m["apr"], m.get("er", 0.0), m.get("dir", 0),
                                    held_h, cfg.carry)
            if why:
                await self._close(sym, why)

    async def _close(self, sym: str, reason: str) -> None:
        eng = self.orch.engine
        res = await eng.broker.close_position(sym, reason)
        if res.ok:
            self.exits += 1
            m = self.meta.pop(sym, {})
            t = eng.portfolio.trades[-1] if eng.portfolio.trades else None
            if t is not None and self.orch.journal is not None:
                try:
                    self.orch.journal.record({
                        "ts": t.exit_ts, "symbol": sym, "side": t.side, "qty": t.qty,
                        "entry": t.entry_price, "exit": t.exit_price, "pnl": round(t.pnl, 6),
                        "r": t.r_multiple, "fees": round(t.fees, 6), "mode": t.mode,
                        "reason_open": t.reason_open, "reason_close": reason,
                        "desk": "carry", "regime": "CARRY", "hour": -1,
                        "apr_at_entry": m.get("apr", 0.0),
                        "champion_id": None,
                    })
                except Exception as e:  # noqa: BLE001
                    log.warning("carry journal failed: %s", e)
            log.info("CARRY CLOSE %s (%s)", sym, reason)
            if self.orch._notify:
                await self.orch._notify("trade")

    # ------------------------------------------------------------- snapshot

    def snapshot(self) -> dict:
        eng = self.orch.engine
        marks = {s: st.mark_price() for s, st in eng.feed.states.items()} if eng else {}
        pos = []
        for sym, p in self.positions().items():
            m = self.meta.get(sym, {})
            pos.append({
                "symbol": sym, "side": p.side, "qty": p.qty, "entry": p.entry_price,
                "mark": marks.get(sym, 0.0), "stop": p.stop_price,
                "apr": round(m.get("apr", 0.0), 4),
                "next_funding_ts": m.get("next_ft", 0),
                "upnl": round(p.unrealized(marks.get(sym, 0.0)), 4) if marks.get(sym) else 0.0,
                "held_h": round((now_ms() - m.get("opened_ts", now_ms())) / 3_600_000, 1),
            })
        return {
            "enabled": self.orch.cfg.carry.enabled,
            "positions": pos,
            "funding_collected": round(self.funding_collected, 4),
            "entries": self.entries, "exits": self.exits,
            "last_reason": self.last_reason,
        }
