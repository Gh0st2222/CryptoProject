"""FastAPI application: REST API + WebSocket push + static dashboard."""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..util import now_ms
from .orchestrator import Orchestrator

log = logging.getLogger("server")
STATIC = Path(__file__).parent / "static"

orch = Orchestrator()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await orch.startup()
    yield
    await orch.shutdown()


app = FastAPI(title="BingX Adaptive Futures Bot", lifespan=lifespan)


class ModeReq(BaseModel):
    mode: str
    confirm: str = ""


class ControlReq(BaseModel):
    action: str
    symbol: str = ""


class ConfigPatch(BaseModel):
    patch: dict


class BacktestReq(BaseModel):
    symbol: str = "BTC-USDT"
    interval: str = "5m"
    days: float = Field(default=30, ge=0.5, le=365)
    synthetic: bool = False


class OptimizeReq(BacktestReq):
    trials: int = Field(default=40, ge=5, le=300)


class PortfolioReq(BaseModel):
    symbols: list[str] = Field(default_factory=lambda: ["BTC-USDT", "ETH-USDT"])
    interval: str = "5m"
    days: float = Field(default=30, ge=0.5, le=365)
    synthetic: bool = False


class WalkforwardReq(BacktestReq):
    folds: int = Field(default=5, ge=3, le=10)
    trials: int = Field(default=20, ge=5, le=120)


class ApplyParamsReq(BaseModel):
    params: dict
    champion_id: str | None = None   # if applying a vault champion, tag it active


@app.get("/api/status")
async def status():
    return orch.status()


@app.post("/api/mode")
async def set_mode(req: ModeReq):
    ok, msg = await orch.set_mode(req.mode, req.confirm)
    return JSONResponse({"ok": ok, "message": msg}, status_code=200 if ok else 400)


@app.post("/api/control")
async def control(req: ControlReq):
    ok, msg = await orch.control(req.action, req.symbol)
    return JSONResponse({"ok": ok, "message": msg}, status_code=200 if ok else 400)


@app.post("/api/config")
async def patch_config(req: ConfigPatch):
    return orch.update_cfg(req.patch)


@app.get("/api/candles")
async def candles(symbol: str, limit: int = 400, tf: str = "1m"):
    """Chart data. tf='1m' (default) serves the tick-aggregated DISPLAY series —
    the chart always shows 1m regardless of the signal timeframe; tf='signal'
    serves the bars the brain actually trades on."""
    eng = orch.engine
    if eng is None or symbol not in eng.feed.states:
        return {"candles": [], "markers": []}
    st = eng.feed.states[symbol]
    series = st.candles
    if tf == "1m":
        disp = getattr(st, "display", None)
        if disp is not None and (len(disp) or disp.partial is not None):
            series = disp
    out = [
        {"time": c.ts // 1000, "open": c.open, "high": c.high, "low": c.low, "close": c.close}
        for c in series.tail(min(limit, 1200))
    ]
    if series.partial is not None:
        p = series.partial
        out.append({"time": p.ts // 1000, "open": p.open, "high": p.high,
                    "low": p.low, "close": p.close})
    markers = []
    for t in eng.portfolio.trades[-120:]:
        if t.symbol != symbol:
            continue
        markers.append({"ts": t.entry_ts, "kind": "entry", "side": t.side, "price": t.entry_price})
        markers.append({"ts": t.exit_ts, "kind": "exit", "side": t.side,
                        "price": t.exit_price, "pnl": t.pnl})
    for s, p in eng.portfolio.positions.items():
        if s == symbol:
            markers.append({"ts": p.opened_ts, "kind": "entry", "side": p.side, "price": p.entry_price})
    return {"candles": out, "markers": markers}


@app.post("/api/backtest")
async def backtest(req: BacktestReq):
    job = orch.start_backtest(req.symbol.upper(), req.interval, req.days, req.synthetic)
    return {"job_id": job.id}


@app.post("/api/optimize")
async def optimize(req: OptimizeReq):
    job = orch.start_optimizer(req.symbol.upper(), req.interval, req.days, req.trials, req.synthetic)
    return {"job_id": job.id}


@app.post("/api/portfolio_backtest")
async def portfolio_backtest(req: PortfolioReq):
    job = orch.start_portfolio_backtest(req.symbols, req.interval, req.days, req.synthetic)
    return {"job_id": job.id}


@app.post("/api/walkforward")
async def walkforward(req: WalkforwardReq):
    job = orch.start_walkforward(req.symbol.upper(), req.interval, req.days,
                                 req.folds, req.trials, req.synthetic)
    return {"job_id": job.id}


@app.get("/api/journal")
async def journal(mode: str | None = None, limit: int = 300):
    return {"summary": orch.journal.summary(mode), "recent": orch.journal.recent(limit)}


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    job = orch.jobs.get(job_id)
    if job is None:
        return JSONResponse({"error": "no such job"}, status_code=404)
    return job.to_dict()


class CarryLabReq(BaseModel):
    days: float = 60.0
    top_n: int = 6


@app.post("/api/carrylab")
async def carrylab(req: CarryLabReq):
    job = orch.start_carry_lab(req.days, req.top_n)
    return {"job_id": job.id}


@app.get("/api/record")
async def record():
    eng = orch.engine
    pf = eng.portfolio if eng else None
    return orch.record.snapshot(pf, pf.mode if pf else "paper")


@app.post("/api/paper_reset")
async def paper_reset():
    return orch.reset_paper()


@app.get("/api/report")
async def report():
    """One-click diagnostic resume: a plain-text dump of the whole system state
    (no secrets), downloadable for sharing/analysis."""
    from fastapi.responses import PlainTextResponse
    from .report import build_report
    import time as _t
    fname = f"pulse_resume_{_t.strftime('%Y%m%d_%H%M', _t.gmtime())}.txt"
    return PlainTextResponse(build_report(orch), headers={
        "Content-Disposition": f'attachment; filename="{fname}"'})


@app.post("/api/apply_params")
async def apply_params(req: ApplyParamsReq):
    orch.apply_params(req.params)
    if req.champion_id:
        orch.mark_champion_used(req.champion_id)
    return {"ok": True, "config": orch.status()["config"]}


# event -> which channel it drives. "hot" is the small high-cadence payload
# (prices, uPnL, stage); everything structural rides the heavier full state.
# NOTE: "heartbeat" (the idle 2s timeout) deliberately rides the HOT channel —
# it used to trigger a full ~100KB status serialize + whole-page re-render
# every 2 seconds around the clock, which cooked both the server CPU and the
# browser's frame rate for zero new information.
FULL_EVENTS = {"state", "bar", "trade", "mode", "job", "autotune", "config", "radar"}
FULL_MIN_GAP = 0.9    # seconds between heavy full-state pushes
HOT_MIN_GAP = 0.25    # seconds between light hot pushes (≈ up to 4/s)


async def _client_reader(ws: WebSocket) -> None:
    """Consume client frames until the peer goes away. Starlette only learns
    about a disconnect inside receive() — a push-only loop that never receives
    keeps writing into the dead transport forever, and asyncio logs
    'socket.send() raised exception.' several times a second for every
    closed/refreshed tab. This task's completion IS the disconnect signal."""
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                return
    except Exception:  # noqa: BLE001 — any transport error means the client is gone
        return


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    q = orch.subscribe()
    reader = asyncio.create_task(_client_reader(ws))
    try:
        await ws.send_text(json.dumps({"type": "state", "data": orch.status()}))
        last_full = last_hot = 0.0
        while not reader.done():
            try:
                kind = await asyncio.wait_for(q.get(), timeout=2.0)
            except asyncio.TimeoutError:
                kind = "heartbeat"
            # coalesce whatever queued up, but remember if a full-state event
            # was among them so a burst of hot ticks can't starve the full push
            want_full = kind in FULL_EVENTS
            while not q.empty():
                try:
                    k2 = q.get_nowait()
                    want_full = want_full or (k2 in FULL_EVENTS)
                except asyncio.QueueEmpty:
                    break
            if reader.done():      # client left while we were coalescing
                break
            now = now_ms() / 1000.0
            if want_full and now - last_full > FULL_MIN_GAP:
                await ws.send_text(json.dumps({"type": "state", "data": orch.status()}))
                last_full = last_hot = now
            elif now - last_hot > HOT_MIN_GAP:
                await ws.send_text(json.dumps({"type": "hot", "data": orch.hot()}))
                last_hot = now
    except Exception:  # noqa: BLE001 — a dying socket must never take the server down
        pass
    finally:
        reader.cancel()
        orch.unsubscribe(q)


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
