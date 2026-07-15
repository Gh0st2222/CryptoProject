"""FastAPI application: REST API + WebSocket push + static dashboard."""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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


class ApplyParamsReq(BaseModel):
    params: dict


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
async def candles(symbol: str, limit: int = 400):
    eng = orch.engine
    if eng is None or symbol not in eng.feed.states:
        return {"candles": [], "markers": []}
    st = eng.feed.states[symbol]
    out = [
        {"time": c.ts // 1000, "open": c.open, "high": c.high, "low": c.low, "close": c.close}
        for c in st.candles.tail(min(limit, 1200))
    ]
    if st.candles.partial is not None:
        p = st.candles.partial
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


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    job = orch.jobs.get(job_id)
    if job is None:
        return JSONResponse({"error": "no such job"}, status_code=404)
    return job.to_dict()


@app.post("/api/apply_params")
async def apply_params(req: ApplyParamsReq):
    orch.apply_params(req.params)
    return {"ok": True, "config": orch.status()["config"]}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    q = orch.subscribe()
    try:
        await ws.send_text(json.dumps({"type": "state", "data": orch.status()}))
        last_state = 0.0
        while True:
            try:
                kind = await asyncio.wait_for(q.get(), timeout=2.0)
            except asyncio.TimeoutError:
                kind = "heartbeat"
            # coalesce whatever queued up into a single push
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break
            now = now_ms() / 1000.0
            if kind in ("state", "bar", "trade", "mode", "job", "heartbeat") and now - last_state > 0.9:
                await ws.send_text(json.dumps({"type": "state", "data": orch.status()}))
                last_state = now
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        orch.unsubscribe(q)


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
