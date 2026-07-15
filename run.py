#!/usr/bin/env python3
"""Entry point: python run.py  ->  dashboard at http://127.0.0.1:8420"""
import uvicorn

from bingxbot.config import load_config
from bingxbot.util import setup_logging

if __name__ == "__main__":
    cfg = load_config()
    setup_logging(cfg.log_level)
    uvicorn.run("bingxbot.server.app:app", host=cfg.server.host, port=cfg.server.port,
                log_level="warning")
