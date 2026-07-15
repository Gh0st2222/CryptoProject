# ⚡ PULSE — BingX Adaptive Futures Bot

A high-frequency-style, self-adapting trading bot for **BingX USDT-M perpetual
futures** with leverage, three execution modes, and a realtime web dashboard.

- **Live trading** — real orders on BingX, exchange-side stop-loss/take-profit
  attached to every entry, kill switch, reconciliation against exchange state.
- **Realtime simulation (paper)** — the *real* BingX market feed, fake money.
  Orders fill against the live order book with slippage. No API keys needed.
- **Historical simulation (backtest)** — event-driven replay over downloaded
  BingX klines running the *identical* strategy code, plus a train/validate
  **optimizer** that tunes parameters and lets you apply the winner in one click.

All three modes share one brain, so what you simulate is what trades.

## Quick start

```bash
pip install -r requirements.txt
python run.py
# open http://127.0.0.1:8420
```

That boots straight into **paper trading on the live BingX market** (public
data, no keys). The dashboard shows the candle chart with trade markers, the
equity curve, and the adaptive brain: current regime, ensemble score vs.
threshold, and each alpha's live weight, signal, and hit rate.

**No internet? Instant demo:** in *Settings → Data feed* pick **Synthetic**
(or set `"feed": "synthetic"` in `config.json`). A regime-switching simulated
market drives the whole stack offline. `BOT_SYNTH_SPEED=60 python run.py`
compresses an hour into a minute so you can watch the bot learn in fast-forward.

## The recommended path to live

1. **Backtest** — dashboard → *Backtest*: pick symbol/interval/days. Data is
   downloaded from BingX once and cached in `data_cache/`.
2. **Optimize** — *Optimizer* tab: random-search over thresholds and exit
   geometry, trained on the first 70% and **ranked on the held-out 30%** so
   overfit parameter sets fall away. Apply the best finalist with one click.
3. **Paper trade** — let it run on the real market for days. Watch win rate,
   profit factor, and the entry-gate reasons in the brain panel.
4. **Go live** — only when paper convinces you:
   ```bash
   cp .env.example .env     # add BINGX_API_KEY / BINGX_API_SECRET
   ```
   Enable **allow_live** in *Settings*, switch mode to *Live*, and type the
   confirmation phrase. The bot sets isolated margin + leverage, attaches
   exchange-side SL/TP to every entry, and reconciles local state against the
   exchange every 30 s.

   *Tip: BingX has a demo-trading environment (VST balance). Point
   `exchange.base_url` in `config.json` at `https://open-api-vst.bingx.com`
   to run "live" mode against demo funds first.*

## How the brain works

**8 alpha signals**, each returning a conviction score in [−1, +1]:

| Alpha | Idea | Data |
|---|---|---|
| `momentum` | EMA-stack drift, ATR-normalized, ROC-confirmed | bars |
| `meanrev_bb` | fade Bollinger extremes with RSI(3) kicker | bars |
| `breakout` | Donchian channel breaks with volume confirmation | bars |
| `vwap_pullback` | in a drift, buy pullbacks toward VWAP | bars |
| `rsi_fade` | RSI(14) exhaustion fade | bars |
| `squeeze` | volatility-squeeze expansion with direction | bars |
| `obi` | order-book imbalance (top 10 levels) | live book |
| `flow` | aggressor trade-flow imbalance + CVD slope | live tape |

Three adaptive layers combine them:

1. **Online learning (Hedge / multiplicative weights).** Every alpha's call is
   graded against the ATR-normalized return `horizon_bars` later; weights
   multiply by `exp(η · payoff)` and renormalize, with mild shrinkage toward
   uniform and a hard floor so a muted alpha can win its weight back when its
   market returns. The dashboard shows weights and per-alpha hit rates live.
2. **Regime gating.** ADX + EMA-stack + ATR-percentile classify TREND_UP /
   TREND_DOWN / RANGE / VOLATILE; each regime re-weights the alpha families
   (momentum leads in trends, fading leads in ranges, everything shrinks in
   chaos) and reshapes stop/target geometry.
3. **Cost-aware adaptive threshold.** The entry threshold tracks the recent
   |score| distribution to hit a target trade rate, and an entry must also
   clear a *calibrated* expected-move test: `β · |score| · ATR% · √horizon`
   must exceed round-trip costs × `cost_multiple`, where β is continuously
   fitted from realized moves. Signals that can't pay for their own fees are
   refused — the block reason is shown in the UI.

A final order-flow veto blocks entries that lean hard against live tape.

## Risk management

- **Volatility-based sizing**: quantity is set so a stop-out loses exactly
  `risk_per_trade` of equity; leverage is derived (and capped), never the driver.
- **Cost-floored geometry**: take-profit distance is floored at
  `cost_floor_mult ×` round-trip cost, stretching the whole bracket when the
  timeframe is too quiet to pay its own fees.
- Exchange-side **STOP_MARKET / TAKE_PROFIT_MARKET** on every live entry — the
  protective exit survives bot crashes and disconnects.
- Breakeven shift, ATR trailing stop, time stop, opposite-signal exit.
- **Kill switch**: daily loss limit flattens everything and halts; loss-streak
  cooldown; spread guard; max concurrent positions; manual KILL button.

## The fee reality (read this)

Taker fees on BingX perpetuals are ~0.05% per side. A 1-minute scalp with a
0.15% stop puts round-trip costs near **0.8R** — unbeatable no matter the
signal. This bot is honest about that: the cost gate refuses trades whose
predicted edge can't clear fees, the cost floor stretches targets until they
can, and the default interval is **5m**. You can trade 1m; the gates will
simply be picky. The backtester charges full taker fees + slippage so the
numbers you see include the tax. "HFT" here means tick-level *reaction* (exits
and order-flow features), not sub-second churn — churning at taker fees only
makes your broker rich.

## Architecture

```
bingxbot/
├── config.py               dataclass config, JSON persistence, .env secrets
├── exchange/
│   ├── rest.py             signed async REST (official signature scheme)
│   ├── ws.py               market + user WebSocket (gzip, Ping/Pong, auto-resub)
│   └── models.py           Candle/Tick/Book/Position/Trade dataclasses
├── data/
│   ├── candles.py          numpy ring buffers
│   ├── feed.py             live feed + microstructure state, synthetic feed
│   └── history.py          paginated kline download + cache, synthetic generator
├── strategy/
│   ├── indicators.py       vectorized numpy indicators
│   ├── features.py         FeatureFrame (same numbers live & backtest)
│   ├── alphas.py           the 8 signals
│   ├── regime.py           regime detection + gating tables
│   └── ensemble.py         Hedge weights, calibration, adaptive threshold
├── risk/manager.py         sizing, exits, kill switch
├── engine/
│   ├── portfolio.py        accounting + stats
│   ├── brokers.py          PaperBroker / LiveBroker (one interface)
│   ├── trader.py           realtime decision loop
│   └── backtest.py         event-driven simulator + optimizer
└── server/
    ├── app.py              FastAPI REST + WebSocket push
    ├── orchestrator.py     mode lifecycle + background jobs
    └── static/             the dashboard (vendored lightweight-charts)
```

Decisions happen on bar close; exits are managed tick-by-tick. The backtester
executes entries at the *next* bar open, resolves intrabar stop/target hits
pessimistically (stop first), and charges fees + slippage on every fill.

## Configuration

`config.json` (created on first save; UI-editable) — key fields:

| Field | Default | Meaning |
|---|---|---|
| `symbols` | BTC-USDT, ETH-USDT | traded contracts |
| `feed` | `bingx` | `bingx` or `synthetic` |
| `strategy.interval` | `5m` | decision bar size |
| `strategy.base_threshold` | 0.34 | min ensemble score |
| `strategy.cost_multiple` | 1.4 | edge must exceed costs × this |
| `risk.risk_per_trade` | 0.005 | equity fraction at stop |
| `risk.max_leverage` | 10 | leverage cap |
| `risk.max_daily_loss_pct` | 0.05 | kill-switch level |
| `risk.cost_floor_mult` | 3.5 | TP ≥ this × round-trip cost |
| `paper.starting_balance` | 10000 | simulation bankroll |
| `allow_live` | false | hard gate for real orders |

API keys go **only** in `.env` (`BINGX_API_KEY`, `BINGX_API_SECRET`) — never in
`config.json`, never in the browser.

## Tests

```bash
python -m pytest bingxbot/tests/ -q      # 27 tests
```

Covering: the exact BingX signature scheme, indicator math, Hedge-weight
learning (a prescient alpha must accumulate weight), cost gating, sizing and
kill-switch logic, paper-broker accounting, backtest determinism and bounded
losses, and a full engine boot on the synthetic feed.

## Disclaimer

Leveraged futures trading can lose more than you expect, quickly. This
software is provided as-is with no warranty and no promise of profit —
backtest results (especially on synthetic data) do not guarantee live
performance. Start in paper mode, size small, and never trade money you
cannot afford to lose. Not financial advice.
