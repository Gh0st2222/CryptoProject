# ⚡ PULSE — BingX Adaptive Futures Terminal

A self-adapting, multi-strategy trading machine for **BingX USDT-M perpetual
futures** with leverage, three execution modes, and a dense realtime web
terminal. It runs like a small trading firm: an analyst floor of 18 alpha
signals across five specialist **desks**, a **meta-allocator** (the CIO) that
shifts capital toward whatever is actually working, **calibrated win
probabilities** driving **fractional-Kelly** sizing, a **health governor** that
throttles risk when it's cold, and a **background auto-tuner** that re-tunes
itself so you don't have to touch settings.

- **Live trading** — real orders on BingX, exchange-side stop-loss/take-profit
  on every entry, kill switch, reconciliation against exchange state.
- **Realtime simulation (paper)** — the *real* BingX market feed, fake money;
  orders fill against the live order book with slippage. No API keys needed.
- **Historical simulation (backtest)** — event-driven replay running the
  *identical* brain code, with a walk-forward optimizer and Monte-Carlo
  robustness.

All three modes share one brain, so what you simulate is what trades.

## Quick start

```bash
pip install -r requirements.txt
python run.py
# open http://127.0.0.1:8420
```

Boots into **paper trading on the live BingX market** (public data, no keys).
No internet? In *Settings → feed* pick **Synthetic**, or set
`"feed":"synthetic"` in `config.json`, and `BOT_SYNTH_SPEED=90 python run.py`
to watch it learn in fast-forward.

## The firm (how the brain works)

```
data ─► 18 alphas ─► 5 desks (per-desk Hedge weights)
     ─► meta-allocator (CIO weights desks by live performance)
     ─► regime gate ─► fused directional edge
     ─► probability calibrator ─► P(win)
     ─► adaptive threshold + cost gate + fractional-Kelly size
```

**The analyst floor — 18 alphas across 5 desks.** Each returns a conviction in
[−1, 1]; event alphas sit dormant until their setup appears (the terminal shows
firing ● vs dormant ○, so a resting alpha never looks broken).

| Desk | Alphas |
|---|---|
| **Trend** | momentum · macd · mtf_trend (multi-timeframe alignment) · breakout · roc_accel |
| **MeanRev** | meanrev_bb · rsi_fade · stoch_fade · vwap_revert · vwap_pullback |
| **Micro** | obi (book imbalance) · flow (aggressor tape+CVD) · cvd_trend · spread_pressure |
| **Vol** | squeeze · vol_breakout (TTM release) |
| **Carry** | funding_skew · oi_divergence *(live-only; dormant offline)* |

**Four nested adaptive loops, all online, all visible:**

1. **Within-desk Hedge weights.** Each alpha's call is graded against the
   ATR-normalized realized return `horizon` bars later; weights multiply by
   `exp(η·payoff)` with shrinkage and a floor, so a muted alpha can win its
   weight back when its market returns.
2. **Meta-allocator (the CIO).** Each desk's *net* call is graded too, and the
   allocator moves capital toward desks with the best live risk-adjusted
   performance (multiplicative weights over desks), auto-muting chronic losers.
   This is "evaluate many strategies at once and back the winners" — you can
   watch the allocation shift in real time and over a backtest.
3. **Regime gating.** ADX + EMA-stack + Kaufman efficiency + multi-timeframe
   alignment classify TREND_UP / TREND_DOWN / RANGE / VOLATILE and re-weight the
   desks and reshape exit geometry per regime.
4. **Probability calibration.** Online logistic regression maps the fused edge
   to a calibrated **P(win)** (Brier-scored). It's also auto-correction: if the
   edge stops predicting, P(win) falls below 0.5 and the gate simply refuses.

A cost gate (`β·|edge|·ATR%·√horizon` must clear round-trip fees × `cost_multiple`)
and an order-flow veto sit in front of every entry.

## Sizing, risk & auto-correction

- **Fractional-Kelly sizing** from calibrated P(win) and the trade's reward:risk,
  on top of a volatility-based base risk — a stop-out still loses a bounded,
  known fraction of equity; Kelly only scales conviction, never the stop.
- **Health governor.** Tracks recent expectancy and drawdown and scales risk
  down when cold, back up as it recovers — hands-off auto-correction.
- Exchange-side **STOP_MARKET / TAKE_PROFIT_MARKET** on every live entry, so the
  protective exit survives a bot crash or disconnect.
- Breakeven shift, ATR trailing stop, time stop, opposite-edge exit.
- **Kill switch** at the daily-loss limit; loss-streak cooldown; spread guard;
  max concurrent positions; manual KILL.

## Self-tuning (don't touch settings)

The **background auto-tuner** is the quant-research desk. On a timer it re-runs
the walk-forward search on recent data and **hot-swaps** new parameters into the
live brains *only* if they beat what's currently running on the held-out
validation slice. If nothing clears the bar, it changes nothing. Everything it
does is reported on the Auto-Tuner tab. You can still drive the same search by
hand on the Optimizer tab and apply a winner with one click.

## Data intake ("not a bit escapes it")

Live mode ingests klines (multi-timeframe context built by resampling), the full
trade tape (CVD / aggressor flow), L20 order book (imbalance), best bid/ask, and
polls **funding rate, mark price and open interest** for the carry desk. Offline
the carry alphas stay dormant rather than firing on absent data.

## The terminal

A dense dark trading terminal: live tape ticker, an execution-cycle pipeline
(Scan→Detect→Validate→Size→Fill→Manage→Settle), fused-edge and calibrated-P(win)
gauges, the live desk-allocation leaderboard, the 18-alpha floor with
firing/dormant state and hit rates, regime + health meters, a candlestick chart
with trade markers, the session equity curve, and a Backtest tab with the
desk-allocation-over-time chart plus a 5,000-path Monte-Carlo robustness panel.

## The path to live

1. **Backtest** a symbol/interval/window (data cached in `data_cache/`).
2. **Optimize** (or just let the auto-tuner run) — train on 70%, rank on the
   held-out 30% so overfit sets fall away.
3. **Paper trade** on the real market for days; watch win rate, P(win)
   calibration (Brier), desk allocations and the health governor.
4. **Go live** only when paper convinces you:
   ```bash
   cp .env.example .env    # add BINGX_API_KEY / BINGX_API_SECRET
   ```
   Enable **allow_live** in Settings, switch to Live, type the confirmation
   phrase. *Tip: point `exchange.base_url` at `https://open-api-vst.bingx.com`
   to trade BingX demo (VST) funds first.*

## The fee reality (read this)

Taker fees are ~0.05%/side; a 1-minute scalp with a 0.15% stop pays ~0.8R
round-trip regardless of signal. PULSE is honest about it: the cost gate refuses
trades whose predicted edge can't clear fees, the cost floor stretches targets
until they can, and the default interval is **5m**. The backtester charges full
taker fees + slippage. "HFT" here means tick-level *reaction* (exits, order-flow
features), not sub-second churn — churning at taker fees only enriches the
exchange.

## Architecture

```
bingxbot/
├── config.py               dataclass config, JSON persistence, .env secrets
├── exchange/               signed async REST + gzip WebSocket (market & user)
├── data/
│   ├── candles.py          numpy ring buffers
│   ├── feed.py             live feed + microstructure + funding/OI context, synthetic feed
│   └── history.py          paginated kline cache + regime-switching synthetic generator
├── strategy/
│   ├── indicators.py       vectorized indicators (+ MACD, Stoch, Keltner, Kaufman ER)
│   ├── features.py         FeatureFrame v2 (52 features incl. multi-timeframe)
│   ├── alphas.py           18 alphas tagged by desk & kind
│   ├── regime.py           regime detection + per-desk gating
│   ├── allocator.py        MetaAllocator — the CIO
│   ├── calibration.py      online logistic P(win) calibration
│   └── brain.py            TradingBrain — the whole firm in one object
├── risk/manager.py         Kelly-aware sizing, exits, health governor, kill switch
├── engine/
│   ├── portfolio.py        accounting + stats
│   ├── brokers.py          PaperBroker / LiveBroker (one interface)
│   ├── trader.py           realtime decision loop + execution pipeline
│   ├── backtest.py         event-driven simulator + walk-forward optimizer
│   └── autotuner.py        background self-tuning research desk
└── server/                 FastAPI + WebSocket + the terminal (vendored charts)
```

## Configuration (`config.json`, UI-editable)

| Field | Default | Meaning |
|---|---|---|
| `strategy.interval` | `5m` | decision bar size |
| `strategy.base_threshold` | 0.34 | min fused edge |
| `strategy.cost_multiple` | 1.4 | edge must beat costs × this |
| `strategy.min_p_win` | 0.50 | refuse trades below this calibrated win prob |
| `strategy.use_kelly` / `kelly_fraction` | true / 0.30 | fractional-Kelly sizing |
| `strategy.auto_tune` / `auto_tune_minutes` | true / 90 | background self-tuning |
| `risk.risk_per_trade` | 0.005 | base equity fraction at stop |
| `risk.max_leverage` | 10 | leverage cap |
| `risk.max_daily_loss_pct` | 0.05 | kill-switch level |
| `allow_live` | false | hard gate for real orders |

API keys live **only** in `.env` — never in `config.json`, never in the browser.

## Tests

```bash
python -m pytest bingxbot/tests/ -q      # 29 tests
```

Covering the exact BingX signature scheme, indicator math, within-desk Hedge
learning (a prescient alpha must accumulate weight), the meta-allocator backing
a winning desk, probability calibration beating the Brier baseline, Kelly
monotonicity/caps, cost/P(win) gating, sizing & kill-switch, paper-broker
accounting, and backtest determinism + bounded losses + a full engine boot.

## Disclaimer

Leveraged futures can lose more than you expect, fast. Provided as-is, no
warranty, no promise of profit — backtest numbers (especially on synthetic data)
do not guarantee live results. Start in paper, size small, never trade money you
can't afford to lose. Not financial advice.
