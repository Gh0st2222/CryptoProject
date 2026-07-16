# ⚡ PULSE — BingX Adaptive Futures Terminal

A self-adapting, multi-strategy trading machine for **BingX USDT-M perpetual
futures** with **volatility-targeted 2–7× leverage**, three execution modes, and
a dense realtime web terminal. It runs like a small trading firm: an analyst
floor of 18 alpha signals across five specialist **desks**, a **meta-allocator**
(the CIO) that shifts capital toward whatever is actually working, **calibrated
win probabilities** driving **fractional-Kelly** sizing, a **health governor**
that throttles risk when it's cold, and a **continuous evolutionary auto-tuner**
that optimizes the entire parameter space around the clock so you never tune a
setting yourself.

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

## Adaptive exits (the algorithm decides when the move is over)

There is no fixed take-profit. Winners are ridden and losers are cut by an
adaptive engine so a handful of big trends pay for many small losses — the only
payoff shape that survives fees:

- **Structure-based initial stop** at the recent swing (Donchian), clamped
  between `sl_atr_min` and `sl_atr_max` × ATR — meaningful, bounded, = 1R.
- **Profit-scaled chandelier trail** that starts wide (lets a young trend
  breathe) and ratchets tighter as the trade gains, so a runner banks its move
  instead of round-tripping to breakeven. Width also widens with trend quality
  (Kaufman efficiency ratio) and regime.
- **Edge-flip exit** — every bar the brain re-scores; if the fused edge turns
  against the position with conviction, it exits *now*. This is the algorithm
  deciding the move is done rather than waiting for a static stop.
- Breakeven once +`be_rr` R; give-back lock protects a large open profit; a long
  time-stop backstops dead trades.

## Leverage, sizing & auto-correction

- **Volatility-targeted 2–7× leverage.** Risk-based sizing is expressed as a
  *leverage* and clamped to your band `[min_leverage, max_leverage]` (default
  2–7). Because the stop is ATR-based, leverage rises in calm markets and falls
  in volatile ones — classic vol targeting — while per-trade risk stays roughly
  constant. Conviction (Kelly) and the health governor move where in the band it
  sits; a **hard per-trade risk cap** overrides the band in a storm. Live mode
  sets the chosen leverage on the exchange per trade.
- **Fractional-Kelly sizing** from calibrated P(win) and the trade's reward:risk.
- **Health governor.** Tracks recent expectancy and drawdown and scales risk
  down when cold, back up as it recovers — hands-off auto-correction.
- Exchange-side **STOP_MARKET** protection on every live entry, so the stop
  survives a bot crash or disconnect; the trail is re-synced each bar.
- **Kill switch** at the daily-loss limit; loss-streak cooldown; spread guard;
  max concurrent positions; manual KILL.

## Continuous self-tuning (you never touch settings)

The auto-tuner is an **always-on research desk**. Every couple of minutes it
evolves the current champion parameters (gaussian perturbations) plus fresh
random explorers across the **entire** strategy + risk/exit space (~22
parameters), and **hot-swaps a challenger into the live brains only when it
clearly beats the running champion** — otherwise it changes nothing. Promotions
persist and show up on the Auto-Tuner tab with their fitness jump and the params
adopted.

Candidates are scored with **robust multi-window fitness**: every parameter set
is run across several disjoint time windows and ranked on *median minus
variance* (plus a worst-window penalty). A config that only prints in one lucky
window scores poorly, so the tuner prefers parameters that hold up across
different market conditions — the main defense against overfitting, which is the
number-one reason retail bots that "backtest great" die in production.

**Exit style is regime-conditional.** Trend setups ride the chandelier trail
(let winners run). Range setups (opt-in, `trade_range`) are *scalped*: enter
maker, exit at a **passive post-only limit target** that captures the spread and
the maker rebate, with a tight stop and short time-stop. Range mean-reversion is
a weak edge, so it ships **off** by default — but the maker-in/maker-out scalp
machinery is there for markets where it works.

It never touches the settings **you** own: symbols, feed, interval, starting
balance, max open positions, the leverage band, and the daily-loss limit. On the
Settings tab those are the only editable fields; everything else is displayed
read-only as the live values the tuner has chosen.

**Champion vault (auto-save / auto-prune).** Every promotion is written to a
persistent store (`data_cache/champions.json`), ranked by validation fitness, and
**pruned to the best 12** — the worst/old sets are deleted automatically. The
Auto-Tuner tab shows the vault, and any champion can be re-applied to the live
brains with one click. The tuner also **hunts faster right after a promotion**
(tight cadence while it's clearly improving, relaxed cadence when stable).

**It runs on other cores.** The heavy scoring — the continuous tuner and every
Backtest / Optimizer / Portfolio job — runs in a **process pool** (spawn workers,
cross-platform), so it never holds the GIL and never stalls the event loop. The
UI stays responsive even while the research desk is grinding through candidates.

## Multi-symbol portfolio backtest (one shared account)

The **Portfolio** tab backtests several symbols on a **single shared account** —
one equity pool, one position cap, one daily-loss kill switch, one health
governor. Symbols are aligned on their common bars and stepped in lockstep, and a
**correlation haircut** shrinks a same-direction add while another symbol is
already carrying that bet (BTC/ETH move together — don't stack the same trade).
Diversification smooths the equity curve, so the account can safely carry size no
single symbol could. Results include the shared-account curve, a per-symbol
contribution breakdown, and the realized average cross-symbol correlation. Same
engine and accounting as the single-symbol backtest — it's literally several
per-symbol simulators sharing one `Portfolio` and `RiskManager`.

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
with trade markers, the session equity curve, a Backtest tab with the
desk-allocation-over-time chart plus a 5,000-path Monte-Carlo robustness panel, a
Portfolio tab (multi-symbol shared-account backtest), and an Auto-Tuner tab with
the promotion log and the champion vault.

The dashboard pushes over a **split WebSocket**: a light **hot** channel streams
prices, open-position uPnL and the execution stage several times a second (so the
numbers feel live between bar closes), while the heavier full snapshot — brain
internals, equity curve, trade history — rides a slower channel. The two never
contend for the same cycle, which is what killed the earlier lag/latency spikes.

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

## The fee reality & what to honestly expect

Taker fees are ~0.05%/side; on 1-minute bars the round-trip cost is bigger than
the average bar's move, so **any** signal bleeds to death — this is why naive
scalpers lose. PULSE attacks that on three fronts:

1. **Maker entries** — resting post-only limit orders pay the maker fee
   (~0.02%) instead of taker (~0.05%), roughly halving round-trip cost. (Default
   `entry_mode: "maker"`.)
2. **Discipline gate** — it only trades where an edge can exist: confirmed,
   efficient, multi-timeframe-aligned trends. It sits out choppy/volatile
   regimes entirely (that's where accounts quietly bleed). Trading *less* is the
   single biggest improvement.
3. **Let winners run** — asymmetric adaptive exits so a few multi-R trends pay
   for the many small losers a trend system takes.

**Be realistic.** No bot is reliably profitable in all conditions — that does
not exist, and anyone claiming it is selling something. This is a disciplined
trend-follower: it makes money in trending markets and takes small, controlled
losses in chop. On *realistic* backtests (a hardened near-random-walk generator,
full fees + slippage) the shipped defaults are net positive across most random
seeds with ~2% drawdown — but a choppy month will still be red, and **real-market
results will differ from any backtest.** Prove it in paper on live BingX data
for weeks before risking a cent. The backtester charges full fees and slippage
and resolves stops pessimistically — the numbers aren't flattered.

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
│   ├── exits.py            adaptive exit engine (structure stop + chandelier + edge)
│   └── brain.py            TradingBrain — the whole firm in one object
├── risk/manager.py         Kelly-aware sizing, health governor, kill switch
├── engine/
│   ├── portfolio.py        accounting + stats
│   ├── brokers.py          PaperBroker / LiveBroker (one interface)
│   ├── trader.py           realtime decision loop + execution pipeline
│   ├── backtest.py         event-driven per-symbol simulator + portfolio (shared account) + optimizer
│   └── autotuner.py        background self-tuning research desk (runs in the process pool)
└── server/                 FastAPI + split WebSocket + process pool + the terminal (vendored charts)
```

## Configuration

The **only** settings you own (editable on the Settings tab); everything else is
owned and continuously optimized by the auto-tuner and shown read-only:

| Field | Default | Meaning |
|---|---|---|
| `symbols` | BTC-USDT, ETH-USDT | traded contracts |
| `feed` | `bingx` | `bingx` (real market) or `synthetic` (offline) |
| `strategy.interval` | `5m` | decision bar size (15m/5m recommended) |
| `paper.starting_balance` | 10000 | simulation bankroll |
| `risk.max_open_positions` | 2 | concurrent positions |
| `risk.min_leverage` / `max_leverage` | 2 / 7 | leverage band (auto-adapted within) |
| `risk.max_risk_hard_pct` | 0.035 | hard cap on any single trade's loss |
| `risk.max_daily_loss_pct` | 0.05 | kill-switch level |
| `strategy.auto_tune` | true | run the continuous tuner |
| `allow_live` | false | hard gate for real orders |

Auto-managed (the tuner owns them): `base_threshold`, `cost_multiple`,
`target_trades_per_hour`, `min_p_win`, `kelly_fraction`, `min_efficiency`,
`hedge_eta`, `horizon_bars`, `risk_per_trade`, and the full exit geometry
(`sl_atr_*`, `trail_atr_*`, `trail_tighten`, `be_rr`, `giveback_*`,
`hold_edge_frac`, `time_stop_bars`, …). API keys live **only** in `.env`.

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
