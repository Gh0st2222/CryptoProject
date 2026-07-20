# ⚡ PULSE — BingX Adaptive Futures Terminal

A self-adapting, multi-strategy trading machine for **BingX USDT-M perpetual
futures** with **volatility-targeted 2–7× leverage**, three execution modes, and
a dense realtime web terminal. It runs like a small trading firm: an analyst
floor of 19 alpha signals across five specialist **desks**, a **meta-allocator**
(the CIO) that shifts capital toward whatever is actually working, **calibrated
win probabilities** driving **fractional-Kelly** sizing, a **health governor**
that throttles risk when it's cold, and a **continuous evolutionary auto-tuner**
that optimizes the entire parameter space around the clock so you never tune a
setting yourself.

## Where the money comes from (the honest map)

Retail latency cannot out-race anyone, so the system is pointed only at edges a
REST/WS bot can actually collect:

1. **Funding carry (the flagship).** When perp funding gets extreme, the
   exchange mechanically pays the unpopular side every 8h — a public,
   predictable payment, no forecasting required. The **market radar** scans the
   whole BingX perp board every few minutes (all-symbol funding + volume in two
   calls, 4h trend probes on the interesting few) and the **carry desk**
   harvests it: receiving side only, never against a strong 4h trend, small
   size, fixed low leverage, always stopped, out when the funding normalizes.
   The radar's eligibility universe is **popular tokens only**: the CoinGecko
   **top-100 by market cap, as-is** — refreshed every few hours and cached to
   disk, falling back to a built-in majors list offline. Long-tail micro-caps
   never reach the board, the carry desk, trend adoption or the tuner's
   research universe, however hard they pump; your own `symbols` and the
   `radar_extra` setting are always admitted on top.
2. **Higher-timeframe trend.** The signal brain runs on **15m bars by default**
   (1m is demoted to *execution*: the reactive intra-bar scanner times entries
   inside the forming bar). On 15m–4h, momentum has decades of evidence and the
   move is large relative to fees; on 1m it isn't — the old tuner kept proving
   that by refusing to trade.
3. **Liquidation-cascade reversion.** A violent, high-volume bar whose extreme
   wick gets reclaimed by the close is forced selling exhausting itself; the
   `capitulation` alpha fades it — the one *fast* edge that doesn't require
   winning a race, because it enters after the race ends.

Honesty guardrails to match: the backtester now **charges funding at every 8h
boundary held**, the tuner optimizes **log-wealth growth with a convex drawdown
penalty** (what compounding actually maximizes), promotion demands a margin that
**grows with how many candidates have been tried** on the same validation window
(multiple-testing bias), and a **liquidation-distance guard** caps leverage so
the stop always fires before isolated-margin liquidation can.

And the machinery to keep it honest over time:

- **Carry Lab** (Radar tab): replays the carry desk's exact rules over *real
  historical funding prints* + 1h prices for the top-volume perps, sweeps the
  threshold grid, and recommends evidence-based `min_apr`/`exit_apr` — the
  desk's thresholds are measured, not assumed.
- **Radar adoption**: strong, liquid 4h trends found by the radar are **adopted
  into the engine at runtime** (own brain, same gates; auto-released when the
  trend dies, never with an open position). The tuner also **rotates its
  research symbol across the top-10 by volume** and validates champions on a
  cross-symbol basket — parameters must earn on the board, not on BTC's quirks.
- **Paper sessions survive restarts**: positions, trades, equity history and
  risk day-state persist to disk and restore on boot ("Reset paper account"
  starts fresh). The **Record tab** appends one row per UTC day — equity,
  PnL, trades — building the provable months-long track record that is this
  project's real asset.
- The chart is **always 1m** (tick-aggregated display series) and
  **auto-follows** the symbol the machine is looking at; the *bar interval*
  setting is the signal timeframe the brain trades on, not a display choice.

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
  against the position with conviction, it exits. Disciplined like the entry
  side: while the 15m/1h backdrop still clearly supports the position, a
  shallow one-bar wobble is treated as a pullback (the stop/trail/give-back own
  that case) — a normal flip must persist two consecutive closes **and** the
  higher-TF backdrop must have decayed; only a severe outright reversal exits
  immediately regardless. These guards are structural constants, not tunables,
  for the same reason the entry veto is — so the tuner can never optimize the
  discipline away.
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

The auto-tuner is an **always-on research desk**, and it's a real optimizer, not
random poking. A **persistent Differential-Evolution population** (it remembers
what worked across cycles *and* survives restarts, saved to disk) proposes trials
over the whole strategy + risk/exit space; each candidate is scored across
several time folds **in parallel on a dedicated pool of research cores** — with
each fold's indicators built **once** and reused for every candidate, which is
what makes a cycle fast. The population's best is then **validated out-of-sample**
on the most recent held-out window (with an overfit penalty for any train→OOS
drop) and **hot-swapped into the live brains only when it clearly beats the
running champion there** — otherwise nothing changes.

The objective is **risk-adjusted profit**, not trade count: total R earned
(net of fees) tempered by drawdown and quality, recency-weighted so recent market
behavior matters more. That means it rewards higher frequency *only* when the
extra trades actually make risk-adjusted money — the honest fix for "trade more"
that can't regress into the fee-bleeding over-trading it used to reward.

**Cores are allocated to the host.** On startup it reads the CPU's logical core
count and splits it: one core for the event loop, a small slice for on-demand
jobs you launch (backtests/optimizer/walk-forward), and the rest as a dedicated
research pool — so the tuner runs several-wide in parallel without ever starving
the UI or a backtest you just started. More cores → more folds per cycle and more
generations per hour, automatically.

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

**Champion vault — a live candidate pool, not a graveyard.** Every promotion is
written to a persistent store (`data_cache/champions.json`). Crucially, the vault
isn't just a history log: **every cycle the tuner re-validates its top sets against
the *current* market** alongside the freshly-evolved DE candidates and the running
champion, and runs whichever wins — so the best available set drives trading no
matter when it was born. Each record keeps **both** its birth evaluation (score
when generated) **and** its current evaluation (re-scored against today), and — once
it's driven live trades — its **real executed track record** (trades + realized PnL,
tagged per champion in the journal). The vault holds the best & most-used **100**;
the **most-used are protected** from pruning (proven, not merely high-scoring) and
never-used sets **age out weekly**. The Auto-Tuner tab shows all of this — birth→now
fitness, live PnL, use count, a 🔥 badge on the top-10 most-used and a **LIVE** badge
on whichever is currently trading — and any champion can be re-applied with one click.
The tuner also **hunts faster right after a promotion** (tight cadence while it's
clearly improving, relaxed when stable).

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

## Continuous evaluation & the timeframe ladder

The brain does **not** wait for a candle to close to look for a trade. Between
closes it re-scores several times a second on the **live-forming bar** plus the
current order book and trade flow, so an entry can fire the moment the setup
appears — no more sitting idle for minutes while opportunities pass. Scoring is
split from learning: the online weights/calibrator still grade exactly once per
**closed** bar (so they can't be corrupted by intra-bar re-scoring), but the
*decision* runs continuously.

It also reads a real **1m / 5m / 15m / 1h timeframe ladder** — each rung with its
own EMA stack, RSI, ADX and slope — and fuses them into a genuine cross-timeframe
alignment the entry gate and trend desk use. Because coarser rungs are exact
aggregations of the base bar, a 1m feed yields the whole ladder with **no extra
data pulled** (well inside rate limits). Rungs are **epoch-anchored and strictly
causal**: every base bar reads only the last higher-TF bucket that had fully
closed by that bar's close, so the backtester and tuner can never peek at a
higher-TF close that hasn't printed yet, and live/backtest see identical numbers
(enforced by a no-lookahead regression test). The terminal shows the ladder live
so you see exactly what the brain sees on each timeframe.

**Hard trend veto.** On top of that ladder sits a non-negotiable rule: when the
higher timeframes (15m/1h) have decided a direction, the bot will **not** take a
trade against them — in any regime. This is the guard that stops the cardinal
sin of a trend system, shorting into an uptrend (or buying a downtrend). It is a
structural gate, not a tuned parameter, so the auto-tuner can never optimize it
away, and range-fading is no longer in the tuner's search space at all.

## Evidence & capital protection

Features don't make an unproven strategy profitable — evidence and discipline do,
so a whole layer exists to expose the real edge and defend the account:

- **Persistent trade journal.** Every closed trade is written to disk with the
  context it was taken under (regime, the 1m/5m/15m/1h ladder, edge, P(win),
  dominant desk, funding, exit reason, hour). It survives restarts.
- **Analytics tab.** Slices that journal — win rate and PnL by regime, by
  higher-TF alignment, by hour, by desk, by exit reason, by side — so the edge is
  something you *see*, not guess.
- **Honest walk-forward.** The Walk-Fwd tab splits real history into sequential
  folds and trades each one **out-of-sample**, with parameters tuned only on the
  data *before* it, equity chained fold to fold. It is deliberately unflattering.
- **True maker entries in live.** Live now rests **post-only limit** orders
  (maker fee ~0.02%) instead of taking (~0.05%); if a fill doesn't come in the
  window it abandons the entry rather than chase — matching what the backtest
  models, so live fees stop silently eating the edge.
- **Funding-aware** entries (a marginal edge into adverse funding is skipped) and
  **exposure control** (correlation haircut + a net directional cap so it can't
  double the same bet across symbols).
- **Divergence monitor + optional alerts.** Live win rate / profit factor are
  compared to the backtest expectation and flagged when they drift; set
  `BOT_ALERT_WEBHOOK` to get a push on kill-switch, daily summary, or divergence.

## Data intake ("not a bit escapes it")

Live mode ingests klines (a true 1m/5m/15m/1h indicator ladder), the full trade
tape (CVD / aggressor flow), L20 order book (imbalance), best bid/ask, and polls
**funding rate, mark price and open interest** for the carry desk — all streaming
continuously into per-symbol state that the brain re-reads on every tick. Offline
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
│   ├── backtest.py         per-symbol simulator + portfolio + optimizer + honest walk-forward
│   ├── journal.py          persistent trade journal (JSONL + decision context) -> analytics
│   ├── search.py           parallel fold-scoring (indicator reuse) + Differential Evolution
│   └── autotuner.py        research desk: DE population + OOS validation on the research pool
└── server/                 FastAPI + split WebSocket + dual process pools + alerts + the terminal
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
| `risk.max_open_positions` | 3 | concurrent positions — always on **different** tokens (one position per symbol is structural: portfolio & brokers refuse a second open on a held token, and the carry desk never touches a token a signal brain is watching) |
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
python -m pytest bingxbot/tests/ -q      # 61 tests
```

Covering the exact BingX signature scheme, indicator math, within-desk Hedge
learning (a prescient alpha must accumulate weight), the meta-allocator backing
a winning desk, probability calibration beating the Brier baseline, Kelly
monotonicity/caps, cost/P(win) gating, sizing & kill-switch, paper-broker
accounting, backtest determinism + bounded losses + a full engine boot, a strict
**no-lookahead regression test** on the multi-timeframe ladder (features at bar
i must be identical with the future removed), the **one-position-per-token**
guards, exactly-once risk settlement, and tunable-bounds clamping.

## Disclaimer

Leveraged futures can lose more than you expect, fast. Provided as-is, no
warranty, no promise of profit — backtest numbers (especially on synthetic data)
do not guarantee live results. Start in paper, size small, never trade money you
can't afford to lose. Not financial advice.
