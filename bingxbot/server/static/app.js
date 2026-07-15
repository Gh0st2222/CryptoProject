/* PULSE dashboard — vanilla JS + lightweight-charts (vendored). */
"use strict";

/* ---------------------------------------------------------------- theme */
const C = {
  page: "#0d0d0d", surface: "#1a1a19", ink: "#ffffff", ink2: "#c3c2b7",
  muted: "#898781", grid: "#2c2c2a", baseline: "#383835",
  up: "#0ca30c", dn: "#d03b3b", accent: "#3987e5",
};
const ALPHA_COLORS = {
  momentum: "#3987e5", meanrev_bb: "#008300", breakout: "#d55181",
  vwap_pullback: "#c98500", rsi_fade: "#199e70", squeeze: "#d95926",
  obi: "#9085e9", flow: "#e66767",
};
const ALPHA_ORDER = Object.keys(ALPHA_COLORS);
const REGIME_META = {
  TREND_UP: { cls: "trend-up", glyph: "▲", label: "Trend up" },
  TREND_DOWN: { cls: "trend-down", glyph: "▼", label: "Trend down" },
  RANGE: { cls: "range", glyph: "◆", label: "Range" },
  VOLATILE: { cls: "volatile", glyph: "⚡", label: "Volatile" },
};

/* ---------------------------------------------------------------- utils */
const $ = (id) => document.getElementById(id);
const fmt = {
  usd: (v, d = 2) => (v == null || isNaN(v)) ? "—" :
    (v < 0 ? "−$" : "$") + Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }),
  signed: (v, d = 2) => (v == null || isNaN(v)) ? "—" : (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d),
  px: (v) => {
    if (v == null || isNaN(v) || v === 0) return "—";
    const d = v >= 1000 ? 1 : v >= 50 ? 2 : v >= 1 ? 4 : 6;
    return v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  },
  pct: (v, d = 1) => (v == null || isNaN(v)) ? "—" : (v * 100).toFixed(d) + "%",
  time: (ms) => ms ? new Date(ms).toLocaleTimeString("en-GB", { hour12: false }) : "—",
  dt: (ms) => ms ? new Date(ms).toLocaleString("en-GB", { hour12: false, day: "2-digit", month: "short" }) : "—",
  dur: (s) => s <= 0 ? "—" : s < 90 ? `${s}s` : s < 5400 ? `${Math.round(s / 60)}m` : `${(s / 3600).toFixed(1)}h`,
};
function toast(msg, kind = "") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  $("toasts").appendChild(el);
  setTimeout(() => { el.style.opacity = "0"; el.style.transition = "opacity .4s"; }, 4200);
  setTimeout(() => el.remove(), 4700);
}
async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.message || data.error || `HTTP ${res.status}`);
  return data;
}
const pnlCls = (v) => v > 0 ? "pnl-pos" : v < 0 ? "pnl-neg" : "";
const sideCls = (s) => s === "LONG" ? "side-long" : "side-short";
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/* ---------------------------------------------------------------- charts */
const baseChartOpts = (h) => ({
  height: h,
  layout: { background: { color: "transparent" }, textColor: C.muted, fontFamily: "system-ui, -apple-system, sans-serif", fontSize: 11 },
  grid: { vertLines: { color: C.grid }, horzLines: { color: C.grid } },
  rightPriceScale: { borderColor: C.baseline },
  timeScale: { borderColor: C.baseline, timeVisible: true, secondsVisible: false },
  crosshair: { mode: 0, vertLine: { color: C.muted, width: 1, style: 2 }, horzLine: { color: C.muted, width: 1, style: 2 } },
});

let mainChart, candleSeries, equityChart, equitySeries;
let btEquityChart, btEquitySeries, btWeightsChart, btWeightSeries = {};

function initCharts() {
  mainChart = LightweightCharts.createChart($("chart-main"), baseChartOpts(430));
  candleSeries = mainChart.addCandlestickSeries({
    upColor: C.up, downColor: C.dn, borderUpColor: C.up, borderDownColor: C.dn,
    wickUpColor: C.up, wickDownColor: C.dn,
  });
  equityChart = LightweightCharts.createChart($("chart-equity"), {
    ...baseChartOpts(130),
    rightPriceScale: { borderColor: C.baseline, scaleMargins: { top: 0.15, bottom: 0.1 } },
    timeScale: { visible: false }, handleScroll: false, handleScale: false,
  });
  equitySeries = equityChart.addAreaSeries({
    lineColor: C.accent, lineWidth: 2,
    topColor: "rgba(57,135,229,0.25)", bottomColor: "rgba(57,135,229,0.02)",
    priceLineVisible: false, lastValueVisible: true,
  });
  new ResizeObserver(() => {
    mainChart.applyOptions({ width: $("chart-main").clientWidth });
    equityChart.applyOptions({ width: $("chart-equity").clientWidth });
  }).observe($("chart-main"));
}

function tradeMarkers(markers) {
  return markers
    .slice().sort((a, b) => a.ts - b.ts)
    .map((m) => m.kind === "entry" ? {
      time: Math.floor(m.ts / 1000), position: m.side === "LONG" ? "belowBar" : "aboveBar",
      color: m.side === "LONG" ? C.up : C.dn, shape: m.side === "LONG" ? "arrowUp" : "arrowDown",
      text: m.side === "LONG" ? "L" : "S",
    } : {
      time: Math.floor(m.ts / 1000), position: "inBar",
      color: (m.pnl ?? 0) >= 0 ? C.up : C.dn, shape: "circle",
      text: m.pnl != null ? fmt.signed(m.pnl, Math.abs(m.pnl) >= 10 ? 0 : 1) : "x",
    });
}

/* ---------------------------------------------------------------- state */
let S = null;                 // latest /api/status payload
let curSymbol = null;
let lastTradeCount = -1;
let candleTimer = 0;

function symbols() { return S?.config?.symbols ?? []; }
function engineSym() { return S?.engine?.symbols?.[curSymbol]; }

async function refreshCandles(full = false) {
  if (!curSymbol || !S?.engine) return;
  try {
    const limit = full ? 500 : 3;
    const d = await api(`/api/candles?symbol=${encodeURIComponent(curSymbol)}&limit=${limit}`);
    if (!d.candles.length) return;
    if (full) {
      candleSeries.setData(d.candles);
      candleSeries.setMarkers(tradeMarkers(d.markers));
      mainChart.timeScale().scrollToRealTime();
    } else {
      for (const c of d.candles) candleSeries.update(c);
      if (S.engine.portfolio.stats.trades !== lastTradeCount) {
        candleSeries.setMarkers(tradeMarkers(d.markers.length ? d.markers : []));
      }
    }
  } catch (e) { /* transient */ }
}

function setSymbol(sym, force = false) {
  if (!sym || (sym === curSymbol && !force)) return;
  curSymbol = sym;
  document.querySelectorAll(".sym-tab").forEach((b) => b.classList.toggle("active", b.dataset.sym === sym));
  $("brain-sym").textContent = sym;
  refreshCandles(true);
}

/* ---------------------------------------------------------------- render */
function renderTop() {
  const mode = S.mode;
  const pill = $("mode-pill");
  pill.className = `pill ${mode}`;
  $("mode-text").textContent = mode.toUpperCase();
  if ($("mode-select").value !== mode) $("mode-select").value = mode;

  const healthy = !!S.engine?.feed_healthy;
  $("feed-dot").className = `feed-dot ${healthy ? "ok" : ""}`;
  $("feed-label").textContent = S.engine ? (S.config.feed === "synthetic" ? "synthetic feed" : "BingX feed") : "no feed";

  const pf = S.engine?.portfolio;
  $("top-equity").textContent = pf ? fmt.usd(pf.equity) : "—";
  const day = S.engine?.risk?.day_realized ?? null;
  $("top-daypnl").textContent = day == null ? "—" : fmt.signed(day, 2);
  $("top-daypnl").className = "v num " + pnlCls(day ?? 0);
  const st = pf?.stats;
  $("top-wr").textContent = st && st.trades ? fmt.pct(st.win_rate) : "—";
  $("top-trades").textContent = st ? String(st.trades) : "—";
}

function renderSymTabs() {
  const wrap = $("sym-tabs");
  const syms = symbols();
  if ([...wrap.children].map((b) => b.dataset.sym).join(",") !== syms.join(",")) {
    wrap.innerHTML = "";
    for (const s of syms) {
      const b = document.createElement("button");
      b.className = "sym-tab"; b.dataset.sym = s; b.textContent = s.replace("-USDT", "");
      b.onclick = () => setSymbol(s);
      wrap.appendChild(b);
    }
    if (!syms.includes(curSymbol)) setSymbol(syms[0], true);
    else setSymbol(curSymbol, true);
  }
}

function renderBrain() {
  const es = engineSym();
  if (!es) return;
  const ens = es.ensemble, micro = es.micro;

  $("px-last").textContent = fmt.px(es.price);
  $("px-meta").textContent = `spread ${micro.spread_bps.toFixed(1)}bp · OBI ${fmt.signed(micro.obi, 2)} · flow ${fmt.signed(micro.flow, 2)}`;

  const reg = REGIME_META[ens.regime] ?? null;
  const badge = $("regime-badge");
  if (reg) {
    badge.className = `badge ${reg.cls}`;
    badge.innerHTML = `<span class="glyph">${reg.glyph}</span><span>${reg.label}</span>`;
  }
  $("conf-badge").textContent = `conf ${fmt.pct(ens.regime_conf, 0)}`;
  $("thr-badge").textContent = `thr ${ens.threshold.toFixed(2)}`;

  const score = ens.score ?? 0, thr = ens.threshold ?? 0.3;
  const needle = $("score-needle");
  needle.style.left = `calc(${50 + Math.max(-1, Math.min(1, score)) * 49}% - 2px)`;
  needle.style.background = Math.abs(score) < thr ? C.ink2 : (score > 0 ? "#7db4ee" : "#f0a0a0");
  $("thr-pos").style.left = `${50 + thr * 49}%`;
  $("thr-neg").style.left = `${50 - thr * 49}%`;
  $("score-read").textContent = `score ${fmt.signed(score, 2)}`;
  const held = S.engine.portfolio.open_positions[curSymbol];
  $("gate-read").textContent = held ? `in position (${es.bars_held} bars)` :
    Math.abs(score) >= thr ? "signal armed" : "below threshold";

  const list = $("alpha-list");
  if (!list.children.length) {
    for (const name of ALPHA_ORDER) {
      const row = document.createElement("div");
      row.className = "alpha-row";
      row.innerHTML = `
        <div class="alpha-name"><span class="swatch" style="background:${ALPHA_COLORS[name]}"></span>${name}</div>
        <div class="alpha-track"><div class="alpha-fill" id="af-${name}" style="background:${ALPHA_COLORS[name]}"></div></div>
        <div class="alpha-meta"><span class="alpha-score" id="as-${name}">0.00</span><br><span id="ah-${name}">—</span></div>`;
      list.appendChild(row);
    }
  }
  const maxW = Math.max(...Object.values(ens.weights), 0.001);
  for (const name of ALPHA_ORDER) {
    const w = ens.weights[name] ?? 0, sc = ens.scores[name] ?? 0;
    $(`af-${name}`).style.width = `${(w / maxW) * 100}%`;
    $(`af-${name}`).style.opacity = 0.45 + 0.55 * (w / maxW);
    const scEl = $(`as-${name}`);
    scEl.textContent = fmt.signed(sc, 2);
    scEl.className = "alpha-score " + (sc > 0.05 ? "score-pos" : sc < -0.05 ? "score-neg" : "score-zero");
    const st = ens.alpha_stats[name];
    $(`ah-${name}`).textContent = st && st.calls > 5 ? `${(st.hit_rate * 100).toFixed(0)}% · ${st.calls}` : "—";
  }

  $("kv-beta").textContent = ens.beta.toFixed(2);
  $("kv-graded").textContent = ens.graded;
  $("kv-bars").textContent = `${es.bars}${es.bars < es.warmup_bars ? ` / ${es.warmup_bars} warmup` : ""}`;
  const risk = S.engine.risk;
  $("kv-risk").textContent = risk.killed ? `KILLED: ${risk.kill_reason}` : "normal";
  $("kv-risk").style.color = risk.killed ? "#f08d8d" : "";
  $("kv-cooldown").textContent = risk.cooldown_s > 0 ? fmt.dur(risk.cooldown_s) : "—";
  $("kv-block").textContent = es.entry_block || "—";
  $("kv-block").title = es.entry_block || "";
}

function renderEquity() {
  const curve = S.engine?.equity_curve ?? [];
  if (curve.length > 1) {
    equitySeries.setData(curve.map(([ts, eq]) => ({ time: Math.floor(ts / 1000), value: eq })));
    const eq = curve[curve.length - 1][1], start = S.engine.portfolio.starting_balance;
    const d = eq - start;
    $("eq-caption").textContent = `${fmt.usd(eq)}  (${fmt.signed(d, 2)} / ${fmt.signed(d / start * 100, 2)}%)`;
    $("eq-caption").className = "val num " + pnlCls(d);
  }
}

function renderPositions() {
  const pf = S.engine?.portfolio;
  const body = $("pos-body");
  const entries = pf ? Object.entries(pf.open_positions) : [];
  if (!entries.length) {
    body.innerHTML = `<tr><td colspan="11" class="empty">No open positions</td></tr>`;
    return;
  }
  body.innerHTML = entries.map(([sym, p]) => {
    const mark = S.engine.symbols[sym]?.price ?? 0;
    return `<tr>
      <td>${esc(sym)}</td>
      <td class="${sideCls(p.side)}">${p.side}</td>
      <td class="r num">${p.qty}</td>
      <td class="r num">${fmt.px(p.entry)}</td>
      <td class="r num">${fmt.px(mark)}</td>
      <td class="r num">${fmt.px(p.stop)}</td>
      <td class="r num">${fmt.px(p.tp)}</td>
      <td class="r num ${pnlCls(p.upnl)}">${fmt.signed(p.upnl, 2)}</td>
      <td class="r num">${p.leverage}x</td>
      <td>${fmt.time(p.opened_ts)}</td>
      <td><button class="btn small" onclick="closePos('${esc(sym)}')">Close</button></td>
    </tr>`;
  }).join("");
}

function renderTrades() {
  const trades = (S.engine?.trades ?? []).slice().reverse();
  const st = S.engine?.portfolio?.stats;
  $("trade-cards").innerHTML = !st ? "" : [
    ["Win rate", st.trades ? fmt.pct(st.win_rate) : "—"],
    ["Profit factor", st.trades ? st.profit_factor.toFixed(2) : "—"],
    ["Trades", st.trades],
    ["Net PnL", fmt.signed(st.total_pnl, 2), pnlCls(st.total_pnl)],
    ["Avg R", st.trades ? fmt.signed(st.avg_r, 2) : "—"],
    ["Max DD", fmt.pct(st.max_drawdown)],
    ["Fees", fmt.usd(st.fees_paid)],
  ].map(([k, v, cls]) => `<div class="card"><div class="k">${k}</div><div class="v num ${cls ?? ""}">${v}</div></div>`).join("");

  const body = $("trades-body");
  if (!trades.length) {
    body.innerHTML = `<tr><td colspan="10" class="empty">No closed trades yet</td></tr>`;
    return;
  }
  body.innerHTML = trades.map((t) => `<tr>
    <td>${fmt.time(t.exit_ts)}</td>
    <td>${esc(t.symbol)}</td>
    <td class="${sideCls(t.side)}">${t.side}</td>
    <td class="r num">${t.qty}</td>
    <td class="r num">${fmt.px(t.entry_price)}</td>
    <td class="r num">${fmt.px(t.exit_price)}</td>
    <td class="r num ${pnlCls(t.pnl)}">${fmt.signed(t.pnl, 2)}</td>
    <td class="r num ${pnlCls(t.r_multiple)}">${fmt.signed(t.r_multiple, 2)}</td>
    <td class="mono-note">${esc(t.reason_open)}</td>
    <td class="mono-note">${esc(t.reason_close)}</td>
  </tr>`).join("");
}

function renderSettingsIfClean() {
  if (settingsDirty || !S) return;
  const c = S.config;
  $("cfg-symbols").value = c.symbols.join(", ");
  $("cfg-feed").value = c.feed;
  $("cfg-interval").value = c.strategy.interval;
  $("cfg-threshold").value = c.strategy.base_threshold;
  $("cfg-costmult").value = c.strategy.cost_multiple;
  $("cfg-tph").value = c.strategy.target_trades_per_hour;
  $("cfg-adapt").checked = c.strategy.threshold_adapt;
  $("cfg-micro").checked = c.strategy.micro_confirm;
  $("cfg-risk").value = c.risk.risk_per_trade;
  $("cfg-lev").value = c.risk.max_leverage;
  $("cfg-dayloss").value = c.risk.max_daily_loss_pct;
  $("cfg-maxpos").value = c.risk.max_open_positions;
  $("cfg-balance").value = c.paper.starting_balance;
  $("cfg-allowlive").checked = c.allow_live;
  $("cfg-keys").textContent = c.has_keys ? "configured ✓" : "not set (paper/backtest only)";
  $("cfg-keys").style.color = c.has_keys ? "#37c837" : "";
}

function renderAll() {
  if (!S) return;
  renderTop();
  renderSymTabs();
  if (S.engine) {
    renderBrain();
    renderEquity();
    renderPositions();
    renderTrades();
    const tc = S.engine.portfolio.stats.trades;
    refreshCandles(false).then(() => { lastTradeCount = tc; });
  }
  renderSettingsIfClean();
}

/* ---------------------------------------------------------------- ws */
let ws, wsRetry = 1;
function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "state") { S = msg.data; renderAll(); }
  };
  ws.onopen = () => { wsRetry = 1; };
  ws.onclose = () => setTimeout(connectWS, Math.min(wsRetry *= 1.6, 8) * 1000);
  ws.onerror = () => ws.close();
}

/* ---------------------------------------------------------------- actions */
window.closePos = async (sym) => {
  try { await api("/api/control", { action: "close", symbol: sym }); toast(`${sym} closed`, "good"); }
  catch (e) { toast(e.message, "bad"); }
};

$("btn-kill").onclick = async () => {
  if (!confirm("Kill switch: flatten all positions and halt entries?")) return;
  try { await api("/api/control", { action: "kill" }); toast("Kill switch engaged", "warn"); }
  catch (e) { toast(e.message, "bad"); }
};
$("btn-flatten").onclick = async () => {
  try { const r = await api("/api/control", { action: "flatten" }); toast(r.message, "good"); }
  catch (e) { toast(e.message, "bad"); }
};
$("btn-reset-kill").onclick = async () => {
  try { const r = await api("/api/control", { action: "reset_kill" }); toast(r.message, "good"); }
  catch (e) { toast(e.message, "bad"); }
};

$("mode-select").onchange = async (ev) => {
  const mode = ev.target.value;
  if (mode === "live") { openLiveModal(); return; }
  try { const r = await api("/api/mode", { mode }); toast(r.message, "good"); }
  catch (e) { toast(e.message, "bad"); ev.target.value = S?.mode ?? "idle"; }
};

/* live modal */
function openLiveModal() {
  $("live-phrase").textContent = S?.live_confirm_phrase ?? "TRADE LIVE";
  $("live-confirm-input").value = "";
  $("live-go").disabled = true;
  $("live-modal").classList.add("open");
}
$("live-confirm-input").oninput = (ev) => {
  $("live-go").disabled = ev.target.value !== (S?.live_confirm_phrase ?? "TRADE LIVE");
};
$("live-cancel").onclick = () => {
  $("live-modal").classList.remove("open");
  $("mode-select").value = S?.mode ?? "idle";
};
$("live-go").onclick = async () => {
  try {
    const r = await api("/api/mode", { mode: "live", confirm: $("live-confirm-input").value });
    toast(r.message, r.ok === false ? "bad" : "warn");
    $("live-modal").classList.remove("open");
  } catch (e) {
    toast(e.message, "bad");
  }
};

/* tabs */
document.querySelectorAll(".tab").forEach((b) => {
  b.onclick = () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
    document.querySelectorAll(".tab-page").forEach((p) =>
      p.classList.toggle("active", p.dataset.page === b.dataset.tab));
    if (b.dataset.tab === "backtest" || b.dataset.tab === "optimizer") ensureBtCharts();
  };
});

/* ---------------------------------------------------------------- settings */
let settingsDirty = false;
document.querySelectorAll('[data-page="settings"] input, [data-page="settings"] select')
  .forEach((el) => el.addEventListener("input", () => { settingsDirty = true; }));

$("cfg-save").onclick = async () => {
  const patch = {
    symbols: $("cfg-symbols").value.split(",").map((s) => s.trim().toUpperCase()).filter(Boolean),
    feed: $("cfg-feed").value,
    allow_live: $("cfg-allowlive").checked,
    strategy: {
      interval: $("cfg-interval").value,
      base_threshold: parseFloat($("cfg-threshold").value),
      cost_multiple: parseFloat($("cfg-costmult").value),
      target_trades_per_hour: parseFloat($("cfg-tph").value),
      threshold_adapt: $("cfg-adapt").checked,
      micro_confirm: $("cfg-micro").checked,
    },
    risk: {
      risk_per_trade: parseFloat($("cfg-risk").value),
      max_leverage: parseInt($("cfg-lev").value, 10),
      max_daily_loss_pct: parseFloat($("cfg-dayloss").value),
      max_open_positions: parseInt($("cfg-maxpos").value, 10),
    },
    paper: { starting_balance: parseFloat($("cfg-balance").value) },
  };
  try {
    const r = await api("/api/config", { patch });
    settingsDirty = false;
    toast(r.needs_restart ? "Saved — switch to Idle and back to apply feed/symbol changes" : "Settings saved", "good");
  } catch (e) { toast(e.message, "bad"); }
};

/* ---------------------------------------------------------------- jobs */
async function pollJob(jobId, progressEl, onDone) {
  progressEl.style.display = "block";
  const bar = progressEl.querySelector(".bar");
  const tick = async () => {
    try {
      const j = await api(`/api/jobs/${jobId}`);
      bar.style.width = `${(j.progress * 100).toFixed(1)}%`;
      if (j.done) {
        progressEl.style.display = "none";
        if (j.error) toast(`Job failed: ${j.error}`, "bad");
        else onDone(j.result);
        return;
      }
    } catch (e) { /* keep polling */ }
    setTimeout(tick, 700);
  };
  tick();
}

/* backtest charts (lazy) */
function ensureBtCharts() {
  if (btEquityChart) return;
  btEquityChart = LightweightCharts.createChart($("chart-bt-equity"), baseChartOpts(240));
  btEquitySeries = btEquityChart.addAreaSeries({
    lineColor: C.accent, lineWidth: 2,
    topColor: "rgba(57,135,229,0.25)", bottomColor: "rgba(57,135,229,0.02)",
    priceLineVisible: false,
  });
  btWeightsChart = LightweightCharts.createChart($("chart-bt-weights"), {
    ...baseChartOpts(240),
    rightPriceScale: { borderColor: "#383835", scaleMargins: { top: 0.1, bottom: 0.1 } },
  });
  for (const name of ALPHA_ORDER) {
    btWeightSeries[name] = btWeightsChart.addLineSeries({
      color: ALPHA_COLORS[name], lineWidth: 2, priceLineVisible: false, lastValueVisible: false,
      title: name,
    });
  }
  new ResizeObserver(() => {
    btEquityChart.applyOptions({ width: $("chart-bt-equity").clientWidth });
    btWeightsChart.applyOptions({ width: $("chart-bt-weights").clientWidth });
  }).observe($("chart-bt-equity"));
}

function statCards(st, start) {
  return [
    ["Win rate", st.trades ? fmt.pct(st.win_rate) : "—", st.win_rate >= 0.55 ? "pnl-pos" : ""],
    ["Profit factor", st.trades ? st.profit_factor.toFixed(2) : "—", st.profit_factor >= 1 ? "pnl-pos" : "pnl-neg"],
    ["Trades", st.trades],
    ["Net PnL", fmt.signed(st.total_pnl, 2), pnlCls(st.total_pnl)],
    ["Return", fmt.signed(st.total_pnl / start * 100, 2) + "%", pnlCls(st.total_pnl)],
    ["Max drawdown", fmt.pct(st.max_drawdown)],
    ["Avg R", st.trades ? fmt.signed(st.avg_r, 2) : "—", pnlCls(st.avg_r)],
    ["Expectancy", fmt.signed(st.expectancy, 2)],
    ["Fees paid", fmt.usd(st.fees_paid)],
  ].map(([k, v, cls]) => `<div class="card"><div class="k">${k}</div><div class="v num ${cls ?? ""}">${v}</div></div>`).join("");
}

$("bt-run").onclick = async () => {
  try {
    const r = await api("/api/backtest", {
      symbol: $("bt-symbol").value.trim().toUpperCase(),
      interval: $("bt-interval").value,
      days: parseFloat($("bt-days").value),
      synthetic: $("bt-synth").checked,
    });
    $("bt-results").style.display = "none";
    pollJob(r.job_id, $("bt-progress"), renderBacktest);
  } catch (e) { toast(e.message, "bad"); }
};

function renderBacktest(res) {
  ensureBtCharts();
  $("bt-results").style.display = "block";
  if (res.error) { toast(res.error, "bad"); return; }
  $("bt-cards").innerHTML = statCards(res.stats, res.starting_balance);
  // charts must size against a *visible* container - defer one frame
  requestAnimationFrame(() => {
    btEquityChart.applyOptions({ width: $("chart-bt-equity").clientWidth });
    btWeightsChart.applyOptions({ width: $("chart-bt-weights").clientWidth });
    btEquitySeries.setData(res.equity_curve.map(([ts, eq]) => ({ time: Math.floor(ts / 1000), value: eq })));
    btEquityChart.timeScale().fitContent();
    for (const name of ALPHA_ORDER) {
      btWeightSeries[name].setData((res.weights_timeline ?? [])
        .map((w) => ({ time: Math.floor(w.ts / 1000), value: w[name] ?? 0 })));
    }
    btWeightsChart.timeScale().fitContent();
  });
  const trades = (res.trades ?? []).slice(-200).reverse();
  $("bt-trades-body").innerHTML = trades.length ? trades.map((t) => `<tr>
    <td>${fmt.dt(t.exit_ts)}</td>
    <td class="${sideCls(t.side)}">${t.side}</td>
    <td class="r num">${fmt.px(t.entry_price)}</td>
    <td class="r num">${fmt.px(t.exit_price)}</td>
    <td class="r num ${pnlCls(t.pnl)}">${fmt.signed(t.pnl, 2)}</td>
    <td class="r num">${fmt.signed(t.r_multiple, 2)}</td>
    <td class="mono-note">${esc(t.reason_close)}</td>
  </tr>`).join("") : `<tr><td colspan="7" class="empty">No trades in this window</td></tr>`;
  const s = res.stats;
  toast(`Backtest done: ${s.trades} trades, WR ${fmt.pct(s.win_rate)}, PF ${s.profit_factor.toFixed(2)}`,
    s.total_pnl >= 0 ? "good" : "warn");
}

$("op-run").onclick = async () => {
  try {
    const r = await api("/api/optimize", {
      symbol: $("op-symbol").value.trim().toUpperCase(),
      interval: $("op-interval").value,
      days: parseFloat($("op-days").value),
      trials: parseInt($("op-trials").value, 10),
      synthetic: $("op-synth").checked,
    });
    $("op-results").style.display = "none";
    pollJob(r.job_id, $("op-progress"), renderOptimizer);
  } catch (e) { toast(e.message, "bad"); }
};

let opFinalists = [];
function renderOptimizer(res) {
  $("op-results").style.display = "block";
  if (res.error) { toast(res.error, "bad"); return; }
  opFinalists = res.finalists ?? [];
  $("op-body").innerHTML = opFinalists.length ? opFinalists.map((f, i) => {
    const v = f.valid ?? {}, params = Object.entries(f.params)
      .map(([k, val]) => `${k}=${val}`).join("  ");
    return `<tr>
      <td>${i + 1}</td>
      <td class="r num ${f.valid_fitness > 0 ? "pnl-pos" : "pnl-neg"}">${f.valid_fitness}</td>
      <td class="r num">${v.win_rate != null ? fmt.pct(v.win_rate) : "—"}</td>
      <td class="r num">${v.profit_factor != null ? v.profit_factor.toFixed(2) : "—"}</td>
      <td class="r num">${v.trades ?? "—"}</td>
      <td class="r num">${f.train_fitness}</td>
      <td class="mono-note" style="max-width:420px">${esc(params)}</td>
      <td><button class="btn small primary" onclick="applyParams(${i})">Apply</button></td>
    </tr>`;
  }).join("") : `<tr><td colspan="8" class="empty">No viable finalists — try more days or trials</td></tr>`;
  toast(`Optimizer done: ${opFinalists.length} finalists`, "good");
}

window.applyParams = async (i) => {
  const f = opFinalists[i];
  if (!f) return;
  try {
    await api("/api/apply_params", { params: f.params });
    toast("Parameters applied to running config", "good");
  } catch (e) { toast(e.message, "bad"); }
};

/* ---------------------------------------------------------------- boot */
initCharts();
connectWS();
setInterval(() => { if (S?.engine) refreshCandles(false); }, 5000);
