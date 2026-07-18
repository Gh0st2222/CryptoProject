/* PULSE terminal — vanilla JS + vendored lightweight-charts. */
"use strict";

/* neon-noir palette — candles are a diverging polarity pair (mint/magenta,
   deutan ΔE 11.9 + direction as secondary encoding); desks are the validated
   categorical set (dark band, adjacent ΔE ≥ 9). */
const C = { up:"#00e0a0", dn:"#ff3d7f", accent:"#00d2ff", muted:"#59637a", grid:"#10141f", baseline:"#1d2436", ink2:"#a6b3c2" };
const DESK_COLORS = { trend:"#009ec2", meanrev:"#9d6bff", micro:"#e8266d", vol:"#bd8610", carry:"#00a874" };
const DESK_ORDER = ["trend","meanrev","micro","vol","carry"];
const DESK_LABEL = { trend:"TREND", meanrev:"MEANREV", micro:"MICRO", vol:"VOL", carry:"CARRY" };
const REGIME_META = {
  TREND_UP:{cls:"trend-up",g:"▲",label:"Trend up"}, TREND_DOWN:{cls:"trend-down",g:"▼",label:"Trend down"},
  RANGE:{cls:"range",g:"◆",label:"Range"}, VOLATILE:{cls:"volatile",g:"⚡",label:"Volatile"} };

const $ = (id)=>document.getElementById(id);
const clamp=(x,a,b)=>x<a?a:x>b?b:x;
const fmt = {
  usd:(v,d=2)=>(v==null||isNaN(v))?"—":(v<0?"−$":"$")+Math.abs(v).toLocaleString("en-US",{minimumFractionDigits:d,maximumFractionDigits:d}),
  signed:(v,d=2)=>(v==null||isNaN(v))?"—":(v>=0?"+":"−")+Math.abs(v).toFixed(d),
  px:(v)=>{ if(v==null||isNaN(v)||v===0) return "—"; const d=v>=1000?1:v>=50?2:v>=1?4:6; return v.toLocaleString("en-US",{minimumFractionDigits:d,maximumFractionDigits:d}); },
  pct:(v,d=1)=>(v==null||isNaN(v))?"—":(v*100).toFixed(d)+"%",
  time:(ms)=>ms?new Date(ms).toLocaleTimeString("en-GB",{hour12:false}):"—",
  dt:(ms)=>ms?new Date(ms).toLocaleString("en-GB",{hour12:false,day:"2-digit",month:"short"}):"—",
  dur:(s)=>s<=0?"—":s<90?`${s}s`:s<5400?`${Math.round(s/60)}m`:`${(s/3600).toFixed(1)}h`,
};
function toast(msg,kind=""){ const el=document.createElement("div"); el.className=`toast ${kind}`; el.textContent=msg;
  $("toasts").appendChild(el); setTimeout(()=>{el.style.opacity="0";el.style.transition="opacity .4s";},4200); setTimeout(()=>el.remove(),4700); }
async function api(path,body){ const res=await fetch(path,body===undefined?{}:{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  const d=await res.json().catch(()=>({})); if(!res.ok) throw new Error(d.message||d.error||`HTTP ${res.status}`); return d; }
const pnlCls=(v)=>v>0?"pnl-pos":v<0?"pnl-neg":"";
const sideCls=(s)=>s==="LONG"?"side-long":"side-short";
const esc=(s)=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

/* ---------------------------------------------------------------- charts */
const baseOpts=(h)=>({ height:h, layout:{background:{color:"transparent"},textColor:C.muted,fontFamily:"ui-monospace,Menlo,monospace",fontSize:10},
  grid:{vertLines:{color:C.grid},horzLines:{color:C.grid}}, rightPriceScale:{borderColor:C.baseline},
  timeScale:{borderColor:C.baseline,timeVisible:true,secondsVisible:false},
  crosshair:{mode:0,vertLine:{color:C.muted,width:1,style:2},horzLine:{color:C.muted,width:1,style:2}} });
let mainChart,candleSeries,equityChart,equitySeries;
let btEquityChart,btEquitySeries,btAllocChart,btAllocSeries={};
let pfEquityChart,pfEquitySeries;
let wfEquityChart,wfEquitySeries;
function initCharts(){
  mainChart=LightweightCharts.createChart($("chart-main"),baseOpts(384));
  candleSeries=mainChart.addCandlestickSeries({upColor:C.up,downColor:C.dn,borderUpColor:C.up,borderDownColor:C.dn,wickUpColor:C.up,wickDownColor:C.dn});
  equityChart=LightweightCharts.createChart($("chart-equity"),{...baseOpts(118),
    rightPriceScale:{borderColor:C.baseline,scaleMargins:{top:0.15,bottom:0.1}},timeScale:{visible:false},handleScroll:false,handleScale:false});
  equitySeries=equityChart.addAreaSeries({lineColor:C.accent,lineWidth:2,topColor:"rgba(0,210,255,0.22)",bottomColor:"rgba(0,210,255,0.02)",priceLineVisible:false});
  new ResizeObserver(()=>{ mainChart.applyOptions({width:$("chart-main").clientWidth}); equityChart.applyOptions({width:$("chart-equity").clientWidth}); }).observe($("chart-main"));
}
function tradeMarkers(markers){ return markers.slice().sort((a,b)=>a.ts-b.ts).map(m=>m.kind==="entry"?
  {time:Math.floor(m.ts/1000),position:m.side==="LONG"?"belowBar":"aboveBar",color:m.side==="LONG"?C.up:C.dn,shape:m.side==="LONG"?"arrowUp":"arrowDown",text:m.side==="LONG"?"L":"S"}:
  {time:Math.floor(m.ts/1000),position:"inBar",color:(m.pnl??0)>=0?C.up:C.dn,shape:"circle"}); }

/* ---------------------------------------------------------------- state */
let S=null, curSymbol=null, lastTradeCount=-1;
const symbols=()=>S?.config?.symbols??[];
const engSym=()=>S?.engine?.symbols?.[curSymbol];

async function refreshCandles(full=false){
  if(!curSymbol||!S?.engine) return;
  try{
    const d=await api(`/api/candles?symbol=${encodeURIComponent(curSymbol)}&limit=${full?500:3}`);
    if(!d.candles.length) return;
    if(full){ candleSeries.setData(d.candles); candleSeries.setMarkers(tradeMarkers(d.markers)); mainChart.timeScale().scrollToRealTime(); }
    else { for(const c of d.candles) candleSeries.update(c);
      if(S.engine.portfolio.stats.trades!==lastTradeCount) candleSeries.setMarkers(tradeMarkers(d.markers)); }
  }catch(e){}
}
function setSymbol(sym,force=false){ if(!sym||(sym===curSymbol&&!force)) return; curSymbol=sym;
  document.querySelectorAll(".sym-tab").forEach(b=>b.classList.toggle("active",b.dataset.sym===sym));
  $("cycle-sym").textContent=sym; refreshCandles(true); }

/* ---------------------------------------------------------------- renderers */
function renderTop(){
  const mode=S.mode, pill=$("mode-pill"); pill.className=`pill ${mode}`; $("mode-text").textContent=mode.toUpperCase();
  if($("mode-select").value!==mode) $("mode-select").value=mode;
  const healthy=!!S.engine?.feed_healthy; $("feed-dot").className="dot"+(healthy?" ok":"");
  $("feed-label").textContent=S.engine?(S.config.feed==="synthetic"?"synthetic":"BingX"):"no feed";
  const es=engSym(); $("lat").textContent=es?`${es.eval_ms} ms`:"— ms";
  const pf=S.engine?.portfolio, st=pf?.stats;
  $("t-eq").textContent=pf?fmt.usd(pf.equity):"—";
  const day=S.engine?.risk?.day_realized??null;
  $("t-day").textContent=day==null?"—":fmt.signed(day,2); $("t-day").className="v "+pnlCls(day??0);
  $("t-wr").textContent=st&&st.trades?fmt.pct(st.win_rate):"—";
  $("t-tr").textContent=st?String(st.trades):"—";
  const h=S.engine?.risk?.health?.scalar; $("t-health").textContent=h!=null?`${(h*100).toFixed(0)}%`:"—";
}
let autoFollow=true;   // the chart follows whatever symbol the machine is looking at
function engineSymbols(){ const es=S?.engine?.symbols; return es?Object.keys(es):symbols(); }
function followFocus(){
  if(!autoFollow) return;
  const f=S?.engine?.focus;
  if(f&&f!==curSymbol&&S?.engine?.symbols?.[f]) setSymbol(f);
}
function renderSymTabs(){
  const wrap=$("sym-tabs"), syms=engineSymbols();
  const adopted=new Set(S?.engine?.adopted||[]);
  const sig=syms.join(",")+"|"+[...adopted].join(",");
  if(wrap.dataset.sig!==sig){
    wrap.dataset.sig=sig;
    wrap.innerHTML="";
    const a=document.createElement("button");
    a.className="sym-tab auto"+(autoFollow?" on":""); a.id="auto-follow-btn"; a.textContent="◉ AUTO";
    a.title="Chart follows the symbol the machine is looking at (position first, else closest to firing)";
    a.onclick=()=>{ autoFollow=!autoFollow; a.classList.toggle("on",autoFollow); if(autoFollow) followFocus(); };
    wrap.appendChild(a);
    for(const s of syms){
      const b=document.createElement("button");
      b.className="sym-tab"+(adopted.has(s)?" adopted":""); b.dataset.sym=s;
      b.textContent=s.replace("-USDT","")+(adopted.has(s)?" ◈":"");
      if(adopted.has(s)) b.title="Adopted by the radar (trending) — auto-released when the trend dies";
      b.onclick=()=>{ autoFollow=false; $("auto-follow-btn")?.classList.remove("on"); setSymbol(s); };
      wrap.appendChild(b);
    }
    setSymbol(syms.includes(curSymbol)?curSymbol:syms[0],true);
  }
  followFocus();
}
function renderTape(){
  const tape=S.engine?.tape??[]; const track=$("tape-track");
  if(!tape.length){ track.innerHTML=`<span class="tape-item" style="color:var(--muted)">awaiting fills…</span>`; return; }
  const items=tape.slice().reverse().map(t=>{
    const tag=t.kind==="OPEN"?`<span class="tag open">OPEN</span>`:`<span class="tag close">CLOSE</span>`;
    const px=fmt.px(t.price); const extra=t.kind==="OPEN"?`P${Math.round((t.p_win||0)*100)}%`:
      `<span class="${pnlCls(t.pnl)}">${fmt.signed(t.pnl,2)}</span>`;
    return `<span class="tape-item">${tag} <b>${esc(t.symbol.replace("-USDT",""))}</b> <span class="${t.side==='LONG'?'up':'dn'}">${t.side}</span> ${px} ${extra}</span>`;
  }).join("");
  track.innerHTML=items+items;   // duplicate for seamless marquee
}
function renderPipeline(){
  const es=engSym(); const stages=S.engine?.stages??["SCAN","DETECT","VALIDATE","SIZE","FILL","MANAGE","SETTLE"];
  const cur=es?.stage||"SCAN"; const ci=stages.indexOf(cur);
  $("pipe").innerHTML=stages.map((s,i)=>{
    const cls=i===ci?"on":(ci>=0&&i<ci?"done":"");
    return `<div class="pstage ${cls}"><div class="n">${String(i+1).padStart(2,"0")}</div><div class="l">${s}</div></div>`;
  }).join("");
}
function renderGates(es){
  // the entry-gate X-ray: every rung of the entry chain with live numbers —
  // the failing rung is exactly why the machine is holding fire.
  const el=$("gate-list"); if(!el) return;
  const held=S?.engine?.portfolio?.open_positions?.[curSymbol];
  if(held){ el.innerHTML=`<span class="mtf-empty">in position — gates re-arm on exit</span>`; return; }
  const g=es?.gates||[];
  if(!g.length){ el.innerHTML=`<span class="mtf-empty">warming up…</span>`; return; }
  el.innerHTML=g.map(x=>`<div class="gate ${x.ok?'pass':'fail'}" title="${esc(x.d)}">
    <span class="gd">${x.ok?"▮":"▯"}</span><span class="gn">${esc(x.n)}</span><span class="gv">${esc(x.d)}</span></div>`).join("");
}
function renderMTF(es){
  const strip=$("mtf-strip"); if(!strip) return;
  const mtf=es?.mtf||{};
  const order=["1m","5m","15m","1h"].filter(tf=>mtf[tf]);
  if(!order.length){ strip.innerHTML=`<span class="mtf-empty">warming up…</span>`; return; }
  strip.innerHTML=order.map(tf=>{
    const m=mtf[tf], d=m.dir||0;
    const cls=d>0.15?"up":d<-0.15?"dn":"flat";
    const arrow=d>0.15?"▲":d<-0.15?"▼":"▬";
    const w=Math.round(Math.abs(clamp(d,-1,1))*100);
    return `<div class="mtf-cell ${cls}"><div class="tf">${tf}</div>
      <div class="dir">${arrow}</div>
      <div class="tfbar"><div class="tffill" style="width:${w}%"></div></div>
      <div class="tfrsi">RSI ${Math.round(m.rsi)}</div></div>`;
  }).join("");
}
function renderEdgeGauge(b, es){
  // price + edge/p(win) gauges + entry gate — the elements that must feel live,
  // so both the full render and the fast 'hot' channel call this.
  $("px-last").textContent=fmt.px(es.price);
  const edge=b.edge||0, thr=b.threshold||0.3;
  $("edge-val").textContent=fmt.signed(edge,2);
  $("edge-val").style.color=Math.abs(edge)<thr?"var(--ink)":(edge>0?"#5fe8ff":"#ff86b0");
  const nd=$("edge-needle"); nd.style.left=`calc(${50+clamp(edge,-1,1)*49}% - 2px)`;
  nd.style.background=Math.abs(edge)<thr?C.ink2:(edge>0?"#5fe8ff":"#ff86b0");
  $("edge-thr-pos").style.left=`${50+thr*49}%`; $("edge-thr-neg").style.left=`${50-thr*49}%`;
  $("edge-thr").textContent=`thr ${thr.toFixed(2)}`;
  const p=b.p_win||0.5; $("pwin-val").textContent=fmt.pct(p,0);
  $("pwin-val").style.color=p>=0.55?"var(--good)":p>=0.5?"var(--ink)":"var(--bad)";
  $("pwin-fill").style.width=`${clamp((p-0.3)/0.6,0,1)*100}%`;
  $("pwin-fill").style.background=p>=0.55?"var(--good)":p>=0.5?"var(--accent)":"var(--bad)";
  const held=S.engine?.portfolio?.open_positions?.[curSymbol];
  $("b-gate").textContent=held?`in position ${es.bars_held}b`:(es.entry_block?es.entry_block:(Math.abs(edge)>=thr?"armed":"scanning"));
}
function renderBrain(){
  const es=engSym(); if(!es) return;
  const b=es.brain, micro=es.micro, ctx=es.context||{};
  const fund=ctx.funding_rate!=null?` · fund ${(ctx.funding_rate*100).toFixed(4)}%`:"";
  const tf=S?.engine?.interval||"";
  $("px-meta").textContent=`1m chart · ${tf} signals · spread ${micro.spread_bps.toFixed(1)}bp · OBI ${fmt.signed(micro.obi,2)} · flow ${fmt.signed(micro.flow,2)}${fund}`;
  $("brain-graded").textContent=`${b.graded} graded`;

  // edge + p(win) gauges (also driven by the fast 'hot' channel)
  renderEdgeGauge(b, es);
  const cal=b.calibration||{}; $("cal-skill").textContent=`skill ${fmt.signed(cal.skill||0,2)}`;

  // badges
  const rm=REGIME_META[b.regime]||REGIME_META.RANGE; const rb=$("b-regime");
  rb.className=`badge ${rm.cls}`; rb.innerHTML=`<span class="g">${rm.g}</span><span>${rm.label}</span>`;
  $("b-conf").textContent=`conf ${fmt.pct(b.regime_conf,0)}`;
  $("b-vol").textContent=`vol ${(micro.spread_bps).toFixed(1)}bp`;
  renderMTF(es);
  renderGates(es);

  // desks
  renderDesks(b.desks);
  // kvs
  $("kv-beta").textContent=(b.beta||0).toFixed(2);
  $("kv-brier").textContent=(cal.brier!=null?cal.brier.toFixed(3):"—");
  $("kv-bars").textContent=`${es.bars}${es.bars<es.warmup_bars?` / ${es.warmup_bars} warmup`:""}`;
  $("kv-graded").textContent=b.graded;
  const risk=S.engine.risk;
  $("kv-risk").textContent=risk.killed?`KILLED`:"normal"; $("kv-risk").style.color=risk.killed?"var(--bad)":"";
  $("kv-cool").textContent=risk.cooldown_s>0?fmt.dur(risk.cooldown_s):"—";
  const hv=risk.health||{}; $("kv-health").textContent=`${(hv.scalar*100||0).toFixed(0)}%  ·  dd ${fmt.pct(hv.drawdown||0)}  ·  exp ${fmt.signed(hv.recent_expectancy||0,2)}R`;
  const hf=$("health-fill"); hf.style.width=`${clamp((hv.scalar||1)/1.3,0,1)*100}%`;
  hf.style.background=(hv.scalar||1)>=0.9?"var(--good)":(hv.scalar||1)>=0.6?"var(--warn)":"var(--bad)";

  renderAlphaFloor(b.alphas);
}
function renderDesks(desks){
  if(!desks) return;
  const maxA=Math.max(...DESK_ORDER.map(d=>desks[d]?.alloc||0),0.001);
  $("desks").innerHTML=DESK_ORDER.map(d=>{
    const v=desks[d]||{}; const col=DESK_COLORS[d];
    const off=v.disabled?`<span class="off">MUTED</span>`:"";
    return `<div class="desk ${v.disabled?'disabled':''}">
      <div class="dn"><span class="sw" style="background:${col}"></span>${DESK_LABEL[d]}</div>
      <div class="track"><div class="fill" style="width:${(v.alloc/maxA)*100}%;background:${col}"></div><span class="alloc">${fmt.pct(v.alloc,0)}</span></div>
      <div class="meta">sig <b>${fmt.signed(v.signal||0,2)}</b> · win <b>${fmt.pct(v.win||0,0)}</b><br>shrp <b>${fmt.signed(v.sharpe||0,2)}</b> ${off}</div>
    </div>`;
  }).join("");
}
function renderAlphaFloor(alphas){
  if(!alphas) return;
  const byDesk={}; for(const [nm,a] of Object.entries(alphas)){ (byDesk[a.desk]=byDesk[a.desk]||[]).push([nm,a]); }
  const wrap=$("alpha-desks");
  wrap.innerHTML=DESK_ORDER.filter(d=>byDesk[d]).map(d=>{
    const col=DESK_COLORS[d];
    const rows=byDesk[d].map(([nm,a])=>{
      const scCls=a.score>0.05?"sc-pos":a.score<-0.05?"sc-neg":"sc-zero";
      const hr=a.calls>4?`${Math.round(a.hit_rate*100)}%`:"·";
      const wbar=`<div class="wt"><div class="fill" style="width:${clamp(a.weight/0.5,0,1)*100}%;background:${col};position:absolute;top:0;bottom:0;left:0;border-radius:3px;opacity:.8"></div></div>`;
      return `<div class="alpha ${a.state==='dormant'?'dim':''}"><span class="st ${a.state}"></span>
        <span class="nm">${nm}</span><span class="sc ${scCls}">${fmt.signed(a.score,2)}</span><span class="hr">${hr}</span></div>`;
    }).join("");
    return `<div><div class="adesk-h"><span class="sw" style="width:8px;height:8px;border-radius:2px;background:${col};display:inline-block"></span>${DESK_LABEL[d]}</div>
      <div class="alpha-grid">${rows}</div></div>`;
  }).join("");
}
function renderEquity(){
  const curve=S.engine?.equity_curve??[];
  if(curve.length>1){
    equitySeries.setData(curve.map(([ts,eq])=>({time:Math.floor(ts/1000),value:eq})));
    const eq=curve[curve.length-1][1], start=S.engine.portfolio.starting_balance, dlt=eq-start;
    $("eq-cap").textContent=`${fmt.usd(eq)}  (${fmt.signed(dlt,2)} / ${fmt.signed(dlt/start*100,2)}%)`;
    $("eq-cap").className="val "+pnlCls(dlt);
  }
}
function renderPositions(){
  const pf=S.engine?.portfolio, body=$("pos-body");
  const entries=pf?Object.entries(pf.open_positions):[];
  if(!entries.length){ body.innerHTML=`<tr><td colspan="11" class="empty">No open positions</td></tr>`; return; }
  body.innerHTML=entries.map(([sym,p])=>{ const mark=S.engine.symbols[sym]?.price??0;
    return `<tr><td>${esc(sym)}</td><td class="${sideCls(p.side)}">${p.side}</td><td class="r">${p.qty}</td>
      <td class="r">${fmt.px(p.entry)}</td><td class="r">${fmt.px(mark)}</td><td class="r">${fmt.px(p.stop)}</td>
      <td class="r">${fmt.px(p.tp)}</td><td class="r ${pnlCls(p.upnl)}">${fmt.signed(p.upnl,2)}</td>
      <td class="r">${p.leverage}x</td><td>${fmt.time(p.opened_ts)}</td>
      <td><button class="btn sm" onclick="closePos('${esc(sym)}')">Close</button></td></tr>`; }).join("");
}
function renderTrades(){
  const trades=(S.engine?.trades??[]).slice().reverse(), st=S.engine?.portfolio?.stats;
  $("trade-cards").innerHTML=!st?"":[
    ["Win rate",st.trades?fmt.pct(st.win_rate):"—"],["Profit factor",st.trades?st.profit_factor.toFixed(2):"—"],
    ["Trades",st.trades],["Net PnL",fmt.signed(st.total_pnl,2),pnlCls(st.total_pnl)],["Avg R",st.trades?fmt.signed(st.avg_r,2):"—"],
    ["Max DD",fmt.pct(st.max_drawdown)],["Sharpe~",st.sharpe_like],["Fees",fmt.usd(st.fees_paid)],
  ].map(([k,v,cls])=>`<div class="card"><div class="k">${k}</div><div class="v ${cls??""}">${v}</div></div>`).join("");
  const body=$("trades-body");
  if(!trades.length){ body.innerHTML=`<tr><td colspan="10" class="empty">No closed trades yet</td></tr>`; return; }
  body.innerHTML=trades.map(t=>`<tr><td>${fmt.time(t.exit_ts)}</td><td>${esc(t.symbol)}</td><td class="${sideCls(t.side)}">${t.side}</td>
    <td class="r">${t.qty}</td><td class="r">${fmt.px(t.entry_price)}</td><td class="r">${fmt.px(t.exit_price)}</td>
    <td class="r ${pnlCls(t.pnl)}">${fmt.signed(t.pnl,2)}</td><td class="r ${pnlCls(t.r_multiple)}">${fmt.signed(t.r_multiple,2)}</td>
    <td style="color:var(--muted)">${esc(t.reason_open)}</td><td style="color:var(--muted)">${esc(t.reason_close)}</td></tr>`).join("");
}
/* ------------------------- market radar + funding-carry desk ------------- */
function fmtVol(v){ if(v==null||isNaN(v)) return "—";
  if(v>=1e9) return (v/1e9).toFixed(1)+"B"; if(v>=1e6) return (v/1e6).toFixed(1)+"M";
  if(v>=1e3) return (v/1e3).toFixed(0)+"K"; return v.toFixed(0); }
function renderRadar(){
  const R=S?.radar, C_=S?.carry;
  const cards=$("carry-cards"), cbody=$("carry-body"), rbody=$("radar-body"); if(!cards) return;
  if(C_){
    const pos=C_.positions||[];
    cards.innerHTML=[
      ["Desk",C_.enabled?"● HARVESTING":"OFF",C_.enabled?"pnl-pos":""],
      ["Open carry",pos.length],
      ["Funding collected",fmt.signed(C_.funding_collected,4),pnlCls(C_.funding_collected)],
      ["Entries",C_.entries],["Exits",C_.exits],
      ["Last check",C_.last_reason||"—"],
    ].map(([k,v,cls])=>`<div class="card"><div class="k">${k}</div><div class="v ${cls??""}" style="font-size:13px">${esc(String(v))}</div></div>`).join("");
    cbody.innerHTML=pos.length?pos.map(p=>`<tr><td>${esc(p.symbol)}</td><td class="${sideCls(p.side)}">${p.side}</td>
      <td class="r">${p.qty}</td><td class="r">${fmt.px(p.entry)}</td><td class="r">${fmt.px(p.mark)}</td>
      <td class="r">${fmt.px(p.stop)}</td><td class="r ${p.apr>=0?'pnl-pos':'pnl-neg'}">${(p.apr*100).toFixed(0)}%</td>
      <td class="r ${pnlCls(p.upnl)}">${fmt.signed(p.upnl,2)}</td><td class="r">${p.held_h}h</td>
      <td>${p.next_funding_ts?fmt.time(p.next_funding_ts):"—"}</td></tr>`).join("")
      :`<tr><td colspan="10" class="empty">No carry positions — the desk waits for genuinely extreme funding</td></tr>`;
  }
  if(R&&rbody){
    $("radar-meta").textContent=R.ts?`· scan ${fmt.time(R.ts)}${R.demo?" · DEMO BOARD (synthetic feed)":""}${R.error?` · ⚠ ${R.error}`:""}`:"";
    const rows=R.rows||[];
    rbody.innerHTML=rows.length?rows.map((r,i)=>{
      const kindCls=r.kind==="carry"?"pnl-pos":(r.kind==="trend"?"sc-pos":"");
      const dir=r.dir_4h>0?"▲":(r.dir_4h<0?"▼":"·");
      const dirCls=r.dir_4h>0?"pnl-pos":(r.dir_4h<0?"pnl-neg":"");
      return `<tr><td style="color:var(--muted)">${i+1}</td><td><b>${esc(r.symbol)}</b></td>
        <td class="${kindCls}">${esc(r.kind)}</td>
        <td class="r ${Math.abs(r.funding_apr)>=0.2?'pnl-pos':''}">${(r.funding_apr*100).toFixed(0)}%</td>
        <td class="r ${sideCls(r.carry_side)}">${r.carry_side}</td>
        <td class="r">${fmtVol(r.quote_volume)}</td>
        <td class="r ${pnlCls(r.change_24h)}">${fmt.signed(r.change_24h,1)}%</td>
        <td class="r">${(r.er_4h||0).toFixed(2)}</td><td class="r ${dirCls}">${dir}</td>
        <td class="r">${(r.score||0).toFixed(2)}</td></tr>`;
    }).join(""):`<tr><td colspan="10" class="empty">Radar warming up…</td></tr>`;
  }
}

function renderAutotuner(){
  const at=S.autotuner; const row=$("at-row"), hist=$("at-history");
  if(!at){ row.innerHTML=`<div class="empty">Auto-tuner idle (engine not running)</div>`; return; }
  const next=at.next_run_ts?fmt.time(at.next_run_ts):"—";
  const lc=at.last_cycle;
  row.innerHTML=[
    ["Status",at.enabled?(at.running?"● RESEARCHING":"ON"):"OFF"],
    ["DE generation",at.generation??"—"],
    ["Population",at.population??"—"],
    ["Research cores",at.research_cores??"—"],
    ["Researching",at.research_symbol||"—"],
    ["Cycles run",at.cycles],
    ["Improvements",at.improvements],
    ["Champion fitness",at.champion_fitness??"—"],
    ["Last challenger",lc?`${lc.best_fitness} (${lc.promoted?"adopted":"kept"})`:"—"],
    ["Diversity",lc?.diversity??"—"],
    ["Next cycle",next],
  ].map(([k,v])=>`<div class="at-badge"><div class="k">${k}</div><div class="v">${esc(String(v))}</div></div>`).join("");
  const H=at.history||[];
  hist.innerHTML=H.length?H.map(h=>{
    const params=Object.entries(h.params||{}).map(([k,v])=>`${k}=${v}`).join("  ");
    return `<tr><td>${fmt.dt(h.ts)}</td><td class="r pnl-pos">${h.from_fitness} → ${h.to_fitness}</td>
      <td class="r">${fmt.pct(h.valid_wr,0)}</td><td class="r">${(h.valid_pf||0).toFixed(2)}</td>
      <td style="color:var(--muted)">${esc(params)}</td></tr>`;
  }).join(""):`<tr><td colspan="5" class="empty">No promotions yet — it only swaps genuine improvements</td></tr>`;
}
let settingsDirty=false;
const AUTO_PARAMS=[
  ["base_threshold","edge threshold","s"],["target_trades_per_hour","target trades/hr","s"],
  ["cost_multiple","cost multiple","s"],["min_p_win","min P(win)","s"],["kelly_fraction","Kelly fraction","s"],
  ["min_efficiency","min trend efficiency","s"],["hedge_eta","hedge learn rate","s"],["horizon_bars","grade horizon","s"],
  ["risk_per_trade","risk per trade","r"],["sl_atr_min","stop min ×ATR","r"],["sl_atr_max","stop max ×ATR","r"],
  ["trail_atr_min","trail min ×ATR","r"],["trail_atr_max","trail max ×ATR","r"],["trail_tighten","trail tighten","r"],
  ["be_rr","breakeven R","r"],["giveback_rr","giveback R","r"],["hold_edge_frac","edge-flip exit","r"],["time_stop_bars","time stop bars","r"],
];
function renderSettings(){
  if(settingsDirty||!S) return; const c=S.config;
  $("cfg-symbols").value=c.symbols.join(", "); $("cfg-feed").value=c.feed; $("cfg-interval").value=c.strategy.interval;
  $("cfg-balance").value=c.paper.starting_balance; $("cfg-maxpos").value=c.risk.max_open_positions;
  $("cfg-levmin").value=c.risk.min_leverage; $("cfg-levmax").value=c.risk.max_leverage;
  $("cfg-dayloss").value=c.risk.max_daily_loss_pct; $("cfg-hardrisk").value=c.risk.max_risk_hard_pct;
  $("cfg-autotune").checked=c.strategy.auto_tune; $("cfg-allowlive").checked=c.allow_live;
  $("cfg-adopt").value=c.strategy.adopt_symbols??2;
  if(c.carry){ $("cfg-carry").checked=c.carry.enabled; $("cfg-carrymax").value=c.carry.max_positions; }
  $("cfg-keys").textContent=c.has_keys?"configured ✓":"not set (paper/backtest only)"; $("cfg-keys").style.color=c.has_keys?"var(--good)":"";
  $("auto-params").innerHTML=AUTO_PARAMS.map(([k,lab,grp])=>{
    const v=(grp==="s"?c.strategy:c.risk)[k];
    const val=typeof v==="number"?(Math.abs(v)<1?v.toFixed(3):v.toFixed(2)):v;
    return `<div style="display:flex;justify-content:space-between"><span style="color:var(--muted)">${lab}</span><span style="color:var(--ink)">${val}</span></div>`;
  }).join("");
}
function renderAll(){
  if(!S) return;
  renderTop(); renderSymTabs(); renderTape();
  if(S.engine){ renderPipeline(); renderBrain(); renderEquity(); renderPositions(); renderTrades();
    const tc=S.engine.portfolio.stats.trades; refreshCandles(false).then(()=>{lastTradeCount=tc;}); }
  renderAutotuner(); renderChampions(); renderRadar(); renderSettings();
}

/* ------- fast 'hot' channel: patch the live numbers between full pushes ----- */
function applyHot(h){
  if(!S||!S.engine||!h?.engine) return;
  const he=h.engine; S.mode=h.mode??S.mode;
  const pf=S.engine.portfolio; if(pf) pf.equity=he.equity;
  if(typeof he.killed==="boolean"&&S.engine.risk) S.engine.risk.killed=he.killed;
  if(typeof he.feed_healthy==="boolean") S.engine.feed_healthy=he.feed_healthy;
  if(he.focus) S.engine.focus=he.focus;
  if(he.adopted) S.engine.adopted=he.adopted;
  for(const [sym,hs] of Object.entries(he.symbols||{})){
    const s=S.engine.symbols?.[sym]; if(!s) continue;
    s.price=hs.price; s.stage=hs.stage; s.eval_ms=hs.eval_ms; s.entry_block=hs.entry_block; s.bars_held=hs.bars_held;
    if(hs.mtf) s.mtf=hs.mtf;
    if(hs.gates) s.gates=hs.gates;
    if(hs.candle) s.candle=hs.candle;
    if(s.brain){ s.brain.edge=hs.edge; s.brain.p_win=hs.p_win; s.brain.regime=hs.regime; }
  }
  for(const [sym,hp] of Object.entries(he.positions||{})){
    const p=pf?.open_positions?.[sym]; if(p) p.upnl=hp.upnl;
  }
  if(he.tape) S.engine.tape=he.tape;
  renderHot();
}
let lastEqT=0;
function renderHot(){
  if(!S?.engine) return;
  renderTop(); renderPipeline(); renderPositions(); renderTape();
  renderSymTabs();   // adopted set + auto-follow react at hot cadence
  const es=engSym(); if(es&&es.brain){ renderEdgeGauge(es.brain, es); renderMTF(es); renderGates(es); }
  // live-forming candle straight off the hot channel — the chart moves at tick
  // cadence now instead of waiting for the next REST poll.
  if(es?.candle&&candleSeries){
    try{ candleSeries.update({time:es.candle.t,open:es.candle.o,high:es.candle.h,low:es.candle.l,close:es.candle.c}); }catch(e){}
  }
  const eq=S.engine.portfolio?.equity;
  if(eq!=null&&equitySeries){
    const t=Math.floor(Date.now()/1000);
    if(t>lastEqT){ lastEqT=t; try{ equitySeries.update({time:t,value:eq}); }catch(e){} }
  }
}

/* ---------------------------------------------------------------- ws */
let ws,wsRetry=1;
function connectWS(){
  const proto=location.protocol==="https:"?"wss":"ws";
  ws=new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage=(ev)=>{ const m=JSON.parse(ev.data);
    if(m.type==="state"){ S=m.data; renderAll(); }
    else if(m.type==="hot"){ applyHot(m.data); } };
  ws.onopen=()=>{wsRetry=1;}; ws.onclose=()=>setTimeout(connectWS,Math.min(wsRetry*=1.6,8)*1000); ws.onerror=()=>ws.close();
}

/* ---------------------------------------------------------------- actions */
window.closePos=async(sym)=>{ try{ await api("/api/control",{action:"close",symbol:sym}); toast(`${sym} closed`,"good"); }catch(e){ toast(e.message,"bad"); } };
$("btn-kill").onclick=async()=>{ if(!confirm("Kill switch: flatten all and halt entries?")) return;
  try{ await api("/api/control",{action:"kill"}); toast("Kill switch engaged","warn"); }catch(e){ toast(e.message,"bad"); } };
$("btn-flatten").onclick=async()=>{ try{ const r=await api("/api/control",{action:"flatten"}); toast(r.message,"good"); }catch(e){ toast(e.message,"bad"); } };
$("btn-reset-kill").onclick=async()=>{ try{ const r=await api("/api/control",{action:"reset_kill"}); toast(r.message,"good"); }catch(e){ toast(e.message,"bad"); } };
$("btn-paper-reset").onclick=async()=>{
  if(!confirm("Reset the paper account? The persisted session (positions, trades, equity history) is wiped.")) return;
  try{ await api("/api/paper_reset"); lastEqT=0; toast("Paper account reset — fresh balance","good"); }catch(e){ toast(e.message,"bad"); } };
$("mode-select").onchange=async(ev)=>{ const mode=ev.target.value; if(mode==="live"){ openLiveModal(); return; }
  try{ const r=await api("/api/mode",{mode}); toast(r.message,"good"); }catch(e){ toast(e.message,"bad"); ev.target.value=S?.mode??"idle"; } };
function openLiveModal(){ $("live-phrase").textContent=S?.live_confirm_phrase??"TRADE LIVE"; $("live-confirm-input").value=""; $("live-go").disabled=true; $("live-modal").classList.add("open"); }
$("live-confirm-input").oninput=(ev)=>{ $("live-go").disabled=ev.target.value!==(S?.live_confirm_phrase??"TRADE LIVE"); };
$("live-cancel").onclick=()=>{ $("live-modal").classList.remove("open"); $("mode-select").value=S?.mode??"idle"; };
$("live-go").onclick=async()=>{ try{ const r=await api("/api/mode",{mode:"live",confirm:$("live-confirm-input").value});
  toast(r.message,r.ok===false?"bad":"warn"); $("live-modal").classList.remove("open"); }catch(e){ toast(e.message,"bad"); } };

document.querySelectorAll(".tab").forEach(b=>{ b.onclick=()=>{
  document.querySelectorAll(".tab").forEach(x=>x.classList.toggle("active",x===b));
  document.querySelectorAll(".tab-page").forEach(p=>p.classList.toggle("active",p.dataset.page===b.dataset.tab));
  if(b.dataset.tab==="backtest") ensureBtCharts();
  if(b.dataset.tab==="portfolio") ensurePfChart();
  if(b.dataset.tab==="walkforward") ensureWfChart();
  if(b.dataset.tab==="analytics") loadAnalytics();
  if(b.dataset.tab==="record") loadRecord();
}; });
document.querySelectorAll('[data-page="settings"] input, [data-page="settings"] select').forEach(el=>el.addEventListener("input",()=>{settingsDirty=true;}));
$("cfg-save").onclick=async()=>{
  const patch={ symbols:$("cfg-symbols").value.split(",").map(s=>s.trim().toUpperCase()).filter(Boolean),
    feed:$("cfg-feed").value, allow_live:$("cfg-allowlive").checked,
    strategy:{ interval:$("cfg-interval").value, auto_tune:$("cfg-autotune").checked,
      adopt_symbols:parseInt($("cfg-adopt").value,10) },
    risk:{ min_leverage:parseInt($("cfg-levmin").value,10), max_leverage:parseInt($("cfg-levmax").value,10),
      max_daily_loss_pct:parseFloat($("cfg-dayloss").value), max_risk_hard_pct:parseFloat($("cfg-hardrisk").value),
      max_open_positions:parseInt($("cfg-maxpos").value,10) },
    carry:{ enabled:$("cfg-carry").checked, max_positions:parseInt($("cfg-carrymax").value,10) },
    paper:{ starting_balance:parseFloat($("cfg-balance").value) } };
  try{ const r=await api("/api/config",{patch}); settingsDirty=false;
    toast(r.needs_restart?"Saved — switch to Idle and back to apply":"Settings saved","good"); }catch(e){ toast(e.message,"bad"); }
};

/* ---------------------------------------------------------------- jobs */
async function pollJob(jobId,progressEl,onDone){
  progressEl.style.display="block"; const bar=progressEl.querySelector(".bar");
  const tick=async()=>{ try{ const j=await api(`/api/jobs/${jobId}`); bar.style.width=`${(j.progress*100).toFixed(1)}%`;
    if(j.done){ progressEl.style.display="none"; if(j.error) toast(`Job failed: ${j.error}`,"bad"); else onDone(j.result); return; } }catch(e){}
    setTimeout(tick,700); }; tick();
}
function ensureBtCharts(){
  if(btEquityChart) return;
  btEquityChart=LightweightCharts.createChart($("chart-bt-equity"),baseOpts(220));
  btEquitySeries=btEquityChart.addAreaSeries({lineColor:C.accent,lineWidth:2,topColor:"rgba(0,210,255,0.22)",bottomColor:"rgba(0,210,255,0.02)",priceLineVisible:false});
  btAllocChart=LightweightCharts.createChart($("chart-bt-alloc"),{...baseOpts(220),rightPriceScale:{borderColor:C.baseline,scaleMargins:{top:0.08,bottom:0.08}}});
  for(const d of DESK_ORDER) btAllocSeries[d]=btAllocChart.addLineSeries({color:DESK_COLORS[d],lineWidth:2,priceLineVisible:false,lastValueVisible:false,title:d});
  new ResizeObserver(()=>{ btEquityChart.applyOptions({width:$("chart-bt-equity").clientWidth}); btAllocChart.applyOptions({width:$("chart-bt-alloc").clientWidth}); }).observe($("chart-bt-equity"));
}
function statCards(st,start){ return [
  ["Win rate",st.trades?fmt.pct(st.win_rate):"—",st.win_rate>=0.55?"pnl-pos":""],
  ["Profit factor",st.trades?st.profit_factor.toFixed(2):"—",st.profit_factor>=1?"pnl-pos":"pnl-neg"],
  ["Trades",st.trades],["Net PnL",fmt.signed(st.total_pnl,2),pnlCls(st.total_pnl)],
  ["Return",fmt.signed(st.total_pnl/start*100,2)+"%",pnlCls(st.total_pnl)],["Max DD",fmt.pct(st.max_drawdown)],
  ["Avg R",st.trades?fmt.signed(st.avg_r,2):"—",pnlCls(st.avg_r)],["Sharpe~",st.sharpe_like],["Fees",fmt.usd(st.fees_paid)],
].map(([k,v,cls])=>`<div class="card"><div class="k">${k}</div><div class="v ${cls??""}">${v}</div></div>`).join(""); }

$("bt-run").onclick=async()=>{ try{ const r=await api("/api/backtest",{symbol:$("bt-symbol").value.trim().toUpperCase(),
  interval:$("bt-interval").value,days:parseFloat($("bt-days").value),synthetic:$("bt-synth").checked});
  $("bt-results").style.display="none"; pollJob(r.job_id,$("bt-progress"),renderBacktest); }catch(e){ toast(e.message,"bad"); } };

function renderBacktest(res){
  ensureBtCharts(); $("bt-results").style.display="block";
  if(res.error){ toast(res.error,"bad"); return; }
  $("bt-cards").innerHTML=statCards(res.stats,res.starting_balance);
  requestAnimationFrame(()=>{
    btEquityChart.applyOptions({width:$("chart-bt-equity").clientWidth}); btAllocChart.applyOptions({width:$("chart-bt-alloc").clientWidth});
    btEquitySeries.setData(res.equity_curve.map(([ts,eq])=>({time:Math.floor(ts/1000),value:eq}))); btEquityChart.timeScale().fitContent();
    for(const d of DESK_ORDER) btAllocSeries[d].setData((res.weights_timeline??[]).map(w=>({time:Math.floor(w.ts/1000),value:w[d]??0})));
    btAllocChart.timeScale().fitContent();
    monteCarlo(res.trades||[], res.starting_balance);
  });
  const trades=(res.trades??[]).slice(-200).reverse();
  $("bt-trades-body").innerHTML=trades.length?trades.map(t=>`<tr><td>${fmt.dt(t.exit_ts)}</td><td class="${sideCls(t.side)}">${t.side}</td>
    <td class="r">${fmt.px(t.entry_price)}</td><td class="r">${fmt.px(t.exit_price)}</td><td class="r ${pnlCls(t.pnl)}">${fmt.signed(t.pnl,2)}</td>
    <td class="r">${fmt.signed(t.r_multiple,2)}</td><td style="color:var(--muted)">${esc(t.reason_close)}</td></tr>`).join("")
    :`<tr><td colspan="7" class="empty">No trades in this window</td></tr>`;
  const s=res.stats; toast(`Backtest: ${s.trades} trades · WR ${fmt.pct(s.win_rate)} · PF ${s.profit_factor.toFixed(2)}`,s.total_pnl>=0?"good":"warn");
}

/* Monte Carlo bootstrap over the trade PnL sequence, drawn on a canvas. */
function monteCarlo(trades,start){
  const host=$("mc-chart"); host.innerHTML="";
  const pnls=trades.map(t=>t.pnl).filter(x=>isFinite(x));
  const stats=$("mc-stats");
  if(pnls.length<10){ host.innerHTML=`<div class="empty">need ≥10 trades</div>`; stats.innerHTML=""; return; }
  const N=5000, K=pnls.length, finals=new Float64Array(N);
  for(let i=0;i<N;i++){ let sum=0; for(let j=0;j<K;j++) sum+=pnls[(Math.random()*K)|0]; finals[i]=sum; }
  finals.sort();
  const pct=(q)=>finals[Math.min(N-1,Math.floor(q*N))];
  const pProfit=finals.filter(x=>x>0).length/N, p5=pct(0.05), p50=pct(0.5), p95=pct(0.95), expv=finals.reduce((a,b)=>a+b,0)/N;
  // histogram canvas
  const w=host.clientWidth||500, h=150, cv=document.createElement("canvas");
  cv.width=w*devicePixelRatio; cv.height=h*devicePixelRatio; cv.style.width=w+"px"; cv.style.height=h+"px";
  host.appendChild(cv); const g=cv.getContext("2d"); g.scale(devicePixelRatio,devicePixelRatio);
  const bins=48, lo=finals[0], hi=finals[N-1], span=(hi-lo)||1, counts=new Array(bins).fill(0);
  for(const x of finals) counts[Math.min(bins-1,Math.floor((x-lo)/span*bins))]++;
  const maxC=Math.max(...counts);
  for(let i=0;i<bins;i++){ const x0=lo+i/bins*span, bh=counts[i]/maxC*(h-16);
    g.fillStyle=x0>=0?"rgba(22,192,96,0.75)":"rgba(240,85,90,0.75)";
    g.fillRect(i/bins*w, h-bh-4, w/bins-1, bh); }
  const zeroX=(0-lo)/span*w; g.strokeStyle="#333a44"; g.beginPath(); g.moveTo(zeroX,0); g.lineTo(zeroX,h); g.stroke();
  stats.innerHTML=[
    ["P(profit)",fmt.pct(pProfit,1),pProfit>=0.5?"pnl-pos":"pnl-neg"],
    ["Expected",fmt.signed(expv,0),pnlCls(expv)],
    ["5th pctile",fmt.signed(p5,0),pnlCls(p5)],
    ["95th pctile",fmt.signed(p95,0),pnlCls(p95)],
  ].map(([k,v,cls])=>`<div class="card"><div class="k">${k}</div><div class="v ${cls??""}">${v}</div></div>`).join("");
}

$("op-run").onclick=async()=>{ try{ const r=await api("/api/optimize",{symbol:$("op-symbol").value.trim().toUpperCase(),
  interval:$("op-interval").value,days:parseFloat($("op-days").value),trials:parseInt($("op-trials").value,10),synthetic:$("op-synth").checked});
  $("op-results").style.display="none"; pollJob(r.job_id,$("op-progress"),renderOptimizer); }catch(e){ toast(e.message,"bad"); } };
let opFinalists=[];
function renderOptimizer(res){
  $("op-results").style.display="block"; if(res.error){ toast(res.error,"bad"); return; }
  opFinalists=res.finalists??[];
  $("op-body").innerHTML=opFinalists.length?opFinalists.map((f,i)=>{ const v=f.valid??{};
    const params=Object.entries(f.params).map(([k,val])=>`${k}=${val}`).join("  ");
    return `<tr><td>${i+1}</td><td class="r ${f.valid_fitness>0?'pnl-pos':'pnl-neg'}">${f.valid_fitness}</td>
      <td class="r">${v.win_rate!=null?fmt.pct(v.win_rate):"—"}</td><td class="r">${v.profit_factor!=null?v.profit_factor.toFixed(2):"—"}</td>
      <td class="r">${v.trades??"—"}</td><td class="r">${f.train_fitness}</td>
      <td style="color:var(--muted);max-width:420px">${esc(params)}</td><td><button class="btn sm primary" onclick="applyParams(${i})">Apply</button></td></tr>`;
  }).join(""):`<tr><td colspan="8" class="empty">No viable finalists — try more days or trials</td></tr>`;
  toast(`Optimizer done: ${opFinalists.length} finalists`,"good");
}
window.applyParams=async(i)=>{ const f=opFinalists[i]; if(!f) return;
  try{ await api("/api/apply_params",{params:f.params}); toast("Parameters applied to running brains","good"); }catch(e){ toast(e.message,"bad"); } };

/* ---------------------------------------------------------------- portfolio */
function ensurePfChart(){
  if(pfEquityChart) return;
  pfEquityChart=LightweightCharts.createChart($("chart-pf-equity"),baseOpts(240));
  pfEquitySeries=pfEquityChart.addAreaSeries({lineColor:C.accent,lineWidth:2,topColor:"rgba(0,210,255,0.22)",bottomColor:"rgba(0,210,255,0.02)",priceLineVisible:false});
  new ResizeObserver(()=>{ pfEquityChart.applyOptions({width:$("chart-pf-equity").clientWidth}); }).observe($("chart-pf-equity"));
}
$("pf-run").onclick=async()=>{
  const symbols=$("pf-symbols").value.split(",").map(s=>s.trim().toUpperCase()).filter(Boolean);
  if(symbols.length<2){ toast("Enter at least 2 symbols","bad"); return; }
  try{ const r=await api("/api/portfolio_backtest",{symbols,interval:$("pf-interval").value,
    days:parseFloat($("pf-days").value),synthetic:$("pf-synth").checked});
    $("pf-results").style.display="none"; pollJob(r.job_id,$("pf-progress"),renderPortfolio); }catch(e){ toast(e.message,"bad"); }
};
function renderPortfolio(res){
  ensurePfChart(); $("pf-results").style.display="block";
  if(res.error){ toast(res.error,"bad"); return; }
  $("pf-cards").innerHTML=statCards(res.stats,res.starting_balance);
  requestAnimationFrame(()=>{
    pfEquityChart.applyOptions({width:$("chart-pf-equity").clientWidth});
    pfEquitySeries.setData(res.equity_curve.map(([ts,eq])=>({time:Math.floor(ts/1000),value:eq})));
    pfEquityChart.timeScale().fitContent();
  });
  const ps=res.per_symbol||{};
  $("pf-symbols-body").innerHTML=Object.keys(ps).length?Object.entries(ps).map(([s,v])=>
    `<tr><td>${esc(s)}</td><td class="r">${v.trades}</td><td class="r">${v.trades?fmt.pct(v.win_rate):"—"}</td>
     <td class="r ${pnlCls(v.pnl)}">${fmt.signed(v.pnl,2)}</td></tr>`).join("")
    :`<tr><td colspan="4" class="empty">No symbols</td></tr>`;
  const corr=res.avg_correlation;
  const corrCls=corr==null?"":corr<0.3?"pnl-pos":corr<0.6?"":"pnl-neg";
  $("pf-div").innerHTML=[
    ["Symbols",res.symbols?res.symbols.length:0],
    ["Aligned bars",res.bars??"—"],
    ["Avg correlation",corr==null?"—":corr.toFixed(2),corrCls],
    ["Max DD",fmt.pct(res.stats.max_drawdown)],
  ].map(([k,v,cls])=>`<div class="card"><div class="k">${k}</div><div class="v ${cls??""}">${v}</div></div>`).join("");
  const s=res.stats; toast(`Portfolio: ${res.symbols.length} symbols · ${s.trades} trades · WR ${fmt.pct(s.win_rate)} · PF ${s.profit_factor.toFixed(2)}`,s.total_pnl>=0?"good":"warn");
}

/* champion vault — the tuner's live candidate pool: each set shown at BIRTH vs
   re-validated against TODAY, with its real executed track record and use count */
const CHAMP_KEYS=["base_threshold","risk_per_trade","sl_atr_min","trail_atr_max","giveback_rr","target_trades_per_hour"];
let champStore=[];
function renderChampions(){
  champStore=S?.champions||[];
  const body=$("champ-body"); if(!body) return;
  if(!champStore.length){ body.innerHTML=`<tr><td colspan="9" class="empty">No champions saved yet — the vault fills as the tuner promotes winners</td></tr>`; return; }
  body.innerHTML=champStore.map((c,i)=>{
    const params=CHAMP_KEYS.filter(k=>c.params&&c.params[k]!=null).map(k=>`${k}=${c.params[k]}`).join("  ");
    const bf=c.birth_fitness??c.fitness??0, cf=c.fitness??0;
    const arrow=cf>bf+1e-9?"▲":(cf<bf-1e-9?"▼":"·");
    const lv=c.live||{trades:0,pnl:0};
    const liveCell=lv.trades?`${lv.trades} · <span class="${pnlCls(lv.pnl)}">${fmt.signed(lv.pnl,2)}</span>`
                            :`<span style="color:var(--muted)">—</span>`;
    const badges=(c.active?`<span class="champ-live" title="Currently driving live trading">LIVE</span> `:"")
                +(c.top_used?`<span title="Top-10 most used — protected from pruning">🔥</span> `:"");
    return `<tr class="${c.active?'champ-active':''}">
      <td>${badges}${fmt.dt(c.born_ts)}</td>
      <td class="r"><span style="color:var(--muted)">${bf.toFixed(2)}</span> ${arrow} <span class="${cf>=0?'pnl-pos':'pnl-neg'}">${cf.toFixed(2)}</span></td>
      <td class="r">${fmt.pct(c.win_rate,0)}</td>
      <td class="r">${(c.profit_factor||0).toFixed(2)}</td>
      <td class="r" style="color:var(--muted)">${c.cur_trades??0}</td>
      <td class="r">${liveCell}</td>
      <td class="r">${c.uses??0}</td>
      <td style="color:var(--muted);max-width:340px">${esc(params)}</td>
      <td><button class="btn sm primary" onclick="applyChampion(${i})">Apply</button></td></tr>`;
  }).join("");
}
window.applyChampion=async(i)=>{ const c=champStore[i]; if(!c?.params) return;
  try{ await api("/api/apply_params",{params:c.params,champion_id:c.id}); toast("Champion applied — now driving live trades","good"); }catch(e){ toast(e.message,"bad"); } };

/* ---------------------------------------------------------------- walk-forward */
function ensureWfChart(){
  if(wfEquityChart) return;
  wfEquityChart=LightweightCharts.createChart($("chart-wf-equity"),baseOpts(240));
  wfEquitySeries=wfEquityChart.addAreaSeries({lineColor:C.accent,lineWidth:2,topColor:"rgba(0,210,255,0.22)",bottomColor:"rgba(0,210,255,0.02)",priceLineVisible:false});
  new ResizeObserver(()=>{ wfEquityChart.applyOptions({width:$("chart-wf-equity").clientWidth}); }).observe($("chart-wf-equity"));
}
$("wf-run").onclick=async()=>{ try{ const r=await api("/api/walkforward",{symbol:$("wf-symbol").value.trim().toUpperCase(),
  interval:$("wf-interval").value,days:parseFloat($("wf-days").value),folds:parseInt($("wf-folds").value,10),
  trials:parseInt($("wf-trials").value,10),synthetic:$("wf-synth").checked});
  $("wf-results").style.display="none"; pollJob(r.job_id,$("wf-progress"),renderWalkforward); }catch(e){ toast(e.message,"bad"); } };
function renderWalkforward(res){
  ensureWfChart(); $("wf-results").style.display="block";
  if(res.error){ toast(res.error,"bad"); return; }
  const ret=res.oos_return_pct;
  $("wf-cards").innerHTML=[
    ["OOS return",fmt.signed(ret,1)+"%",pnlCls(ret)],
    ["OOS win rate",res.oos_trades?fmt.pct(res.oos_win_rate):"—",res.oos_win_rate>=0.5?"pnl-pos":""],
    ["OOS profit factor",res.oos_profit_factor.toFixed(2),res.oos_profit_factor>=1?"pnl-pos":"pnl-neg"],
    ["OOS trades",res.oos_trades],
    ["Max drawdown",fmt.pct(res.oos_max_drawdown)],
    ["Final equity",fmt.usd(res.final_equity)],
  ].map(([k,v,cls])=>`<div class="card"><div class="k">${k}</div><div class="v ${cls??""}">${v}</div></div>`).join("");
  requestAnimationFrame(()=>{
    wfEquityChart.applyOptions({width:$("chart-wf-equity").clientWidth});
    wfEquitySeries.setData((res.equity_curve||[]).map(([ts,eq])=>({time:Math.floor(ts/1000),value:eq})));
    wfEquityChart.timeScale().fitContent();
  });
  $("wf-body").innerHTML=(res.per_fold||[]).map(f=>`<tr><td>${f.fold}</td>
    <td class="r ${pnlCls(f.return_pct)}">${fmt.signed(f.return_pct,1)}%</td>
    <td class="r">${f.trades?fmt.pct(f.win_rate):"—"}</td><td class="r">${(f.profit_factor||0).toFixed(2)}</td>
    <td class="r">${f.trades}</td><td class="r">${fmt.pct(f.max_drawdown)}</td>
    <td class="r">${f.tuned?"yes":"default"}</td></tr>`).join("");
  toast(`Walk-forward OOS: ${fmt.signed(ret,1)}% · WR ${fmt.pct(res.oos_win_rate)} · PF ${res.oos_profit_factor.toFixed(2)}`,ret>=0?"good":"warn");
}

/* ---------------------------------------------------------------- carry lab */
$("cl-run").onclick=async()=>{
  try{
    const r=await api("/api/carrylab",{days:parseFloat($("cl-days").value),top_n:parseInt($("cl-topn").value,10)});
    $("cl-results").style.display="none";
    pollJob(r.job_id,$("cl-progress"),renderCarryLab);
  }catch(e){ toast(e.message,"bad"); }
};
function renderCarryLab(res){
  $("cl-results").style.display="block";
  if(res.error){ toast(res.error,"bad"); return; }
  $("cl-note").textContent=res.demo?"DEMO DATA (no exchange access) — run on your machine for real funding history":"";
  const rec=res.recommend;
  const cur=res.current||{};
  $("cl-cards").innerHTML=[
    ["Days",res.days],["Symbols",(res.symbols||[]).length],
    ["Current thresholds",`${(cur.min_apr*100).toFixed(0)}% / ${(cur.exit_apr*100).toFixed(0)}%`],
    ["Evidence pick",rec?`enter ≥${(rec.min_apr*100).toFixed(0)}% · exit <${(rec.exit_apr*100).toFixed(0)}%`:"no combo traded"],
    ["Net @ pick",rec?fmt.signed(rec.net*100,2)+"%":"—",rec&&rec.net>0?"pnl-pos":"pnl-neg"],
    ["Entries @ pick",rec?rec.entries:"—"],
  ].map(([k,v,cls])=>`<div class="card"><div class="k">${k}</div><div class="v ${cls??""}" style="font-size:13px">${esc(String(v))}</div></div>`).join("");
  $("cl-body").innerHTML=(res.symbols||[]).map(s=>{
    const c=s.current||{};
    return `<tr><td><b>${esc(s.symbol)}</b></td><td class="r">${s.prints}</td>
      <td class="r">${c.entries}</td><td class="r">${c.wins}</td>
      <td class="r ${c.funding_ret>=0?'pnl-pos':'pnl-neg'}">${fmt.signed(c.funding_ret*100,2)}%</td>
      <td class="r ${pnlCls(c.price_ret)}">${fmt.signed(c.price_ret*100,2)}%</td>
      <td class="r">${(c.fees*100).toFixed(2)}%</td>
      <td class="r ${pnlCls(c.net)}"><b>${fmt.signed(c.net*100,2)}%</b></td>
      <td class="r pnl-neg">${fmt.signed(c.worst*100,1)}%</td>
      <td class="r">${c.avg_hold_h}h</td></tr>`;
  }).join("");
  if(rec) toast(`Carry lab: evidence says enter ≥${(rec.min_apr*100).toFixed(0)}% APR (net ${fmt.signed(rec.net*100,2)}%)`,rec.net>0?"good":"warn");
}

/* ---------------------------------------------------------------- record */
let recordChart=null, recordSeries=null, recordRows=[];
function ensureRecordChart(){
  if(recordChart) return;
  recordChart=LightweightCharts.createChart($("chart-record"),baseOpts(200));
  recordSeries=recordChart.addAreaSeries({lineColor:C.accent,lineWidth:2,topColor:"rgba(0,210,255,0.22)",bottomColor:"rgba(0,210,255,0.02)",priceLineVisible:false});
  new ResizeObserver(()=>{ recordChart.applyOptions({width:$("chart-record").clientWidth}); }).observe($("chart-record"));
}
async function loadRecord(){
  try{ const d=await api("/api/record"); renderRecord(d); }catch(e){ toast(e.message,"bad"); }
}
function renderRecord(d){
  ensureRecordChart();
  const rows=(d.rows||[]); recordRows=rows;
  const today=d.today;
  const all=today?rows.concat([{...today,partial:true}]):rows;
  const wins=rows.filter(r=>r.pnl>0).length;
  const tot=rows.reduce((a,r)=>a+r.pnl,0);
  const best=rows.length?Math.max(...rows.map(r=>r.pnl)):0;
  const worst=rows.length?Math.min(...rows.map(r=>r.pnl)):0;
  $("rec-cards").innerHTML=[
    ["Days recorded",rows.length],
    ["Total PnL",fmt.signed(tot,2),pnlCls(tot)],
    ["Win days",rows.length?`${wins}/${rows.length}`:"—"],
    ["Best day",fmt.signed(best,2),"pnl-pos"],["Worst day",fmt.signed(worst,2),"pnl-neg"],
    ["Today (partial)",today?fmt.signed(today.pnl,2):"—",pnlCls(today?.pnl??0)],
  ].map(([k,v,cls])=>`<div class="card"><div class="k">${k}</div><div class="v ${cls??""}">${v}</div></div>`).join("");
  requestAnimationFrame(()=>{
    recordChart.applyOptions({width:$("chart-record").clientWidth});
    recordSeries.setData(all.map(r=>({time:r.d,value:r.equity})));
    recordChart.timeScale().fitContent();
  });
  const months={};
  for(const r of rows){ const m=r.d.slice(0,7);
    const g=months[m]=months[m]||{pnl:0,trades:0,windays:0,days:0,eq0:null,eq1:0};
    if(g.eq0==null) g.eq0=r.equity-r.pnl;
    g.eq1=r.equity; g.pnl+=r.pnl; g.trades+=r.trades; g.days++; if(r.pnl>0) g.windays++; }
  const mk=Object.keys(months).sort().reverse();
  $("rec-months").innerHTML=mk.length?mk.map(m=>{ const g=months[m];
    const ret=g.eq0>0?(g.eq1/g.eq0-1)*100:0;
    return `<tr><td>${m}</td><td class="r ${pnlCls(g.pnl)}">${fmt.signed(g.pnl,2)}</td>
      <td class="r ${pnlCls(ret)}">${fmt.signed(ret,2)}%</td><td class="r">${g.trades}</td>
      <td class="r">${g.windays}/${g.days}</td></tr>`; }).join("")
    :`<tr><td colspan="5" class="empty">No complete months yet</td></tr>`;
  $("rec-body").innerHTML=all.length?all.slice().reverse().map(r=>`<tr${r.partial?' style="color:var(--accent-2)"':''}>
    <td>${r.d}${r.partial?" (today)":""}</td><td>${esc(r.mode||"")}</td>
    <td class="r">${fmt.usd(r.equity)}</td><td class="r ${pnlCls(r.pnl)}">${fmt.signed(r.pnl,2)}</td>
    <td class="r">${r.trades}</td><td class="r">${r.wins??0}</td><td class="r">${(r.fees??0).toFixed(2)}</td></tr>`).join("")
    :`<tr><td colspan="7" class="empty">The first row appears after the first UTC midnight of running</td></tr>`;
}
$("rec-export").onclick=()=>{
  const head="date,mode,equity,pnl,trades,wins,fees";
  const csv=[head,...recordRows.map(r=>[r.d,r.mode,r.equity,r.pnl,r.trades,r.wins??0,r.fees??0].join(","))].join("\n");
  const a=document.createElement("a");
  a.href=URL.createObjectURL(new Blob([csv],{type:"text/csv"}));
  a.download="track_record.csv"; a.click(); URL.revokeObjectURL(a.href);
};

/* ---------------------------------------------------------------- analytics */
async function loadAnalytics(){
  try{ const mode=$("an-mode").value; const d=await api(`/api/journal${mode?`?mode=${mode}`:""}`); renderAnalytics(d); }
  catch(e){ toast(e.message,"bad"); }
}
$("an-refresh").onclick=loadAnalytics; $("an-mode").onchange=loadAnalytics;
function anRows(obj){ const e=Object.entries(obj||{}); if(!e.length) return `<tr><td colspan="4" class="empty">no data</td></tr>`;
  return e.sort((a,b)=>b[1].n-a[1].n).map(([k,v])=>`<tr><td>${esc(k)}</td><td class="r">${v.n}</td>
    <td class="r ${v.win_rate>=0.5?'pnl-pos':''}">${fmt.pct(v.win_rate,0)}</td>
    <td class="r ${pnlCls(v.pnl)}">${fmt.signed(v.pnl,2)}</td></tr>`).join(""); }
function renderAnalytics(d){
  const s=d.summary||{trades:0};
  const div=S?.divergence; let dt="";
  if(div){ if(div.status==="gathering") dt=` · divergence: gathering (${div.live_trades} live trades)`;
    else dt=` · live WR ${fmt.pct(div.live_win_rate,0)}${div.expected_win_rate!=null?` vs backtest ${fmt.pct(div.expected_win_rate,0)}`:""} ${div.diverged?"⚠ DIVERGED":"✓ on track"}`; }
  $("an-count").innerHTML=`${s.trades||0} journaled trades<span style="color:${div?.diverged?'var(--bad)':'var(--muted)'}">${esc(dt)}</span>`
    +(S?.alerts_on?` · <span style="color:var(--good)">alerts on</span>`:"");
  $("an-cards").innerHTML=!s.trades?`<div class="empty" style="grid-column:1/-1">No journaled trades yet — they accrue as paper/live trades close.</div>`:[
    ["Win rate",fmt.pct(s.win_rate),s.win_rate>=0.5?"pnl-pos":""],
    ["Profit factor",(s.profit_factor||0).toFixed(2),s.profit_factor>=1?"pnl-pos":"pnl-neg"],
    ["Trades",s.trades],["Net PnL",fmt.signed(s.pnl,2),pnlCls(s.pnl)],
  ].map(([k,v,cls])=>`<div class="card"><div class="k">${k}</div><div class="v ${cls??""}">${v}</div></div>`).join("");
  $("an-align").innerHTML=anRows(s.by_alignment); $("an-regime").innerHTML=anRows(s.by_regime);
  $("an-desk").innerHTML=anRows(s.by_desk); $("an-exit").innerHTML=anRows(s.by_exit);
  $("an-hour").innerHTML=anRows(s.by_hour); $("an-side").innerHTML=anRows(s.by_side);
}

initCharts(); connectWS();
// slow reconciliation only (closed bars + markers) — the live candle rides the
// 0.4s hot channel now; skip entirely while the tab is hidden.
setInterval(()=>{ if(S?.engine&&!document.hidden) refreshCandles(false); },10000);
