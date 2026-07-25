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
  const cm=$("chart-main"), ce=$("chart-equity");
  mainChart=LightweightCharts.createChart(cm,baseOpts(cm.clientHeight||384));
  candleSeries=mainChart.addCandlestickSeries({upColor:C.up,downColor:C.dn,borderUpColor:C.up,borderDownColor:C.dn,wickUpColor:C.up,wickDownColor:C.dn});
  equityChart=LightweightCharts.createChart(ce,{...baseOpts(ce.clientHeight||118),
    rightPriceScale:{borderColor:C.baseline,scaleMargins:{top:0.15,bottom:0.1}},timeScale:{visible:false},handleScroll:false,handleScale:false});
  equitySeries=equityChart.addAreaSeries({lineColor:C.accent,lineWidth:2,topColor:"rgba(0,210,255,0.22)",bottomColor:"rgba(0,210,255,0.02)",priceLineVisible:false});
  // width AND height track the container — the responsive layout resizes both
  new ResizeObserver(()=>{
    mainChart.applyOptions({width:cm.clientWidth,height:cm.clientHeight||384});
    equityChart.applyOptions({width:ce.clientWidth,height:ce.clientHeight||118});
  }).observe(cm);
}
function tradeMarkers(markers){ return markers.slice().sort((a,b)=>a.ts-b.ts).map(m=>m.kind==="entry"?
  {time:Math.floor(m.ts/1000),position:m.side==="LONG"?"belowBar":"aboveBar",color:m.side==="LONG"?C.up:C.dn,shape:m.side==="LONG"?"arrowUp":"arrowDown",text:m.side==="LONG"?"L":"S"}:
  {time:Math.floor(m.ts/1000),position:"inBar",color:(m.pnl??0)>=0?C.up:C.dn,shape:"circle"}); }

/* ---------------------------------------------------------------- state */
let S=null, curSymbol=null, lastTradeCount=-1;
const symbols=()=>S?.config?.symbols??[];
const engSym=()=>S?.engine?.symbols?.[curSymbol];

/* one rAF paints per frame no matter how many ws messages arrived — the page
   stops re-laying-out several times a second for identical pixels. */
let _dirtyFull=false,_dirtyHot=false,_rafQueued=false;
function scheduleRender(kind){
  if(kind==="full") _dirtyFull=true; else _dirtyHot=true;
  if(_rafQueued) return; _rafQueued=true;
  requestAnimationFrame(()=>{ _rafQueued=false;
    const full=_dirtyFull; _dirtyFull=_dirtyHot=false;
    if(full) renderAll(); else renderHot(); });
}

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
  const healthy=!!S.engine?.feed_healthy, stale=!!S.engine?.bar_stale;
  $("feed-dot").className="dot"+(stale?" stale":healthy?" ok":"");
  // a red STALE chip beats a silently frozen terminal: prices can look live
  // while the kline stream is dead and the brain starves for bar closes.
  $("feed-label").textContent=stale?`STALE ${S.engine?.bar_age_s??"?"}s`
    :S.engine?(S.config.feed==="synthetic"?"synthetic":"BingX"):"no feed";
  $("feed-label").className=stale?"stale":"";
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
/* Auto-follow with HYSTERESIS. focus_symbol is "whichever symbol is closest to
   firing", which can flip between two near-threshold symbols on consecutive
   0.4s pushes — and every flip used to reload 500 candles + all markers into
   the chart, the most expensive single operation in the UI. A new focus must
   now hold for FOCUS_HOLD_MS before the chart follows it. */
const FOCUS_HOLD_MS=4000;
let _focusCand=null,_focusSince=0;
function followFocus(){
  if(!autoFollow) return;
  const f=S?.engine?.focus;
  if(!f||!S?.engine?.symbols?.[f]||f===curSymbol){ _focusCand=null; return; }
  const now=performance.now();
  if(_focusCand!==f){ _focusCand=f; _focusSince=now; return; }
  if(now-_focusSince>=FOCUS_HOLD_MS){ _focusCand=null; setSymbol(f); }
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
let _tapeSig=null;
function renderTape(){
  const tape=S.engine?.tape??[]; const track=$("tape-track");
  // rebuild ONLY when a fill actually arrived — re-setting innerHTML restarts
  // the marquee animation, which made the ticker stutter 4x a second.
  const sig=tape.length+"|"+JSON.stringify(tape[tape.length-1]??0);
  if(sig===_tapeSig) return; _tapeSig=sig;
  if(!tape.length){ track.innerHTML=`<span class="tape-item" style="color:var(--muted)">awaiting fills…</span>`; return; }
  const items=tape.slice().reverse().map(t=>{
    const tag=t.kind==="OPEN"?`<span class="tag open">OPEN</span>`:`<span class="tag close">CLOSE</span>`;
    const px=fmt.px(t.price); const extra=t.kind==="OPEN"?`P${Math.round((t.p_win||0)*100)}%`:
      `<span class="${pnlCls(t.pnl)}">${fmt.signed(t.pnl,2)}</span>`;
    return `<span class="tape-item">${tag} <b>${esc(t.symbol.replace("-USDT",""))}</b> <span class="${t.side==='LONG'?'up':'dn'}">${t.side}</span> ${px} ${extra}</span>`;
  }).join("");
  track.innerHTML=items+items;   // duplicate for seamless marquee
}
let _pipeSig=null;
function renderPipeline(){
  const es=engSym(); const stages=S.engine?.stages??["SCAN","DETECT","VALIDATE","SIZE","FILL","MANAGE","SETTLE"];
  const cur=es?.stage||"SCAN"; const ci=stages.indexOf(cur);
  const sig=curSymbol+"|"+cur;
  if(sig===_pipeSig) return; _pipeSig=sig;
  $("pipe").innerHTML=stages.map((s,i)=>{
    const cls=i===ci?"on":(ci>=0&&i<ci?"done":"");
    return `<div class="pstage ${cls}"><div class="n">${String(i+1).padStart(2,"0")}</div><div class="l">${s}</div></div>`;
  }).join("");
}
/* gates + MTF ladder: the structure is built ONCE per shape, then only text,
   classes and bar widths are patched in place — the old innerHTML rebuilds
   re-parsed and re-laid-out these widgets several times a second. */
let _gateStruct=null,_gateRows=null;
function renderGates(es){
  const el=$("gate-list"); if(!el) return;
  const held=!!S?.engine?.portfolio?.open_positions?.[curSymbol];
  const g=es?.gates||[];
  const struct=curSymbol+"|"+(held?"H":"")+"|"+g.map(x=>x.n).join(",");
  if(struct!==_gateStruct){
    _gateStruct=struct; _gateRows=null;
    if(held){ el.innerHTML=`<span class="mtf-empty">in position — gates re-arm on exit</span>`; return; }
    if(!g.length){ el.innerHTML=`<span class="mtf-empty">warming up…</span>`; return; }
    el.innerHTML=g.map(x=>`<div class="gate"><span class="gd"></span><span class="gn">${esc(x.n)}</span><span class="gv"></span></div>`).join("");
    _gateRows=[...el.querySelectorAll(".gate")];
  }
  if(!_gateRows) return;
  for(let i=0;i<g.length&&i<_gateRows.length;i++){
    const r=_gateRows[i], x=g[i];
    const cls="gate "+(x.ok?"pass":"fail");
    if(r.className!==cls) r.className=cls;
    const glyph=x.ok?"▮":"▯";
    if(r.firstChild.textContent!==glyph) r.firstChild.textContent=glyph;
    if(r.lastChild.textContent!==x.d){ r.lastChild.textContent=x.d; r.title=x.d; }
  }
}
let _mtfStruct=null,_mtfCells=null;
function renderMTF(es){
  const strip=$("mtf-strip"); if(!strip) return;
  const mtf=es?.mtf||{};
  const order=["1m","5m","15m","1h"].filter(tf=>mtf[tf]);
  const struct=curSymbol+"|"+order.join(",");
  if(struct!==_mtfStruct){
    _mtfStruct=struct; _mtfCells=null;
    if(!order.length){ strip.innerHTML=`<span class="mtf-empty">warming up…</span>`; return; }
    strip.innerHTML=order.map(tf=>`<div class="mtf-cell"><div class="tf">${tf}</div>
      <div class="dir"></div><div class="tfbar"><div class="tffill"></div></div><div class="tfrsi"></div></div>`).join("");
    _mtfCells=[...strip.querySelectorAll(".mtf-cell")];
  }
  if(!_mtfCells) return;
  order.forEach((tf,i)=>{
    const c=_mtfCells[i]; if(!c) return;
    const m=mtf[tf], d=m.dir||0;
    const cls="mtf-cell "+(d>0.15?"up":d<-0.15?"dn":"flat");
    if(c.className!==cls) c.className=cls;
    const arrow=d>0.15?"▲":d<-0.15?"▼":"▬";
    if(c.children[1].textContent!==arrow) c.children[1].textContent=arrow;
    c.children[2].firstChild.style.width=Math.round(Math.abs(clamp(d,-1,1))*100)+"%";
    const rsi="RSI "+Math.round(m.rsi);
    if(c.children[3].textContent!==rsi) c.children[3].textContent=rsi;
  });
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
  const r24=(es.hi24&&es.lo24)?` · 24h ${fmt.px(es.lo24)}–${fmt.px(es.hi24)} (pos ${Math.round((es.rpos24||0)*100)}%)`:"";
  $("px-meta").textContent=`1m chart · ${tf} signals${r24} · spread ${micro.spread_bps.toFixed(1)}bp · OBI ${fmt.signed(micro.obi,2)} · flow ${fmt.signed(micro.flow,2)}${fund}`;
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
  $("b-overlay").style.display=es.overlay?"":"none";

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
  renderConstellation(alphas);
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
let _eqCurveLen=-1;
function renderEquity(){
  const curve=S.engine?.equity_curve??[];
  if(curve.length>1){
    // setData of a multi-thousand-point series every full push was a major
    // frame killer; the live tip already rides equitySeries.update() on the
    // hot channel, so a full reload only matters when history itself changed.
    if(curve.length!==_eqCurveLen){
      _eqCurveLen=curve.length;
      equitySeries.setData(curve.map(([ts,eq])=>({time:Math.floor(ts/1000),value:eq})));
    }
    const eq=curve[curve.length-1][1], start=S.engine.portfolio.starting_balance, dlt=eq-start;
    $("eq-cap").textContent=`${fmt.usd(eq)}  (${fmt.signed(dlt,2)} / ${fmt.signed(dlt/start*100,2)}%)`;
    $("eq-cap").className="val "+pnlCls(dlt);
  }
}
let _posSig=null;
function renderPositions(){
  const pf=S.engine?.portfolio, body=$("pos-body");
  const entries=pf?Object.entries(pf.open_positions):[];
  // structure (which positions, side, qty, stop) rebuilds rows; the fast-moving
  // mark/uPnL cells are PATCHED in place — no 4 Hz innerHTML churn.
  const sig=entries.map(([s,p])=>`${s}${p.side}${p.qty}${p.stop}`).join("~");
  if(sig!==_posSig){
    _posSig=sig;
    if(!entries.length){ body.innerHTML=`<tr><td colspan="11" class="empty">No open positions</td></tr>`; return; }
    body.innerHTML=entries.map(([sym,p])=>{ const mark=S.engine.symbols[sym]?.price??0;
      return `<tr><td>${esc(sym)}</td><td class="${sideCls(p.side)}">${p.side}</td><td class="r">${p.qty}</td>
        <td class="r">${fmt.px(p.entry)}</td><td class="r" data-mark="${esc(sym)}">${fmt.px(mark)}</td><td class="r">${fmt.px(p.stop)}</td>
        <td class="r">${fmt.px(p.tp)}</td><td class="r ${pnlCls(p.upnl)}" data-upnl="${esc(sym)}">${fmt.signed(p.upnl,2)}</td>
        <td class="r">${p.leverage}x</td><td>${fmt.time(p.opened_ts)}</td>
        <td><button class="btn sm" onclick="closePos('${esc(sym)}')">Close</button></td></tr>`; }).join("");
    return;
  }
  for(const [sym,p] of entries){
    const mc=body.querySelector(`[data-mark="${CSS.escape(sym)}"]`);
    if(mc) mc.textContent=fmt.px(S.engine.symbols[sym]?.price??0);
    const uc=body.querySelector(`[data-upnl="${CSS.escape(sym)}"]`);
    if(uc){ uc.textContent=fmt.signed(p.upnl,2); uc.className="r "+pnlCls(p.upnl); }
  }
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
    const uni=(R.top_volume||[]).map(s=>s.replace("-USDT","")).join(" ");
    const uv=R.universe; const uvTxt=uv?` · eligibility: ${uv.source} (${uv.count} tokens${uv.age_min!=null?`, ${uv.age_min<90?uv.age_min+"m":Math.round(uv.age_min/60)+"h"} old`:""})`:"";
    $("radar-meta").textContent=R.ts?`· scan ${fmt.time(R.ts)}${R.demo?" · DEMO BOARD (synthetic feed)":""}${uvTxt}${uni?` · tuner universe: ${uni}`:""}${R.error?` · ⚠ ${R.error}`:""}`:"";
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
    ["Research duty",at.duty!=null?`${Math.round(at.duty*100)}%`:"—"],
    ["Researching",at.research_symbol||"—"],
    ["Cycles run",at.cycles],
    ["Improvements",at.improvements],
    ["Champion fitness",at.champion_fitness??"—"],
    ["Promotion bar",lc?.bar!=null?lc.bar:"—"],
    ["PF gate",lc&&lc.cands_judged!=null?`${lc.pf_passed??0}/${lc.cands_judged} passed${lc.thin_rejected?` · ${lc.thin_rejected} too thin`:""}`:"—"],
    ["Co-trained on",lc?.co_symbol??"—"],
    ["Last challenger",lc?(lc.best_fitness==null?"none passed profit gate":`${lc.best_fitness} (${lc.promoted?"adopted":"kept"})`):"—"],
    ["Cycle clock",lc?.clock??"—"],
    ["Trial clock",at.clock_trial?(at.last_trial?`${at.last_trial.clock} · best ${at.last_trial.best_fitness??"—"} · gen ${at.last_trial.generation??0}`:"warming up…"):"off"],
    ["Shadow race",S.shadow?(S.shadow.equity!=null
      ?`${S.shadow.clock}: ${fmt.usd(S.shadow.equity)} · ${S.shadow.stats?.trades??0} trades · PF ${S.shadow.stats?.trades?(S.shadow.stats.profit_factor??0).toFixed(2):"—"}`
      :S.shadow.status||"waiting"):"—"],
    ["Gauntlet",lc?.gauntlet?`med ${lc.gauntlet.median} · ${lc.gauntlet.pf_ge1}/${lc.gauntlet.n} eras PF≥1${lc.gauntlet.weak?" ⚠ weak":""}`:"—"],
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
  ["entry_pullback_atr","pullback entry ×ATR","s"],
  ["min_efficiency","min trend efficiency","s"],["hedge_eta","hedge learn rate","s"],["horizon_bars","grade horizon","s"],
  ["risk_per_trade","risk per trade","r"],["sl_atr_min","stop min ×ATR","r"],["sl_atr_max","stop max ×ATR","r"],
  ["trail_atr_min","trail min ×ATR","r"],["trail_atr_max","trail max ×ATR","r"],["trail_tighten","trail tighten","r"],
  ["be_rr","breakeven R","r"],["giveback_rr","giveback R","r"],["hold_edge_frac","edge-flip exit","r"],["time_stop_bars","time stop bars","r"],
];
function renderSettings(){
  if(settingsDirty||!S) return; const c=S.config;
  $("cfg-symbols").value=c.symbols.join(", "); $("cfg-feed").value=c.feed; $("cfg-interval").value=c.strategy.interval;
  $("cfg-radarextra").value=(c.radar_extra||[]).join(", ");
  $("cfg-balance").value=c.paper.starting_balance; $("cfg-maxpos").value=c.risk.max_open_positions;
  $("cfg-levmin").value=c.risk.min_leverage; $("cfg-levmax").value=c.risk.max_leverage;
  $("cfg-dayloss").value=c.risk.max_daily_loss_pct; $("cfg-hardrisk").value=c.risk.max_risk_hard_pct;
  $("cfg-autotune").checked=c.strategy.auto_tune; $("cfg-allowlive").checked=c.allow_live;
  $("cfg-adopt").value=c.strategy.adopt_symbols??2;
  $("cfg-clocktrial").checked=!!c.strategy.clock_trial; $("cfg-trialint").value=c.strategy.trial_interval||"5m";
  if(c.tape) $("cfg-tape").checked=!!c.tape.enabled;
  $("cfg-makerexit").checked=c.strategy.maker_exits!==false;
  const duty=c.strategy.research_duty??0.28;
  $("cfg-duty").value=duty; $("cfg-duty-val").textContent=`${Math.round(duty*100)}%`;
  if(c.carry){ $("cfg-carry").checked=c.carry.enabled; $("cfg-carrymax").value=c.carry.max_positions; }
  $("cfg-keys").textContent=c.has_keys?"configured ✓":"not set (paper/backtest only)"; $("cfg-keys").style.color=c.has_keys?"var(--good)":"";
  $("auto-params").innerHTML=AUTO_PARAMS.map(([k,lab,grp])=>{
    const v=(grp==="s"?c.strategy:c.risk)[k];
    const val=typeof v==="number"?(Math.abs(v)<1?v.toFixed(3):v.toFixed(2)):v;
    return `<div style="display:flex;justify-content:space-between"><span style="color:var(--muted)">${lab}</span><span style="color:var(--ink)">${val}</span></div>`;
  }).join("");
}
/* only the VISIBLE bottom tab gets rendered — rebuilding five hidden tables
   (trades, vault, tuner history, radar, settings) on every full push was a
   layout+GC burst the user could FEEL as a periodic stutter. Hidden tabs
   render from state the moment they're opened. */
const activePage=()=>document.querySelector(".tab-page.active")?.dataset.page;
function renderBottomTab(page){
  if(!S) return;
  if(page==="positions"){ if(S.engine) renderPositions(); }
  else if(page==="trades"){ if(S.engine) renderTrades(); }
  else if(page==="progress"){ renderProgress(); }
  else if(page==="autotuner"){ renderAutotuner(); renderChampions(); }
  else if(page==="radar"){ renderRadar(); }
  else if(page==="settings"){ renderSettings(); }
}
function renderAll(){
  if(!S) return;
  renderTop(); renderRank(); pollAchievements(); renderSymTabs(); renderTape();
  if(S.engine){ renderPipeline(); renderBrain(); renderEquity();
    cortexFeed();
    const tc=S.engine.portfolio.stats.trades; refreshCandles(false).then(()=>{lastTradeCount=tc;}); }
  renderLiveViews();
  renderBottomTab(activePage());
}

/* ------- fast 'hot' channel: patch the live numbers between full pushes ----- */
function applyHot(h){
  if(!S||!S.engine||!h?.engine) return;
  const he=h.engine; S.mode=h.mode??S.mode;
  const pf=S.engine.portfolio; if(pf) pf.equity=he.equity;
  if(typeof he.killed==="boolean"&&S.engine.risk) S.engine.risk.killed=he.killed;
  if(typeof he.feed_healthy==="boolean") S.engine.feed_healthy=he.feed_healthy;
  if(typeof he.bar_stale==="boolean"){ S.engine.bar_stale=he.bar_stale; S.engine.bar_age_s=he.bar_age_s; }
  if(he.focus) S.engine.focus=he.focus;
  if(he.adopted) S.engine.adopted=he.adopted;
  for(const [sym,hs] of Object.entries(he.symbols||{})){
    const s=S.engine.symbols?.[sym]; if(!s) continue;
    s.price=hs.price; s.stage=hs.stage; s.eval_ms=hs.eval_ms; s.entry_block=hs.entry_block; s.bars_held=hs.bars_held;
    if(hs.hi24){ s.hi24=hs.hi24; s.lo24=hs.lo24; s.rpos24=hs.rpos24; }
    if(hs.micro){ s.micro=hs.micro; flowPush(sym,hs.micro); }
    if(hs.mtf) s.mtf=hs.mtf;
    if(hs.gates) s.gates=hs.gates;
    if(hs.candle) s.candle=hs.candle;
    if(hs.viz) s.viz=hs.viz;
    if(s.brain){ s.brain.edge=hs.edge; s.brain.p_win=hs.p_win; s.brain.regime=hs.regime; }
  }
  for(const [sym,hp] of Object.entries(he.positions||{})){
    const p=pf?.open_positions?.[sym]; if(p) p.upnl=hp.upnl;
  }
  if(he.tape) S.engine.tape=he.tape;
  cortexFeed();               // animation targets update immediately, off-paint
  scheduleRender("hot");
}
function cortexFeed(){
  const es=engSym(); if(!es?.viz) return;
  cortex.data(es.viz,{edge:es.brain?.edge??0, p_win:es.brain?.p_win??0.5,
    regime:es.brain?.regime||"RANGE", sym:curSymbol, price:es.price,
    stage:es.stage, block:es.entry_block,
    held:!!S?.engine?.portfolio?.open_positions?.[curSymbol],
    ticks:es.eval_ms??0});
}
let lastEqT=0;
function renderHot(){
  if(!S?.engine) return;
  renderTop(); renderPipeline(); renderTape();
  if(activePage()==="positions") renderPositions();
  renderSymTabs();   // adopted set + auto-follow react at hot cadence
  const es=engSym(); if(es&&es.brain){ renderEdgeGauge(es.brain, es); renderMTF(es); renderGates(es); }
  renderLiveViews();
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
  // One unparseable frame must cost us that frame, not the whole dashboard.
  // An exception thrown out of onmessage aborts the handler, and since every
  // update arrives this way the UI then just stops — looking connected, with
  // nothing in view to say why. The server no longer emits NaN (which is not
  // valid JSON and is what used to do this), but a truncated frame should be
  // survivable regardless.
  ws.onmessage=(ev)=>{
    let m; try{ m=JSON.parse(ev.data); }
    catch(err){ console.warn("dropped an unparseable frame:",err.message); return; }
    if(m.type==="state"){ S=m.data; scheduleRender("full"); }
    else if(m.type==="hot"){ applyHot(m.data); } };
  ws.onopen=()=>{wsRetry=1;}; ws.onclose=()=>setTimeout(connectWS,Math.min(wsRetry*=1.6,8)*1000); ws.onerror=()=>ws.close();
}

/* ---------------------------------------------------------------- actions */
window.closePos=async(sym)=>{ try{ await api("/api/control",{action:"close",symbol:sym}); toast(`${sym} closed`,"good"); }catch(e){ toast(e.message,"bad"); } };
$("btn-kill").onclick=async()=>{ if(!confirm("Kill switch: flatten all and halt entries?")) return;
  try{ await api("/api/control",{action:"kill"}); toast("Kill switch engaged","warn"); }catch(e){ toast(e.message,"bad"); } };
$("btn-flatten").onclick=async()=>{ try{ const r=await api("/api/control",{action:"flatten"}); toast(r.message,"good"); }catch(e){ toast(e.message,"bad"); } };
$("btn-reset-kill").onclick=async()=>{ try{ const r=await api("/api/control",{action:"reset_kill"}); toast(r.message,"good"); }catch(e){ toast(e.message,"bad"); } };
$("btn-report").onclick=async()=>{
  try{
    const res=await fetch("/api/report"); if(!res.ok) throw new Error(`HTTP ${res.status}`);
    const blob=await res.blob(); const a=document.createElement("a");
    a.href=URL.createObjectURL(blob);
    a.download=`pulse_resume_${new Date().toISOString().slice(0,16).replace(/[-T:]/g,"")}.txt`;
    a.click(); URL.revokeObjectURL(a.href);
    toast("Resume downloaded — share the .txt for analysis","good");
  }catch(e){ toast(`Report failed: ${e.message}`,"bad"); }
};
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
  renderBottomTab(b.dataset.tab);   // hidden tabs render on open, not on every push
  if(b.dataset.tab==="backtest") ensureBtCharts();
  if(b.dataset.tab==="portfolio") ensurePfChart();
  if(b.dataset.tab==="walkforward") ensureWfChart();
  if(b.dataset.tab==="analytics") loadAnalytics();
  if(b.dataset.tab==="record") loadRecord();
}; });
document.querySelectorAll('[data-page="settings"] input, [data-page="settings"] select').forEach(el=>el.addEventListener("input",()=>{settingsDirty=true;}));
$("cfg-duty").addEventListener("input",()=>{ $("cfg-duty-val").textContent=`${Math.round($("cfg-duty").value*100)}%`; });
$("cfg-save").onclick=async()=>{
  const patch={ symbols:$("cfg-symbols").value.split(",").map(s=>s.trim().toUpperCase()).filter(Boolean),
    radar_extra:$("cfg-radarextra").value.split(",").map(s=>s.trim().toUpperCase()).filter(Boolean),
    feed:$("cfg-feed").value, allow_live:$("cfg-allowlive").checked,
    strategy:{ interval:$("cfg-interval").value, auto_tune:$("cfg-autotune").checked,
      adopt_symbols:parseInt($("cfg-adopt").value,10),
      clock_trial:$("cfg-clocktrial").checked, trial_interval:$("cfg-trialint").value,
      research_duty:parseFloat($("cfg-duty").value),
      maker_exits:$("cfg-makerexit").checked },
    tape:{ enabled:$("cfg-tape").checked },
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
    const oldScale=(c.fver??1)!==4;   // born under an older fitness scale — numbers not comparable
                                      // (v4 = evidence-shrunk; v3 = honest fills; v2 and earlier
                                      //  were the flattering fill model)
    const arrow=oldScale?"·":(cf>bf+1e-9?"▲":(cf<bf-1e-9?"▼":"·"));
    const bfCell=oldScale?`<span style="color:var(--muted);opacity:.5" title="recorded under an older fitness scale — not comparable to the current number">${bf.toFixed(2)}*</span>`
                         :`<span style="color:var(--muted)">${bf.toFixed(2)}</span>`;
    const lv=c.live||{trades:0,pnl:0};
    const liveCell=lv.trades?`${lv.trades} · <span class="${pnlCls(lv.pnl)}">${fmt.signed(lv.pnl,2)}</span>`
                            :`<span style="color:var(--muted)">—</span>`;
    const g=c.gauntlet;
    const badges=(c.active?`<span class="champ-live" title="Currently driving live trading">LIVE</span> `:"")
                +(c.clock?`<span class="clock-chip" title="Bar clock this set was validated on — only same-clock champions are candidates for the live engine">${esc(c.clock)}</span> `:"")
                +(g?`<span title="Regime gauntlet (Binance archive, BingX fees): median fitness ${g.median} across ${g.n} historical eras, ${g.pf_ge1} with PF≥1${g.weak?" — WEAK: live probation doubled":""}">${g.weak?"🧪⚠":"🧪"}</span> `:"")
                +(c.top_used?`<span title="Top-10 most used — protected from pruning">🔥</span> `:"")
                +(c.live_flag?`<span title="Demoted on LIVE evidence: real PF ${(c.live_flag.pf??0).toFixed(2)} over ${c.live_flag.trades??0} trades — excluded as a candidate for 48h">⚠</span> `:"");
    return `<tr class="${c.active?'champ-active':''}">
      <td>${badges}${fmt.dt(c.born_ts)}</td>
      <td class="r">${bfCell} ${arrow} <span class="${cf>=0?'pnl-pos':'pnl-neg'}">${cf.toFixed(2)}</span></td>
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
  try{ const mode=$("an-mode").value; const d=await api(`/api/journal?limit=1200${mode?`&mode=${mode}`:""}`); renderAnalytics(d); }
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
    ["Avg MAE (heat)",(s.avg_mae_r??0).toFixed(2)+"R","pnl-neg"],
    ["Avg MFE (shown)",(s.avg_mfe_r??0).toFixed(2)+"R","pnl-pos"],
    ["MFE captured",fmt.pct(s.mfe_capture??0,0),(s.mfe_capture??0)>=0.4?"pnl-pos":""],
  ].map(([k,v,cls])=>`<div class="card"><div class="k">${k}</div><div class="v ${cls??""}">${v}</div></div>`).join("");
  // the same payload the tables use, drawn
  const rows=d.recent||[];
  _exDist = s.r_dist;
  renderExcursion(rows); renderRDist(rows, s.r_dist); renderGhostBook(S?.engine?.refusals);
  $("an-align").innerHTML=anRows(s.by_alignment); $("an-regime").innerHTML=anRows(s.by_regime);
  $("an-desk").innerHTML=anRows(s.by_desk); $("an-exit").innerHTML=anRows(s.by_exit);
  $("an-hour").innerHTML=anRows(s.by_hour); $("an-side").innerHTML=anRows(s.by_side);
}

/* ==================== OPERATOR PROGRESSION ====================================
   A game layer over REAL work — nothing here is invented or cosmetic-only:
   every point, badge and mission reads state the engine already publishes
   (graded predictions, closed trades, vault champions, tuner generations,
   gauntlet verdicts, the shadow race). The point is to make "is this thing
   actually getting better?" legible at a glance. */
const RANKS=[[0,"ROOKIE"],[500,"ANALYST"],[1500,"TRADER"],[3500,"SPECIALIST"],
             [7000,"STRATEGIST"],[13000,"PORTFOLIO MGR"],[22000,"QUANT"],
             [36000,"MARKET MAKER"],[60000,"ALPHA"],[95000,"LEGEND"]];
function xpBreakdown(){
  const eng=S?.engine, at=S?.autotuner, st=eng?.portfolio?.stats;
  let graded=0; for(const s of Object.values(eng?.symbols||{})) graded+=s?.brain?.graded||0;
  const trades=st?.trades||0;
  const wins=Math.round((st?.win_rate||0)*trades);
  const champs=(S?.champions||[]).length;
  const gens=at?.generation||0;
  // weights: learning is the grind, a WIN is the payoff, a champion is a
  // milestone. Deliberately not PnL-scaled — a $100 account shouldn't rank
  // lower than the same machine on $10k for identical work.
  return [["graded predictions",graded,2],["closed trades",trades,25],
          ["winning trades",wins,60],["vault champions",champs,180],
          ["DE generations",gens,4]];
}
function operator(){
  const parts=xpBreakdown();
  const xp=parts.reduce((a,[,n,w])=>a+n*w,0);
  let i=0; while(i+1<RANKS.length&&xp>=RANKS[i+1][0]) i++;
  const base=RANKS[i][0], next=i+1<RANKS.length?RANKS[i+1][0]:null;
  return {xp,lvl:i+1,title:RANKS[i][1],base,next,parts,
          pct:next?clamp((xp-base)/(next-base),0,1):1};
}
function streaks(){
  const tr=S?.engine?.trades||[];
  let cur=0,best=0,run=0;
  for(const t of tr){ if(t.pnl>0){ run++; best=Math.max(best,run); } else run=0; }
  cur=run;
  return {cur,best};
}
/* achievements: [id, icon, name, unlocked?, hint] */
function achievements(){
  const eng=S?.engine, st=eng?.portfolio?.stats, at=S?.autotuner;
  const ch=S?.champions||[], sk=streaks();
  let graded=0; for(const s of Object.values(eng?.symbols||{})) graded+=s?.brain?.graded||0;
  const meta=at?.meta?.model, up=(Date.now()-(eng?.started_ts||Date.now()))/3600000;
  const gaunt=ch.find(c=>c.gauntlet&&!c.gauntlet_weak);
  const funded=Math.abs(st?.funding_paid||0)>1e-9;
  return [
    ["first_blood","🩸","FIRST BLOOD",(st?.trades||0)>=1,"Close your first trade"],
    ["ten","🎯","TEN DOWN",(st?.trades||0)>=10,"Close 10 trades"],
    ["fifty","💯","FIFTY",(st?.trades||0)>=50,"Close 50 trades"],
    ["green","📈","IN THE GREEN",(st?.total_pnl||0)>0,"Finish net positive"],
    ["pf","⚖️","EDGE PROVEN",(st?.trades||0)>=20&&(st?.profit_factor||0)>=1.2,"PF ≥ 1.2 over 20+ trades"],
    ["streak3","🔥","HAT-TRICK",sk.best>=3,"Win 3 trades in a row"],
    ["streak5","🌋","ON FIRE",sk.best>=5,"Win 5 trades in a row"],
    ["champ","👑","FIRST CHAMPION",ch.length>=1,"The tuner promotes a champion"],
    ["vault","🏛️","VAULT KEEPER",ch.length>=5,"Bank 5 champions"],
    ["gauntlet","🛡️","BATTLE TESTED",!!gaunt,"A champion survives the regime gauntlet"],
    ["brain","🧠","SELF-TAUGHT",graded>=500,"500 graded predictions"],
    ["ml","🤖","ML ONLINE",!!(meta&&meta.ready),"Meta-model earns its credentials"],
    ["gen","🧬","EVOLVER",(at?.generation||0)>=100,"100 DE generations"],
    ["uptime","⏱️","MARATHON",up>=24,"24h of continuous operation"],
    ["funding","💰","CARRY COLLECTED",funded,"Settle perp funding"],
    ["shadow","👻","THE RACE",!!(S?.shadow&&S.shadow.equity!=null),"Start the shadow-clock race"],
  ].filter(Boolean);
}
function missions(){
  const eng=S?.engine, st=eng?.portfolio?.stats, at=S?.autotuner;
  const ch=S?.champions||[], out=[];
  const active=ch.find(c=>c.active);
  const push=(t,have,need,note)=>out.push({t,have,need,note,
    pct:need>0?clamp(have/need,0,1):1,done:have>=need});
  // probation: the real gate on full-size risk
  if(active){
    const need=active.gauntlet_weak?16:8, have=(active.live?.trades)||0;
    push(`Clear probation — ${esc(active.id)}`,have,need,
      active.gauntlet_weak?"weak gauntlet → double proof required, trading at ½ size"
                          :"trading at ½ size until proven live");
  }
  const warm=Object.values(eng?.symbols||{});
  if(warm.length){
    const b=Math.min(...warm.map(s=>s.bars||0)), need=warm[0]?.warmup_bars||350;
    if(b<need) push("Warm up the brains",b,need,"bars of history before the gates arm");
  }
  const meta=at?.meta;
  if(meta&&!meta.model?.ready) push("Credential the meta-model",meta?.last_training?.n||0,3000,
      `needs AUC ≥ 0.53 · last ${meta?.last_training?.auc??"—"}`);
  push("Bank champions",ch.length,5,"a deeper vault means a better fallback");
  if(S?.autotuner?.clock_trial) push("Shadow race",(S?.shadow&&S.shadow.equity!=null)?1:0,1,
      S?.shadow?.status||"waiting for the first trial-clock champion");
  push("Prove the edge",st?.trades||0,20,"20 closed trades makes the stats mean something");
  return out;
}
let _achSeen=null,_achFresh=new Set();
/* Unlock detection runs on every full push, not only while the Progress tab is
   open — an achievement earned at 3am should still announce itself. */
function pollAchievements(){
  const ac=achievements();
  const got=new Set(ac.filter(a=>a[3]).map(a=>a[0]));
  if(_achSeen===null){ _achSeen=got; return; }   // first sight: no retro-toasts
  for(const a of ac) if(a[3]&&!_achSeen.has(a[0])){
    _achFresh.add(a[0]); toast(`🏆 Achievement unlocked — ${a[2]}`,"good");
  }
  _achSeen=got;
}
function renderProgress(){
  const op=operator(), sk=streaks(), ms=missions(), ac=achievements();
  const mEl=$("missions");
  if(mEl) mEl.innerHTML=ms.map(m=>`<div class="mission${m.done?" done":""}">
      <div class="mt"><span>${m.t}</span><span class="mv">${m.have} / ${m.need}${m.done?" ✓":""}</span></div>
      <div class="mbar"><i style="width:${(m.pct*100).toFixed(1)}%"></i></div>
      ${m.note?`<div class="mw">${esc(m.note)}</div>`:""}</div>`).join("");
  const got=ac.filter(a=>a[3]).length;
  const cEl=$("ach-count"); if(cEl) cEl.textContent=`— ${got}/${ac.length} unlocked`;
  const aEl=$("achievements");
  if(aEl){
    aEl.innerHTML=ac.map(([id,ic,nm,ok,hint])=>
      `<div class="ach${ok?" got":""}${_achFresh.has(id)?" fresh":""}" title="${esc(ok?nm:hint)}">
         <div class="ai">${ic}</div><div class="an">${nm}</div></div>`).join("");
    _achFresh.clear();   // the pop animation plays once, on first sight
  }
  const ref=$("refusals"), rf=S?.engine?.refusals;
  if(ref) ref.innerHTML=!rf?.gates?.length
    ? `<div class="cr"><span>${rf?.pending?`${rf.pending} pending · ${rf.graded||0} graded`:"no refusals graded yet"}</span><b>—</b></div>`
    : rf.gates.map(g=>{
        const m=g.mean_move_atr, cls=m>0.15?"pnl-pos":m<-0.15?"pnl-neg":"";
        return `<div class="cr" title="${g.refused} refused · ${(g.win_rate*100).toFixed(0)}% went the signal's way">
          <span>${esc(g.gate)}</span><b class="${cls}">${fmt.signed(m,2)} ATR <span style="color:var(--muted)">×${g.refused}</span></b></div>`;
      }).join("");
  const car=$("career");
  if(car) car.innerHTML=[
    ["Rank",`${op.title} · L${op.lvl}`],["Total XP",op.xp.toLocaleString()],
    ["Next rank",op.next?`${(op.next-op.xp).toLocaleString()} XP`:"MAXED"],
    ...op.parts.map(([k,n,w])=>[k,`${n.toLocaleString()} × ${w}`]),
    ["Best streak",`${sk.best} wins`],["Current streak",`${sk.cur} wins`],
  ].map(([k,v])=>`<div class="cr"><span>${k}</span><b>${v}</b></div>`).join("");
}
function renderRank(){
  const op=operator(), sk=streaks();
  const l=$("op-lvl"), t=$("op-title"), f=$("op-xp-fill"), s=$("op-streak"), n=$("op-streak-n");
  if(!l) return;
  if(l._v!==op.lvl){ l._v=op.lvl; l.textContent=op.lvl; }
  if(t._v!==op.title){ t._v=op.title; t.textContent=op.title; }
  const w=(op.pct*100).toFixed(1)+"%"; if(f._v!==w){ f._v=w; f.style.width=w; }
  if(n._v!==sk.cur){ n._v=sk.cur; n.textContent=sk.cur; s.classList.toggle("hot",sk.cur>=2); }
  const rk=$("op-rank");
  if(rk) rk.title=`${op.title} · level ${op.lvl} · ${op.xp.toLocaleString()} XP`
    +(op.next?` · ${(op.next-op.xp).toLocaleString()} to next rank`:"")
    +`\n${op.parts.map(([k,nn,ww])=>`${nn} ${k} × ${ww}`).join("\n")}`;
}

/* ================= NEURAL CORTEX — the brain, visibly firing =================
   A 60fps canvas of the focused symbol's actual wiring: 19 alpha neurons on an
   outer ring, clustered around their 5 desk hubs, all feeding the fused-edge
   core. Pulses travel the axons at the real signal strengths riding the 0.4s
   hot channel; between updates everything eases and breathes so the machine
   reads as alive, not as a chart. Perf rules: pre-rendered glow sprites (no
   shadowBlur), one trail-fade fillRect, DPR capped, hidden-tab pause. */
const ALPHA_DESK={momentum:"trend",macd:"trend",mtf_trend:"trend",breakout:"trend",roc_accel:"trend",
  meanrev_bb:"meanrev",capitulation:"meanrev",rsi_fade:"meanrev",stoch_fade:"meanrev",vwap_revert:"meanrev",vwap_pullback:"meanrev",
  obi:"micro",flow:"micro",cvd_trend:"micro",spread_pressure:"micro",
  squeeze:"vol",vol_breakout:"vol",funding_skew:"carry",oi_divergence:"carry"};
const ALPHA_SHORT={momentum:"MOM",macd:"MACD",mtf_trend:"MTF",breakout:"BRK",roc_accel:"ROC",
  meanrev_bb:"BB",capitulation:"CAP",rsi_fade:"RSI",stoch_fade:"STO",vwap_revert:"VWR",vwap_pullback:"VWP",
  obi:"OBI",flow:"FLOW",cvd_trend:"CVD",spread_pressure:"SPR",squeeze:"SQZ",vol_breakout:"VBK",
  funding_skew:"FND",oi_divergence:"OI"};
const REGIME_TINT={TREND_UP:"#00e0a0",TREND_DOWN:"#ff3d7f",RANGE:"#9d6bff",VOLATILE:"#ffc93d"};
const cortex=(()=>{
  const cv=$("cortex"); if(!cv) return {data(){}};
  const g=cv.getContext("2d");
  const bg=$("cortex-bg"), bgx=bg&&bg.getContext("2d");   // STATIC layer, own element
  const hud=$("cortex-hud"), hudLabels=$("hud-labels");   // TEXT layer, DOM
  const DPR=Math.min(window.devicePixelRatio||1,1.5);
  let W=0,H=0,cx=0,cy=0,R=0;
  let nodes=[],hubs={},layoutSig="",bgDirty=true,bgRegime="";
  let tgt=null;                                  // latest viz payload
  const ex={edge:0,p_win:0.5,regime:"RANGE",sym:"",stage:"SCAN",block:"",held:false};
  const e={edge:0,p_win:0.5,charge:0};           // eased display values
  const deskAcc={},pulses=[],ripples=[],motes=[],ekg=[];
  let spin=0,lastT=0,capT=0,hudT=0;
  const SPR={};
  const lerp=(a,b,k)=>a+(b-a)*k;
  const dirCol=(v)=>v>=0?C.up:C.dn;
  const TAU=Math.PI*2;
  const _colA={};
  function colA(col,a){ const q=Math.min(31,Math.max(0,(a*31)|0)); const k2=col+q;
    return _colA[k2]||(_colA[k2]=col+Math.round(q/31*255).toString(16).padStart(2,"0")); }
  function sprite(col){
    if(SPR[col]) return SPR[col];
    const s=document.createElement("canvas"); s.width=s.height=64;
    const c2=s.getContext("2d");
    const gr=c2.createRadialGradient(32,32,2,32,32,30);
    gr.addColorStop(0,col); gr.addColorStop(0.4,col+"66"); gr.addColorStop(1,col+"00");
    c2.fillStyle=gr; c2.fillRect(0,0,64,64);
    return SPR[col]=s;
  }
  let RS=1;   // render scale: DPR capped by a fixed PIXEL BUDGET, so the 30fps
              // trail-fade + glow pass costs the same on any monitor. Soft glow
              // art loses nothing visible at a slightly lower backing density.
  function resize(){
    const w=cv.clientWidth,h=cv.clientHeight;
    if(!w||!h) return;
    RS=Math.max(0.75,Math.min(DPR,Math.sqrt(560000/(w*h))));
    cv.width=Math.round(w*RS); cv.height=Math.round(h*RS);
    g.setTransform(RS,0,0,RS,0,0);
    // soft radial sprites don't need bilinear filtering — and the Firefox
    // profiler showed every filtered blit being rasterized on the CPU
    g.imageSmoothingEnabled=false;
    if(bg){ bg.width=Math.round(w*RS); bg.height=Math.round(h*RS); }
    W=w; H=h; cx=W*0.5; cy=H*0.52;
    R=Math.min(W*0.335,(cy-24)/1.16);
    if(!motes.length) for(let i=0;i<26;i++) motes.push({x:Math.random()*w,y:Math.random()*h,
      vx:(Math.random()-0.5)*5,vy:-3-Math.random()*5,ph:Math.random()*TAU});
    layout(); bgDirty=true;
    start();   // self-heal: a canvas that was zero-sized at load (hidden panel)
               // stopped the loop; any resize revives it
  }
  function layout(){
    if(!tgt) return;
    const old={}; for(const nd of nodes) old[nd.nm]=nd;
    nodes=[]; hubs={};
    DESK_ORDER.forEach((d,i)=>{
      const ang=-Math.PI/2+i*(TAU/DESK_ORDER.length);
      hubs[d]={x:cx+Math.cos(ang)*R*0.46,y:cy+Math.sin(ang)*R*0.46,ang,col:DESK_COLORS[d],sig:0,alloc:0.2};
      deskAcc[d]=deskAcc[d]||Math.random();
    });
    const byDesk={};
    for(const [nm] of tgt.a){ const d=ALPHA_DESK[nm]||"trend"; (byDesk[d]=byDesk[d]||[]).push(nm); }
    let idx=0;
    for(const d of DESK_ORDER){
      const names=byDesk[d]||[],hub=hubs[d],n=names.length;
      names.forEach((nm,j)=>{
        const off=(n>1?(j/(n-1)-0.5):0)*Math.min(1.25,0.36*(n-1));
        const ang=hub.ang+off;
        const x=cx+Math.cos(ang)*R*0.97, y=cy+Math.sin(ang)*R*0.97;
        // curved axon: control point pushed sideways so the wiring weaves
        const mx=(x+hub.x)/2, my=(y+hub.y)/2;
        const dx=hub.x-x, dy=hub.y-y, dl=Math.hypot(dx,dy)||1;
        const side=(idx++%2?1:-1)*0.22;
        const cpx=mx-dy/dl*dl*side, cpy=my+dx/dl*dl*side;
        const p=old[nm]||{sc:0,tsc:0,wt:0.2,twt:0.2,acc:Math.random(),ph:Math.random()*TAU};
        nodes.push({nm,short:ALPHA_SHORT[nm]||nm.slice(0,4).toUpperCase(),d,col:hub.col,hub,
          x,y,ang,cpx,cpy,sc:p.sc,tsc:p.tsc,wt:p.wt,twt:p.twt,acc:p.acc,ph:p.ph,fire:false});
      });
    }
    bgDirty=true;
    buildLabels();
  }
  const fscale=()=>Math.max(0.85,Math.min(2.0,R/165));
  /* The static layer. Repainted ONLY when the layout or the regime changes —
     it used to be re-blitted over the full canvas 30 times a second. Text is
     no longer drawn here at all; it lives in the DOM overlay below. */
  function drawBg(){
    if(!bgx) { bgDirty=false; return; }
    bgx.setTransform(RS,0,0,RS,0,0);
    bgx.clearRect(0,0,W,H);
    const tint=REGIME_TINT[bgRegime]||REGIME_TINT.RANGE;
    const wash=bgx.createRadialGradient(cx,cy,R*0.1,cx,cy,Math.max(W,H)*0.75);
    wash.addColorStop(0,tint+"1c"); wash.addColorStop(0.55,tint+"0a"); wash.addColorStop(1,"#00000000");
    bgx.fillStyle=wash; bgx.fillRect(0,0,W,H);
    bgx.strokeStyle="rgba(0,210,255,0.05)"; bgx.lineWidth=1;
    for(const rr of [0.46,0.97]){ bgx.beginPath(); bgx.arc(cx,cy,R*rr,0,TAU); bgx.stroke(); }
    bgx.setLineDash([2,7]);
    bgx.strokeStyle="rgba(0,210,255,0.06)";
    for(const d of DESK_ORDER){ const h=hubs[d]; if(!h) continue;
      bgx.beginPath(); bgx.moveTo(cx+Math.cos(h.ang)*R*0.18,cy+Math.sin(h.ang)*R*0.18);
      bgx.lineTo(cx+Math.cos(h.ang)*R*1.06,cy+Math.sin(h.ang)*R*1.06); bgx.stroke(); }
    bgx.setLineDash([]);
    bgDirty=false;
  }
  /* Text layer: one DOM node per alpha + per desk, positioned once per layout.
     Per frame we only ever touch a colour class on the few that are firing. */
  function buildLabels(){
    if(!hudLabels) return;
    const F=fscale();
    let html="";
    for(const nd of nodes){
      const x=cx+Math.cos(nd.ang)*R*1.115, y=cy+Math.sin(nd.ang)*R*1.115;
      html+=`<div class="hud-lab" data-nm="${nd.nm}" style="left:${x.toFixed(1)}px;top:${y.toFixed(1)}px;`
           +`font-size:${(9*F).toFixed(1)}px">${nd.short}</div>`;
    }
    for(const d of DESK_ORDER){ const h=hubs[d]; if(!h) continue;
      const x=cx+Math.cos(h.ang)*R*0.71, y=cy+Math.sin(h.ang)*R*0.71;
      html+=`<div class="hud-desk" style="left:${x.toFixed(1)}px;top:${y.toFixed(1)}px;`
           +`color:${h.col};font-size:${(11*F).toFixed(1)}px">${DESK_LABEL[d]}</div>`;
    }
    hudLabels.innerHTML=html;
    for(const el of hudLabels.querySelectorAll(".hud-lab")) labEl[el.dataset.nm]=el;
    const F2=fscale();
    const he=$("hud-edge"), hs=$("hud-sub"), hr=$("hud-regime"), ha=$("hud-armed");
    if(he){ he.style.left=cx+"px"; he.style.top=(cy-6*F2)+"px"; he.style.fontSize=(26*F2)+"px"; }
    if(hs){ hs.style.left=cx+"px"; hs.style.top=(cy+15*F2)+"px"; hs.style.fontSize=(11*F2)+"px"; }
    if(hr){ hr.style.left=cx+"px"; hr.style.top=(cy+31*F2)+"px"; hr.style.fontSize=(10.5*F2)+"px"; }
    if(ha){ ha.style.left=cx+"px"; ha.style.top=(cy-R*0.285-12*F2)+"px"; ha.style.fontSize=(11*F2)+"px"; }
    if(hud) hud.classList.add("ready");
  }
  const labEl={};
  function data(viz,extra){
    tgt=viz; Object.assign(ex,extra||{});
    const sig=viz.a.map(x=>x[0]).join(",");
    if(sig!==layoutSig){ layoutSig=sig; layout(); }
    if(ex.regime!==bgRegime){ bgRegime=ex.regime; bgDirty=true; }
    const m={}; for(const [nm,sc,wt] of viz.a) m[nm]=[sc,wt];
    for(const nd of nodes){ const v=m[nd.nm]; if(v){ nd.tsc=v[0]; nd.twt=v[1]; } }
    for(const d of DESK_ORDER){ const v=viz.d?.[d]; if(v&&hubs[d]){ hubs[d].sig=v[0]; hubs[d].alloc=v[1]; } }
    ekg.push(clamp(ex.edge,-1,1)); if(ekg.length>170) ekg.shift();   // ~1 min memory
  }
  function spawn(nd){   // pulse along the curved axon
    if(pulses.length>150) pulses.shift();
    pulses.push({x0:nd.x,y0:nd.y,cx:nd.cpx,cy:nd.cpy,x1:nd.hub.x,y1:nd.hub.y,
                 p:0,spd:0.9+1.6*Math.abs(nd.sc),col:dirCol(nd.sc),size:2.0+2.5*Math.abs(nd.sc)});
  }
  function spawnCore(h){   // hub -> core, straight but glowing
    if(pulses.length>150) pulses.shift();
    const s=Math.abs(h.sig)*Math.min(1,h.alloc*4);
    pulses.push({x0:h.x,y0:h.y,cx:(h.x+cx)/2,cy:(h.y+cy)/2,x1:cx,y1:cy,
                 p:0,spd:1.1+1.4*s,col:dirCol(h.sig),size:2.6+3.2*s,core:true});
  }
  /* One rAF per display refresh — motion is now locked to vsync. The old
     setTimeout(17)->rAF chain fired between refreshes, so frames landed at
     irregular 17-34ms intervals and the eye read it as judder even though
     each frame was cheap. Running every refresh costs ~0.5% CPU (measured)
     because the expensive parts (static layer, all text) no longer repaint. */
  let running=false;
  // ADAPTIVE limiter: we measure our OWN draw cost (EWMA) and halve the paint
  // rate if a frame is consistently expensive — so a capable machine gets the
  // full vsync-locked 60fps while a weak one (or a software-rendered browser)
  // degrades to 30 instead of dropping frames raggedly.
  let costMs=0,skip=false;
  const COST_BUDGET=5.0;
  function start(){ if(!running){ running=true; requestAnimationFrame(frame); } }
  function frame(t){
    if(document.hidden||!W){ running=false; return; }   // idle completely when hidden
    requestAnimationFrame(frame);
    if(costMs>COST_BUDGET){ skip=!skip; if(skip) return; }   // paint every other refresh
    const t0=performance.now();
    const dt=Math.min(0.08,Math.max(0.001,(t-lastT)/1000)); lastT=t;
    const k=Math.min(1,dt*5);
    // trail fade only — the static mesh is its own GPU-composited element now
    g.globalCompositeOperation="source-over";
    g.fillStyle="rgba(5,7,12,0.30)"; g.fillRect(0,0,W,H);
    if(bgDirty) drawBg();
    if(!tgt||!nodes.length) return;
    const thr=tgt.thr||0.3;
    e.edge=lerp(e.edge,ex.edge,k); e.p_win=lerp(e.p_win,ex.p_win,k);
    e.charge=lerp(e.charge,clamp(Math.abs(e.edge)/Math.max(thr,0.01),0,1.35),k);
    spin+=dt*(0.5+e.charge*2.4);
    // ambient motes — depth without cost (integer coords: no CPU filtering)
    g.globalCompositeOperation="lighter";
    for(const m of motes){
      m.x+=m.vx*dt; m.y+=m.vy*dt;
      if(m.y<-4) { m.y=H+4; m.x=Math.random()*W; }
      if(m.x<-4) m.x=W+4; else if(m.x>W+4) m.x=-4;
      g.globalAlpha=0.05+0.05*Math.sin(t*0.0011+m.ph);
      g.drawImage(sprite(C.accent),m.x-3|0,m.y-3|0,6,6);
    }
    g.globalAlpha=1; g.globalCompositeOperation="source-over";
    // curved axons — brightness follows live strength. Color strings are
    // CACHED per (color, quantized alpha): building ~750 fresh strings a
    // second fed Firefox's GC a periodic cleanup burst.
    for(const nd of nodes){
      nd.sc=lerp(nd.sc,nd.tsc,k); nd.wt=lerp(nd.wt,nd.twt,k);
      const s=Math.min(1,Math.abs(nd.sc));
      g.strokeStyle=colA(nd.col,0.05+0.22*s);
      g.lineWidth=0.8+1.4*s;
      g.beginPath(); g.moveTo(nd.x,nd.y); g.quadraticCurveTo(nd.cpx,nd.cpy,nd.hub.x,nd.hub.y); g.stroke();
      if(s>0.1){ nd.acc+=dt*(0.25+2.6*s); if(nd.acc>=1){ nd.acc=0; spawn(nd); } }
    }
    for(const d of DESK_ORDER){
      const h=hubs[d]; if(!h) continue;
      const s=Math.min(1,Math.abs(h.sig));
      g.strokeStyle=colA(h.col,0.10+0.30*s);
      g.lineWidth=1+2.6*h.alloc;
      g.beginPath(); g.moveTo(h.x,h.y); g.lineTo(cx,cy); g.stroke();
      const ss=s*Math.min(1,h.alloc*4);
      if(ss>0.05){ deskAcc[d]+=dt*(0.3+3.0*ss); if(deskAcc[d]>=1){ deskAcc[d]=0; spawnCore(h); } }
    }
    // pulses ride the curves; arrivals ripple
    g.globalCompositeOperation="lighter";
    for(let i=pulses.length-1;i>=0;i--){
      const p=pulses[i]; p.p+=p.spd*dt;
      if(p.p>=1){
        if(ripples.length<24) ripples.push({x:p.x1,y:p.y1,r:2,max:p.core?26:14,col:p.col});
        pulses.splice(i,1); continue;
      }
      const u=1-p.p;
      const x=u*u*p.x0+2*u*p.p*p.cx+p.p*p.p*p.x1;
      const y=u*u*p.y0+2*u*p.p*p.cy+p.p*p.p*p.y1;
      const s=Math.max(2,p.size*4*(1-p.p*0.35))|0;
      g.globalAlpha=0.85; g.drawImage(sprite(p.col),x-s/2|0,y-s/2|0,s,s);
    }
    for(let i=ripples.length-1;i>=0;i--){
      const r=ripples[i]; r.r+=dt*46;
      if(r.r>=r.max){ ripples.splice(i,1); continue; }
      g.globalAlpha=(1-r.r/r.max)*0.4; g.strokeStyle=r.col; g.lineWidth=1.2;
      g.beginPath(); g.arc(r.x,r.y,r.r,0,TAU); g.stroke();
    }
    // neuron + hub glows
    const F=fscale();
    for(const nd of nodes){
      const s=Math.min(1,Math.abs(nd.sc));
      const breathe=0.5+0.5*Math.sin(t*0.001+nd.ph);
      const sz=(7+26*s+3*breathe*(0.3+s))*F|0;
      g.globalAlpha=0.26+0.62*s;
      g.drawImage(sprite(s>0.05?dirCol(nd.sc):nd.col),nd.x-sz/2|0,nd.y-sz/2|0,sz,sz);
    }
    for(const d of DESK_ORDER){
      const h=hubs[d]; const sz=(15+32*h.alloc+15*Math.min(1,Math.abs(h.sig)))*F|0;
      g.globalAlpha=0.55; g.drawImage(sprite(h.col),h.x-sz/2|0,h.y-sz/2|0,sz,sz);
    }
    const coreCol=dirCol(e.edge);
    const coreSz=R*0.60*(0.75+0.45*e.charge)|0;
    g.globalAlpha=0.28+0.45*Math.min(1,e.charge);
    g.drawImage(sprite(coreCol),cx-coreSz/2|0,cy-coreSz/2|0,coreSz,coreSz);
    g.globalAlpha=1; g.globalCompositeOperation="source-over";
    // crisp marks: neurons, hub cores + allocation arcs
    for(const nd of nodes){
      const s=Math.min(1,Math.abs(nd.sc));
      g.fillStyle=s>0.05?dirCol(nd.sc):"#2a3346";
      g.beginPath(); g.arc(nd.x,nd.y,1.6+1.7*s,0,TAU); g.fill();
    }
    for(const d of DESK_ORDER){
      const h=hubs[d];
      g.fillStyle=h.col; g.beginPath(); g.arc(h.x,h.y,2.4+3.2*h.alloc,0,TAU); g.fill();
      g.strokeStyle=h.col+"88"; g.lineWidth=1.6;   // allocation arc around the hub
      g.beginPath(); g.arc(h.x,h.y,8.5,-Math.PI/2,-Math.PI/2+clamp(h.alloc/0.4,0,1)*TAU); g.stroke();
    }
    // the core: threshold ring, |edge| arc, p(win) inner arc, ARMED reticle
    const cr=R*0.285;
    g.lineWidth=3+F; g.strokeStyle="#141827";
    g.beginPath(); g.arc(cx,cy,cr,0,TAU); g.stroke();
    g.strokeStyle=coreCol;
    g.beginPath(); g.arc(cx,cy,cr,-Math.PI/2,-Math.PI/2+clamp(Math.abs(e.edge),0,1)*TAU*(e.edge>=0?1:-1),e.edge<0); g.stroke();
    g.lineWidth=2+F*0.6; g.strokeStyle="#1d2436";
    g.beginPath(); g.arc(cx,cy,cr-7*F,0,TAU); g.stroke();
    g.strokeStyle=C.accent+"cc";
    g.beginPath(); g.arc(cx,cy,cr-7*F,-Math.PI/2,-Math.PI/2+clamp(e.p_win,0,1)*TAU); g.stroke();
    for(const s of [-1,1]){   // threshold ticks on the edge ring
      const a=-Math.PI/2+s*clamp(thr,0,1)*TAU;
      g.strokeStyle="#8a97ad"; g.lineWidth=2.5;
      g.beginPath(); g.moveTo(cx+Math.cos(a)*(cr-4),cy+Math.sin(a)*(cr-4));
      g.lineTo(cx+Math.cos(a)*(cr+5),cy+Math.sin(a)*(cr+5)); g.stroke();
    }
    if(e.charge>=1){          // ARMED: twin counter-rotating reticles
      g.strokeStyle=coreCol; g.lineWidth=1.5;
      g.setLineDash([5,10]); g.lineDashOffset=-spin*30;
      g.beginPath(); g.arc(cx,cy,cr+8,0,TAU); g.stroke();
      g.setLineDash([2,12]); g.lineDashOffset=spin*44;
      g.beginPath(); g.arc(cx,cy,cr+14,0,TAU); g.stroke();
      g.setLineDash([]);
    }
    // edge EKG — the last minute of conviction, breathing along the bottom
    if(ekg.length>2){
      const eh=15, ey=H-10, x0=10, x1=W-10;
      g.strokeStyle="rgba(0,210,255,0.12)"; g.lineWidth=1;
      g.beginPath(); g.moveTo(x0,ey); g.lineTo(x1,ey); g.stroke();
      g.beginPath();
      for(let i=0;i<ekg.length;i++){
        const x=x0+(x1-x0)*(i/(ekg.length-1)), y=ey-ekg[i]*eh;
        i?g.lineTo(x,y):g.moveTo(x,y);
      }
      g.strokeStyle=dirCol(ekg[ekg.length-1])+"bb"; g.lineWidth=1.5; g.stroke();
    }
    // ---- TEXT: DOM, ~12Hz. Canvas text was re-rasterized every frame (up to
    // 19 alpha labels + 4 HUD lines); as elements the browser keeps a cached
    // layer and this loop only writes strings that actually changed.
    if(t-hudT>80){ hudT=t;
      const he=$("hud-edge"), hs=$("hud-sub"), hr=$("hud-regime"), ha=$("hud-armed");
      const ev=(e.edge>=0?"+":"−")+Math.abs(e.edge).toFixed(2);
      if(he&&he._v!==ev){ he._v=ev; he.textContent=ev; }
      if(he&&he._c!==coreCol){ he._c=coreCol; he.style.color=coreCol; }
      const sv=`P ${Math.round(e.p_win*100)}%${tgt.meta_p!=null?`  ·  ML ${Math.round(tgt.meta_p*100)}%`:""}`;
      if(hs&&hs._v!==sv){ hs._v=sv; hs.textContent=sv; }
      const rm=REGIME_META[ex.regime]||REGIME_META.RANGE;
      const rv=`${rm.g} ${ex.regime.replace("_"," ")}`;
      if(hr&&hr._v!==rv){ hr._v=rv; hr.textContent=rv; hr.style.color=REGIME_TINT[ex.regime]||"#8a97ad"; }
      const armed=e.charge>=1;
      if(ha&&ha._on!==armed){ ha._on=armed; ha.classList.toggle("on",armed); ha.style.color=coreCol; }
      for(const nd of nodes){    // a firing alpha lights its own name
        const fire=Math.abs(nd.sc)>0.35, el=labEl[nd.nm];
        if(el&&nd.fire!==fire){ nd.fire=fire;
          el.classList.toggle("fire",fire); el.style.color=fire?dirCol(nd.sc):""; }
      }
      const cap=$("cortex-cap");
      if(cap&&t-capT>400){ capT=t;
        const cv2=`${ex.sym||"—"} · ${ex.held?"IN POSITION":(armed?"⚡ ARMED":ex.stage||"SCAN")}${ex.block&&!ex.held?` · ${ex.block}`:""}`;
        if(cap._v!==cv2){ cap._v=cv2; cap.textContent=cv2; }
      }
    }
    costMs=costMs*0.9+(performance.now()-t0)*0.1;   // EWMA of our own draw cost
  }
  new ResizeObserver(resize).observe(cv);
  document.addEventListener("visibilitychange",()=>{ if(!document.hidden){ lastT=performance.now(); start(); } });
  resize(); start();
  return {data};
})();

initCharts(); connectWS();
// slow reconciliation only (closed bars + markers) — the live candle rides the
// 0.4s hot channel now; skip entirely while the tab is hidden.
setInterval(()=>{ if(S?.engine&&!document.hidden) refreshCandles(false); },10000);

/* ==========================================================================
   VIZ LAYER — canvas views over data the dashboard previously printed as
   numbers. Measured first: the profiler showed the page 77% idle with zero
   long tasks, so the goal here is to ADD information without spending that
   headroom. Three rules keep it that way:

     1. nothing draws unless its tab is the active one (renderBottomTab
        already dispatches per page — these hang off the same switch);
     2. every view is a pure function of a snapshot; no view owns a timer or
        a rAF loop of its own;
     3. hover state repaints only the view under the cursor, and only on
        pointermove, never on a clock.
   ========================================================================== */

const VIZ_DPR = () => Math.min(window.devicePixelRatio || 1, 2);

/** Size a canvas's backing store to its CSS box. Returns the 2D context with
 *  the transform already set so all drawing is in CSS pixels. */
const _vizBox = new WeakMap();
function vizInvalidate(cv){ _vizBox.delete(cv); }
function vizCtx(cv){
  // clientWidth/clientHeight force a layout flush. The box only changes on
  // resize, and a ResizeObserver already tells us when that happens, so read
  // it once and reuse it rather than on every repaint.
  let box = _vizBox.get(cv);
  if(!box){ box = { w: cv.clientWidth, h: cv.clientHeight }; _vizBox.set(cv, box); }
  const w = box.w, h = box.h;
  if(!w || !h){ _vizBox.delete(cv); return null; }
  const d = VIZ_DPR();
  const bw = Math.round(w * d), bh = Math.round(h * d);
  if(cv.width !== bw || cv.height !== bh){ cv.width = bw; cv.height = bh; }
  const g = cv.getContext("2d", { alpha: true });
  g.setTransform(d, 0, 0, d, 0, 0);
  g.clearRect(0, 0, w, h);
  g.__w = w; g.__h = h;
  return g;
}
/** Theme colours, resolved ONCE.
 *  getComputedStyle() forces a style resolution on every call, and the views
 *  below ask for a colour inside their draw loops — the constellation alone
 *  wanted ~15 per repaint. Profiling the main tab after adding it showed p95
 *  drifting 23.6ms -> 29ms with long tasks appearing, which is the entire cost
 *  of asking the engine the same fixed question over and over. The palette is
 *  a static dark theme, so it is read once and memoized. */
const _cssCache = new Map();
function cssVar(n){
  let v = _cssCache.get(n);
  if(v === undefined){
    v = getComputedStyle(document.documentElement).getPropertyValue(n).trim();
    _cssCache.set(n, v);
  }
  return v;
}
function vizEmpty(host, msg){
  const e = host.querySelector(".viz-empty");
  if(e) e.textContent = msg, e.style.display = msg ? "grid" : "none";
}
/** Nice axis ticks covering [lo,hi] — at most `n` of them, on 1/2/5 steps. */
function niceTicks(lo, hi, n = 5){
  if(!isFinite(lo) || !isFinite(hi) || hi <= lo) return [lo || 0];
  const raw = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * mag;
  const out = [];
  for(let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

/** p-quantile of an ASCENDING array. */
function pq(sorted, p){
  if(!sorted.length) return 0;
  const i = Math.min(sorted.length - 1, Math.max(0, Math.round((sorted.length - 1) * p)));
  return sorted[i];
}

/* ---------------------------------------------------- 1. EXCURSION MAP
   Every closed trade as a point: how good it ever looked (MFE, x) against how
   much heat it took first (MAE, y), both in R. The single most diagnostic
   picture a discretionary-style system can draw, and this dashboard only ever
   had the two AVERAGES of it as cards.

   Read it like this:
     · far right, low     -> clean winners, little heat. More of these.
     · far right, HIGH y  -> we were up big and gave it back through the stop.
     · clustered near 0,0 -> chop; the gates should have refused these.
     · above the diagonal -> the trade hurt more than it ever showed. */
let _exTrades = [], _exHit = -1, _exGeom = null, _exDist = null;
function renderExcursion(rows){
  const host = $("viz-excursion"); if(!host) return;
  const cv = host.querySelector("canvas"); const g = vizCtx(cv); if(!g) return;
  const W = g.__w, H = g.__h;
  const pts = (rows || []).filter(r =>
    Number.isFinite(r.mfe_r) && Number.isFinite(r.mae_r) && Number.isFinite(r.r));
  _exTrades = pts;
  vizEmpty(host, pts.length ? "" : "No closed trades yet — the map fills as trades journal.");
  if(!pts.length){ _exGeom = null; return; }

  const pad = { l: 56, r: 16, t: 26, b: 26 };
  // ROBUST axes. R-multiples are heavy-tailed — one 30R excursion on a linear
  // scale crushes every other trade into the corner and the chart says nothing.
  // Scale to the 95th percentile and pin the rest to the rim as hollow markers,
  // so the bulk is readable and the outliers are still visibly THERE rather
  // than quietly dropped.
  const sx = pts.map(p => p.mfe_r).sort((a, b) => a - b);
  const sy = pts.map(p => p.mae_r).sort((a, b) => a - b);
  const maxX = Math.max(1, pq(sx, 0.98));
  const maxY = Math.max(0.5, pq(sy, 0.98));
  // SQUARE-ROOT axes. Percentile clipping alone was not enough: excursions are
  // so skewed (avg MFE ~10R against avg MAE ~0.6R on this account) that even
  // the p95 range leaves the whole population stacked against the origin. sqrt
  // keeps 0 at 0, spreads the dense low end where the trades actually are, and
  // compresses the tail that was drowning them — while the ticks stay labelled
  // in real R, so nothing about the reading changes.
  const sq = (v, m) => Math.sqrt(Math.max(0, Math.min(v, m)) / m);
  const X = v => pad.l + sq(v, maxX) * (W - pad.l - pad.r);
  const Y = v => H - pad.b - sq(v, maxY) * (H - pad.t - pad.b);
  _exGeom = { X, Y, pad, W, H };

  g.strokeStyle = cssVar("--grid"); g.lineWidth = 1;
  g.fillStyle = cssVar("--muted"); g.font = "9.5px " + cssVar("--mono");
  // ticks at fractions of the SQRT range, so they are evenly spaced on screen
  const axTicks = (m) => [0.04, 0.16, 0.36, 0.64, 1].map(f => +(m * f).toFixed(m < 3 ? 2 : 1));
  for(const t of axTicks(maxX)){
    const x = Math.round(X(t)) + .5;
    g.beginPath(); g.moveTo(x, pad.t); g.lineTo(x, H - pad.b); g.stroke();
    g.textAlign = "center"; g.fillText(t + "R", x, H - pad.b + 13);
  }
  for(const t of axTicks(maxY)){
    const y = Math.round(Y(t)) + .5;
    g.beginPath(); g.moveTo(pad.l, y); g.lineTo(W - pad.r, y); g.stroke();
    g.textAlign = "right"; g.fillText(t + "R", pad.l - 6, y + 3);
  }
  // the break-even diagonal: heat taken == profit ever shown
  g.strokeStyle = "rgba(255,201,61,0.30)"; g.setLineDash([3, 3]);
  g.beginPath();
  const dEnd = Math.min(maxX, maxY);
  for(let k = 0; k <= 24; k++){
    const v = dEnd * k / 24;
    k ? g.lineTo(X(v), Y(v)) : g.moveTo(X(v), Y(v));
  }
  g.stroke(); g.setLineDash([]);
  g.fillStyle = "rgba(255,201,61,0.6)"; g.textAlign = "left";
  g.fillText("heat = reward", X(dEnd) + 5, Y(dEnd) + 3);

  const good = cssVar("--good"), bad = cssVar("--bad");
  let clipped = 0;
  for(let i = 0; i < pts.length; i++){
    const p = pts[i], win = p.r > 0;
    const off = p.mfe_r > maxX || p.mae_r > maxY;
    if(off) clipped++;
    const rad = Math.max(2.2, Math.min(8, 2.2 + Math.abs(p.r) * 1.2));
    const x = X(Math.min(p.mfe_r, maxX)), y = Y(Math.min(p.mae_r, maxY));
    const col = win ? good : bad;
    g.beginPath(); g.arc(x, y, rad, 0, 6.2832);
    if(off){                       // outlier: hollow, so it reads as "beyond"
      g.strokeStyle = col; g.lineWidth = 1.4; g.stroke();
    } else {
      g.fillStyle = col + (i === _exHit ? "" : "55"); g.fill();
      g.strokeStyle = col + (i === _exHit ? "" : "99");
      g.lineWidth = i === _exHit ? 1.8 : 1; g.stroke();
    }
    if(i === _exHit){ g.strokeStyle = cssVar("--accent-2"); g.lineWidth = 1.8; g.stroke(); }
  }
  if(clipped){
    g.fillStyle = cssVar("--muted"); g.font = "9.5px " + cssVar("--mono");
    g.textAlign = "right";
    g.fillText(`${clipped} beyond scale (hollow)`, W - pad.r - 2, pad.t - 12);
  }
  g.fillStyle = cssVar("--muted"); g.font = "9.5px " + cssVar("--mono");
  g.save(); g.translate(12, (H - pad.b + pad.t) / 2); g.rotate(-Math.PI / 2);
  g.textAlign = "center"; g.fillText("MAE — heat taken", 0, 0); g.restore();
  g.textAlign = "right"; g.fillText("MFE — best it ever showed", W - pad.r, H - 5);
}
function exHover(ev){
  const host = $("viz-excursion"); if(!host || !_exGeom || !_exTrades.length) return;
  const cv = host.querySelector("canvas"), b = cv.getBoundingClientRect();
  const mx = ev.clientX - b.left, my = ev.clientY - b.top;
  let hit = -1, best = 14 * 14;
  for(let i = 0; i < _exTrades.length; i++){
    const p = _exTrades[i];
    const dx = _exGeom.X(p.mfe_r) - mx, dy = _exGeom.Y(p.mae_r) - my;   // X/Y clamp internally
    const d2 = dx * dx + dy * dy;
    if(d2 < best){ best = d2; hit = i; }
  }
  const tip = host.querySelector(".viz-tip");
  if(hit === _exHit && hit < 0) return;
  if(hit !== _exHit){ _exHit = hit; renderExcursion(_exTrades); }
  if(hit < 0){ tip.classList.remove("on"); return; }
  const p = _exTrades[hit];
  tip.innerHTML = `<b>${esc(p.symbol || "")}</b> ${esc(p.side || "")} · <b>${(p.r).toFixed(2)}R</b><br>`
    + `showed ${p.mfe_r.toFixed(2)}R · took ${p.mae_r.toFixed(2)}R heat<br>`
    + `${esc(p.regime || "")} · ${esc(p.reason_close || "")}`;
  tip.classList.add("on");
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  tip.style.left = Math.min(Math.max(4, mx + 12), cv.clientWidth - tw - 4) + "px";
  tip.style.top = Math.max(4, my - th - 10) + "px";
}

/* ---------------------------------------------------- 2. R DISTRIBUTION
   The SHAPE of outcomes, not their mean. A trend system is supposed to look
   like this: a fat stack of small negatives just left of zero and a long thin
   tail to the right. If the right tail is missing, exits are cutting winners;
   if the left stack extends past -1R, stops are slipping. */
function renderRDist(rows, dist){
  const host = $("viz-rdist"); if(!host) return;
  const cv = host.querySelector("canvas"); const g = vizCtx(cv); if(!g) return;
  const W = g.__w, H = g.__h;
  const vals = (rows || []).map(r => r.r).filter(Number.isFinite);
  vizEmpty(host, vals.length ? "" : "No journaled R-multiples yet.");
  if(!vals.length) return;

  // same heavy-tail problem as the scatter: a single -12R drags the axis and
  // leaves 30 empty bins. Clip to the 2nd/98th percentile and let the end bins
  // ACCUMULATE everything past them, so the tail is counted, just not plotted
  // to scale.
  const sv = vals.slice().sort((a, b) => a - b);
  const lo = Math.min(-1.5, Math.floor(pq(sv, 0.02) * 2) / 2);
  const hi = Math.max(2.0, Math.ceil(pq(sv, 0.98) * 2) / 2);
  const BINS = Math.max(12, Math.min(34, Math.round(W / 22)));
  const step = (hi - lo) / BINS;
  const bins = new Array(BINS).fill(0);
  let under = 0, over = 0;
  for(const v of vals){
    if(v < lo) under++; else if(v > hi) over++;
    bins[Math.max(0, Math.min(BINS - 1, Math.floor((v - lo) / step)))]++;
  }
  const peak = Math.max(1, ...bins);
  const pad = { l: 28, r: 10, t: 24, b: 24 };
  const X = v => pad.l + ((v - lo) / (hi - lo)) * (W - pad.l - pad.r);
  const bw = (W - pad.l - pad.r) / BINS;

  for(let i = 0; i < BINS; i++){
    if(!bins[i]) continue;
    const c0 = lo + i * step;
    const h = (bins[i] / peak) * (H - pad.t - pad.b);
    const x = pad.l + i * bw;
    const grd = g.createLinearGradient(0, H - pad.b - h, 0, H - pad.b);
    const col = c0 >= 0 ? cssVar("--good") : cssVar("--bad");
    grd.addColorStop(0, col + "ee"); grd.addColorStop(1, col + "33");
    g.fillStyle = grd;
    g.fillRect(x + 0.6, H - pad.b - h, Math.max(1, bw - 1.2), h);
  }
  // zero line
  g.strokeStyle = cssVar("--baseline"); g.lineWidth = 1;
  const zx = Math.round(X(0)) + .5;
  g.beginPath(); g.moveTo(zx, pad.t - 4); g.lineTo(zx, H - pad.b); g.stroke();

  // percentile markers straight from the journal summary
  const marks = dist ? [["p10", dist.p10], ["p50", dist.p50], ["p90", dist.p90]] : [];
  g.font = "9.5px " + cssVar("--mono");
  for(const [lab, v] of marks){
    if(!Number.isFinite(v) || v < lo || v > hi) continue;
    const x = Math.round(X(v)) + .5;
    g.strokeStyle = "rgba(0,210,255,0.5)"; g.setLineDash([2, 3]);
    g.beginPath(); g.moveTo(x, pad.t); g.lineTo(x, H - pad.b); g.stroke();
    g.setLineDash([]);
    g.fillStyle = cssVar("--accent-2"); g.textAlign = "center";
    g.fillText(lab, x, pad.t - 6);
  }
  g.fillStyle = cssVar("--muted"); g.textAlign = "center"; g.font = "9.5px " + cssVar("--mono");
  for(const t of niceTicks(lo, hi, 6)) g.fillText(t + "R", X(t), H - 8);
  if(under || over){
    g.textAlign = "right"; g.fillStyle = cssVar("--muted");
    g.fillText(`${under ? `${under} ≤${lo}R` : ""}${under && over ? " · " : ""}${over ? `${over} ≥${hi}R` : ""}`,
               W - pad.r, H - 8);
  }
}

/* ---------------------------------------------------- 3. THE GHOST BOOK
   What the gates REFUSED, and whether refusing was right. The refusal ledger
   already grades every blocked signal against what price actually did next —
   this draws it. Bar length is how many that gate refused; the colour is the
   win rate of the trades it turned away, so a gate glowing red is one that has
   been rejecting winners and is costing money by existing. */
function renderGhostBook(ref){
  const host = $("viz-ghost"); if(!host) return;
  const cv = host.querySelector("canvas"); const g = vizCtx(cv); if(!g) return;
  const W = g.__w, H = g.__h;
  const gates = ((ref && ref.gates) || []).filter(x => x && x.refused > 0)
    .sort((a, b) => b.refused - a.refused).slice(0, 8);
  vizEmpty(host, gates.length
    ? "" : "No graded refusals yet — the ledger needs a few bars to mature.");
  if(!gates.length) return;

  const pad = { l: 106, r: 96, t: 10, b: 14 };   // r fits "123 · 44% won"
  const maxN = Math.max(...gates.map(x => x.refused));
  const rowH = Math.min(24, (H - pad.t - pad.b) / gates.length);
  g.font = "10px " + cssVar("--mono");
  gates.forEach((x, i) => {
    const y = pad.t + i * rowH, h = Math.max(6, Math.min(15, rowH - 7));
    const w = Math.max(2, (x.refused / maxN) * (W - pad.l - pad.r));
    // hue by win rate of what it refused: green = correctly refused losers,
    // red = it has been throwing away winners
    const wr = Number.isFinite(x.win_rate) ? x.win_rate : 0.5;
    const col = wr >= 0.55 ? cssVar("--bad") : wr <= 0.4 ? cssVar("--good") : cssVar("--warn");
    const grd = g.createLinearGradient(pad.l, 0, pad.l + w, 0);
    grd.addColorStop(0, col + "22"); grd.addColorStop(1, col + "cc");
    g.fillStyle = grd; g.fillRect(pad.l, y, w, h);
    g.fillStyle = cssVar("--ink-2"); g.textAlign = "right";
    g.fillText(x.gate.length > 15 ? x.gate.slice(0, 14) + "…" : x.gate, pad.l - 7, y + h - 1);
    g.fillStyle = col; g.textAlign = "left";
    g.fillText(`${x.refused} · ${(wr * 100).toFixed(0)}% won`, pad.l + w + 6, y + h - 1);
  });
}

/* ---------------------------------------------------- 4. ALPHA CONSTELLATION
   The 19-signal floor as a live radial field instead of a list. Angle groups
   by desk, radius is |score| (centre = silent, rim = maximum conviction), and
   the ring colour is the desk. A firing alpha pulses. Same data the alpha list
   shows — this makes "which desks are speaking right now" readable at a glance
   instead of by scanning nineteen numbers. */
let _constBg = null, _constKey = "";

/* Each desk owns an equal angular SECTOR. The first draft gave every desk the
   same 0.92rad spread around a centre point, which let neighbouring desks
   overlap and stacked low-conviction alphas on top of each other at the hub.
   Sectors with a sqrt radius fix both: quiet alphas still separate near the
   centre, and a desk's members always read as one group. */
function renderConstellation(alphas){
  const host = $("viz-constellation"); if(!host) return;
  const cv = host.querySelector("canvas"); const g = vizCtx(cv); if(!g) return;
  const W = g.__w, H = g.__h;
  // the snapshot keys alphas by NAME, so normalize to a list first — the
  // constellation needs the name for nothing but grouping, and reading
  // `.length` off the raw object silently produced undefined (an empty view
  // with no error, which is exactly how this shipped broken the first time).
  const list = alphas && !Array.isArray(alphas)
    ? Object.entries(alphas).map(([nm, a]) => ({ ...a, name: nm }))
    : (alphas || []);
  vizEmpty(host, list.length ? "" : "Warming up…");
  if(!list.length) return;

  const cx = W / 2, cy = H / 2;
  const R = Math.max(24, Math.min(W * 0.34, (H - 30) * 0.46));
  const HUB = 9;
  const byDesk = {};
  for(const a of list) (byDesk[a.desk] ||= []).push(a);
  const desks = DESK_ORDER.filter(d => byDesk[d]);
  if(!desks.length) return;

  const SEC = 6.2832 / desks.length;
  // STATIC LAYER, cached. Rings, sector dividers and desk labels depend only on
  // the box and the desk roster, but they were being re-rasterized on every
  // repaint. Measured with the constellation removed vs present: p50 was
  // identical (17.2 vs 17.3ms) while p95 went 21.2 -> 26.1ms and frames over
  // 33ms went 9 -> 23 — i.e. the cost was not per-frame, it was a spike each
  // time it redrew. Blitting a cached bitmap turns that back into one drawImage.
  const key = `${W}x${H}|${desks.join(",")}`;
  if(_constKey !== key){
    _constKey = key;
    const d = VIZ_DPR();
    _constBg = document.createElement("canvas");
    _constBg.width = Math.round(W * d); _constBg.height = Math.round(H * d);
    const bg = _constBg.getContext("2d");
    bg.setTransform(d, 0, 0, d, 0, 0);
    bg.strokeStyle = cssVar("--grid"); bg.lineWidth = 1;
    for(const f of [0.4, 0.7, 1]){
      bg.beginPath(); bg.arc(cx, cy, HUB + (R - HUB) * f, 0, 6.2832); bg.stroke();
    }
    desks.forEach((desk, di) => {
      const c0 = -Math.PI / 2 + di * SEC;
      bg.strokeStyle = cssVar("--grid");
      bg.beginPath(); bg.moveTo(cx, cy);
      bg.lineTo(cx + Math.cos(c0 - SEC / 2) * R * 1.06, cy + Math.sin(c0 - SEC / 2) * R * 1.06);
      bg.stroke();
      const col = cssVar("--" + desk) || cssVar("--accent");
      const lx = cx + Math.cos(c0) * (R + 15), ly = cy + Math.sin(c0) * (R + 15);
      bg.fillStyle = col + "dd"; bg.font = "9px " + cssVar("--mono");
      bg.textAlign = Math.abs(Math.cos(c0)) < 0.3 ? "center" : (Math.cos(c0) > 0 ? "left" : "right");
      bg.fillText(desk, lx, ly + 3);
    });
    bg.fillStyle = cssVar("--muted"); bg.font = "9px " + cssVar("--mono");
    bg.textAlign = "center";
    bg.fillText("centre = silent · rim = full conviction", cx, H - 4);
  }
  g.drawImage(_constBg, 0, 0, W, H);

  const t = performance.now() / 1000;
  desks.forEach((desk, di) => {
    const arr = byDesk[desk];
    const col = cssVar("--" + desk) || cssVar("--accent");
    const c0 = -Math.PI / 2 + di * SEC;
    arr.forEach((a, i) => {
      // spread across 80% of the sector, centred
      const frac = arr.length === 1 ? 0 : (i / (arr.length - 1)) - 0.5;
      const ang = c0 + frac * SEC * 0.8;
      const mag = Math.min(1, Math.abs(a.score || 0));
      // sqrt keeps quiet alphas apart instead of piling them on the hub
      const r = HUB + Math.sqrt(mag) * (R - HUB);
      const x = cx + Math.cos(ang) * r, y = cy + Math.sin(ang) * r;
      g.strokeStyle = col + "2e"; g.lineWidth = 1;
      g.beginPath();
      g.moveTo(cx + Math.cos(ang) * HUB, cy + Math.sin(ang) * HUB);
      g.lineTo(x, y); g.stroke();
      const firing = mag > 0.35;
      const rad = (2 + mag * 3.6) * (firing ? 1 + 0.16 * Math.sin(t * 3 + i) : 1);
      if(firing){          // halo kept tight — a big filled arc is pure overdraw
        g.fillStyle = col + "1f";
        g.beginPath(); g.arc(x, y, rad * 2.0, 0, 6.2832); g.fill();
      }
      g.fillStyle = a.score >= 0 ? col : col + "aa";
      g.beginPath(); g.arc(x, y, rad, 0, 6.2832); g.fill();
    });
  });
}

/* hover readout — bound once, fires only while the pointer is over the map */
(() => {
  const host = $("viz-excursion"); if(!host) return;
  const cv = host.querySelector("canvas");
  cv.addEventListener("pointermove", exHover);
  cv.addEventListener("pointerleave", () => {
    host.querySelector(".viz-tip").classList.remove("on");
    if(_exHit !== -1){ _exHit = -1; renderExcursion(_exTrades); }
  });
  // a canvas has no intrinsic size, so its backing store must follow its box
  new ResizeObserver(() => {
    document.querySelectorAll(".viz canvas").forEach(vizInvalidate);
    if(activePage() === "analytics"){
      // all three, not two: a view left un-redrawn keeps a stale backing store
      // and gets CSS-stretched to the new box, i.e. silently blurry
      renderExcursion(_exTrades);
      renderRDist(_exTrades, _exDist);
      renderGhostBook(S?.engine?.refusals);
    }
    renderLiveViews();   // the main-grid views are never behind a tab
  }).observe(host);
})();

/* ==========================================================================
   MAIN-GRID LIVE VIEWS — the lower-right of the terminal

   Four canvases that live in the always-visible grid rather than behind a tab,
   so unlike the analytics views they repaint on the HOT channel (up to 4/s).
   Each one draws something the dashboard already knew but only ever printed:

     · symbol board  — the whole watchlist at once (everything else follows one
                       focused symbol, so the other markets were invisible)
     · trade runway  — open positions on a stop -> target axis
     · order flow    — OBI/flow as a moving ribbon instead of one line of text
     · risk budget   — distance to the kill switch, exposure cap and slot cap

   Same three rules as the analytics views: no view owns a timer, every draw is
   a pure function of the current snapshot, and hover repaints only its own
   view. They are cheap by construction — a few dozen rects and two polylines —
   because at 4 Hz on the main screen they share a frame with the cortex.
   ========================================================================== */

/** Blank a live view and report why, so a warming-up panel never looks broken. */
function _liveEmpty(host, msg){ vizEmpty(host, msg); return null; }

/** Common setup: fetch the context, mark the view empty when there is no
 *  engine yet. Returns null when there is nothing to draw. */
function _liveCtx(id, emptyMsg){
  const host = $(id); if(!host) return null;
  const cv = host.querySelector("canvas"); const g = vizCtx(cv);
  if(!g) return null;
  if(!S?.engine){ _liveEmpty(host, emptyMsg); return null; }
  vizEmpty(host, "");
  return { host, cv, g, W: g.__w, H: g.__h };
}

const shortSym = (s) => String(s || "").replace("-USDT", "").replace("-USD", "");

/* ------------------------------------------------------------ 1. SYMBOL BOARD
   Every market the machine watches, on one line each: regime, price, where the
   price sits in its own 24h range, how close the edge is to the threshold, and
   the calibrated P(win). The chart, cortex and brain all show ONE symbol; this
   is the only place the other four are visible at all. Click focuses. */
let _boardGeom = null, _boardHit = -1;
function renderBoard(){
  const c = _liveCtx("viz-board", "Engine idle — start paper or live mode."); if(!c) return;
  const { host, g, W, H } = c;
  const syms = engineSymbols(), es = S.engine.symbols || {};
  if(!syms.length){ _liveEmpty(host, "No symbols"); _boardGeom = null; return; }

  const posns = S.engine.portfolio?.open_positions || {};
  const adopted = new Set(S.engine.adopted || []);
  const narrow = W < 430;
  const HEAD = 15, TOP = HEAD + 5;
  // rows take the room they are given: a three-symbol watchlist should breathe,
  // an eight-symbol one should still fit without a scrollbar. Past the cap the
  // block is centred rather than left hanging from the header.
  const rh = clamp((H - TOP - 2) / syms.length, 19, 70);
  const yTop = TOP + Math.max(0, (H - TOP - 2 - rh * syms.length) / 2);
  // right-hand columns are fixed width; the 24h range takes whatever is left
  const edgeR = W - (narrow ? 6 : 58), edgeL = edgeR - (narrow ? 62 : 86);
  const rngL = 116, rngR = edgeL - 12, rngW = rngR - rngL;
  const showRange = !narrow && rngW > 60;
  _boardGeom = { TOP: yTop, rh, syms };

  const mono = cssVar("--mono"), muted = cssVar("--muted");
  g.fillStyle = muted; g.font = "8.5px " + mono; g.textAlign = "center";
  if(showRange) g.fillText("24H RANGE", (rngL + rngR) / 2, HEAD - 3);
  g.fillText("EDGE vs THRESHOLD", (edgeL + edgeR) / 2, HEAD - 3);
  if(!narrow){ g.textAlign = "right"; g.fillText("P(WIN)", W - 6, HEAD - 3); }

  syms.forEach((sym, i) => {
    const s = es[sym]; if(!s) return;
    const y0 = yTop + i * rh, y = y0 + rh / 2;
    const b = s.brain || {}, held = posns[sym];
    if(i){ g.fillStyle = "rgba(255,255,255,.035)"; g.fillRect(8, y0 - 0.5, W - 16, 1); }
    if(sym === curSymbol){
      g.fillStyle = "rgba(0,210,255,.055)"; g.fillRect(0, y0, W, rh - 1);
      g.fillStyle = cssVar("--accent"); g.fillRect(0, y0, 2, rh - 1);
    } else if(i === _boardHit){
      g.fillStyle = "rgba(255,255,255,.035)"; g.fillRect(0, y0, W, rh - 1);
    }
    // regime dot
    const rc = REGIME_TINT[b.regime] || muted;
    g.fillStyle = rc; g.beginPath(); g.arc(9, y, 3, 0, 6.2832); g.fill();

    g.textAlign = "left";
    g.fillStyle = held ? cssVar("--ink") : cssVar("--ink-2");
    g.font = (held ? "700 " : "") + "11px " + mono;
    g.fillText(shortSym(sym) + (adopted.has(sym) ? " ◈" : ""), 18, y - 3);
    g.fillStyle = muted; g.font = "9.5px " + mono;
    g.fillText(fmt.px(s.price), 18, y + 9);
    if(held){   // side chip, so an open market is unmistakable in the list
      const up = held.side === "LONG";
      const col = up ? cssVar("--good") : cssVar("--bad");
      g.fillStyle = col + "26"; g.fillRect(78, y - 11, 15, 11);
      g.fillStyle = col; g.font = "700 8.5px " + mono; g.textAlign = "center";
      g.fillText(up ? "L" : "S", 85.5, y - 2.5);
    }

    if(showRange && Number.isFinite(s.hi24) && Number.isFinite(s.lo24) && s.hi24 > s.lo24){
      const rp = clamp(Number.isFinite(s.rpos24) ? s.rpos24
        : (s.price - s.lo24) / (s.hi24 - s.lo24), 0, 1);
      g.fillStyle = cssVar("--surface-3"); g.fillRect(rngL, y - 1.5, rngW, 3);
      // the low/high thirds are where mean-reversion and breakout live, so tint
      // the marker by which third the price is actually in
      const mc = rp > 0.72 ? cssVar("--good") : rp < 0.28 ? cssVar("--bad") : cssVar("--accent");
      const mx = rngL + rp * rngW;
      g.fillStyle = mc + "55"; g.fillRect(rngL, y - 1.5, rp * rngW, 3);
      g.fillStyle = mc; g.fillRect(mx - 1.5, y - 6, 3, 12);
      g.fillStyle = muted; g.font = "8px " + mono;
      g.textAlign = "left";  g.fillText(fmt.px(s.lo24), rngL, y + 14);
      g.textAlign = "right"; g.fillText(fmt.px(s.hi24), rngR, y + 14);
    }

    // Edge vs threshold, drawn in units of the threshold rather than in raw
    // edge: thresholds are ~0.3 and live edges spend their time inside that, so
    // an absolute -1..+1 bar rendered every symbol as a 4px stub. Pinning the
    // threshold tick at a fixed 62% makes "how close is this to firing" the
    // thing the bar is actually about, and an armed signal visibly crosses it.
    const thr = clamp(b.threshold || 0.3, 0.02, 1), edge = clamp(b.edge || 0, -1, 1);
    const cxE = (edgeL + edgeR) / 2, halfW = (edgeR - edgeL) / 2, TK = 0.62;
    g.fillStyle = cssVar("--surface-3"); g.fillRect(edgeL, y - 4, edgeR - edgeL, 8);
    g.fillStyle = cssVar("--baseline");
    g.fillRect(cxE - TK * halfW, y - 7, 1, 14);
    g.fillRect(cxE + TK * halfW, y - 7, 1, 14);
    g.fillRect(cxE, y - 5, 1, 10);
    const armed = Math.abs(edge) >= thr;
    const ew = Math.min(1, (Math.abs(edge) / thr) * TK) * halfW;
    const ec = edge >= 0 ? cssVar("--accent") : cssVar("--bad");
    if(armed){ g.fillStyle = ec + "33"; g.fillRect(edge >= 0 ? cxE : cxE - ew, y - 7, ew, 14); }
    g.fillStyle = armed ? ec : ec + "77";
    g.fillRect(edge >= 0 ? cxE : cxE - ew, y - 3, ew, 6);

    if(!narrow){
      const p = b.p_win;
      g.textAlign = "right"; g.font = "10.5px " + mono;
      g.fillStyle = p == null ? muted : p >= 0.55 ? cssVar("--good") : p >= 0.5 ? cssVar("--ink-2") : cssVar("--bad");
      g.fillText(p == null ? "—" : Math.round(p * 100) + "%", W - 6, y - 3);
      // The block reason is a full sentence ("edge -0.11 < thr 0.30") and only
      // its subject fits — the rung that refused. "edge" is suppressed because
      // the bar immediately to the left already draws exactly that distance,
      // and six rows all reading "edge" looks like a stuck render.
      const blk = String(s.entry_block || "").split(/[ :]/)[0].slice(0, 10);
      const st = held ? `${s.bars_held || 0} bars`
        : armed ? "armed" : (blk && blk !== "edge" ? blk : "scanning");
      g.fillStyle = held ? cssVar("--accent-2") : armed ? cssVar("--good") : muted;
      g.font = "8.5px " + mono;
      g.fillText(st, W - 6, y + 9);
    }
  });
}
function boardHover(ev){
  const host = $("viz-board"), tip = host.querySelector(".viz-tip");
  if(!_boardGeom){ tip.classList.remove("on"); return; }
  const r = ev.currentTarget.getBoundingClientRect();
  const i = Math.floor((ev.clientY - r.top - _boardGeom.TOP) / _boardGeom.rh);
  const sym = (i >= 0 && i < _boardGeom.syms.length) ? _boardGeom.syms[i] : null;
  if(i !== _boardHit){ _boardHit = sym ? i : -1; renderBoard(); }
  if(!sym){ tip.classList.remove("on"); return; }
  const s = S?.engine?.symbols?.[sym] || {}, b = s.brain || {};
  const held = S?.engine?.portfolio?.open_positions?.[sym];
  const rp = Number.isFinite(s.rpos24) ? Math.round(s.rpos24 * 100) + "% of 24h range" : "24h range unknown";
  tip.innerHTML = `<b>${esc(sym)}</b> ${fmt.px(s.price)}<br>${esc(b.regime || "—")} · ${rp}<br>`
    + `edge ${fmt.signed(b.edge || 0, 2)} vs thr ${(b.threshold || 0).toFixed(2)} · P ${fmt.pct(b.p_win, 0)}<br>`
    + (held ? `<b>${held.side}</b> ${held.leverage}x from ${fmt.px(held.entry)} · ${fmt.signed(held.upnl, 2)}`
            : esc(s.entry_block || "scanning"));
  tip.classList.add("on");
  tip.style.left = Math.min(r.width - tip.offsetWidth - 6, Math.max(4, ev.clientX - r.left + 12)) + "px";
  tip.style.top = Math.min(r.height - tip.offsetHeight - 4, ev.clientY - r.top + 12) + "px";
}

/* ------------------------------------------------------------ 2. TRADE RUNWAY
   Each open position drawn on the axis it actually lives on: stop on the left,
   target on the right, entry and the live mark in between. The positions table
   has all four numbers; only a picture tells you at a glance that price has
   given back two thirds of the way to the stop.
   When flat, the same canvas shows which symbol is closest to firing — the
   panel is then answering "why is nothing open?" instead of being blank. */
function renderRunway(){
  const c = _liveCtx("viz-runway", "Engine idle."); if(!c) return;
  const { g, W, H } = c;
  const pf = S.engine.portfolio, posns = pf?.open_positions || {};
  const list = Object.entries(posns);
  const mono = cssVar("--mono"), muted = cssVar("--muted");
  const cap = $("runway-cap");

  if(!list.length){
    if(cap) cap.textContent = "flat";
    // charging ladder: |edge| as a fraction of each symbol's own threshold
    const rows = engineSymbols().map(sym => {
      const b = S.engine.symbols?.[sym]?.brain || {};
      const thr = clamp(b.threshold || 0.3, 0.02, 1);
      return { sym, frac: Math.min(1.35, Math.abs(b.edge || 0) / thr), dir: (b.edge || 0) >= 0 };
    }).sort((a, b2) => b2.frac - a.frac).slice(0, 4);
    g.fillStyle = muted; g.font = "9.5px " + mono; g.textAlign = "left";
    g.fillText("no capital at risk — closest to firing:", 8, 14);
    const rh = Math.min(30, (H - 26) / Math.max(1, rows.length));
    rows.forEach((r, i) => {
      const y = 24 + i * rh + rh / 2, bx = 92, bw = W - bx - 46;
      g.fillStyle = cssVar("--ink-2"); g.font = "10.5px " + mono; g.textAlign = "left";
      g.fillText(shortSym(r.sym), 8, y + 3.5);
      g.fillStyle = cssVar("--surface-3"); g.fillRect(bx, y - 4, bw, 8);
      const full = bx + bw / 1.35;                 // where frac == 1.0 lands
      const col = r.frac >= 1 ? cssVar("--good") : r.dir ? cssVar("--accent") : cssVar("--bad");
      g.fillStyle = col + (r.frac >= 1 ? "" : "aa");
      g.fillRect(bx, y - 4, (r.frac / 1.35) * bw, 8);
      g.fillStyle = cssVar("--baseline"); g.fillRect(full, y - 7, 1, 14);
      g.fillStyle = r.frac >= 1 ? cssVar("--good") : muted; g.font = "9.5px " + mono;
      g.textAlign = "right"; g.fillText(Math.round(r.frac * 100) + "%", W - 6, y + 3.5);
    });
    g.fillStyle = muted; g.font = "8.5px " + mono; g.textAlign = "left";
    if(rows.length) g.fillText("100% = threshold", 92, H - 2);
    return;
  }

  if(cap) cap.textContent = `${list.length} open`;
  // one position must not sit in the top third of an otherwise empty panel:
  // rows take a sane height and the BLOCK is centred in whatever is left
  const rh = clamp(H / list.length, 44, 72);
  const y00 = Math.max(0, (H - rh * list.length) / 2);
  list.forEach(([sym, p], i) => {
    const y0 = y00 + i * rh, mark = S.engine.symbols?.[sym]?.price || p.entry;
    const up = p.side === "LONG", sgn = up ? 1 : -1;
    // 1R is the stop distance AT ENTRY, not the current one. A trailing stop
    // ratchets toward and then past entry, so |entry - current stop| collapses
    // and the R printed off it climbs on its own with price standing still.
    const risk = p.init_risk > 0 ? p.init_risk : Math.abs(p.entry - p.stop);
    const r = risk > 1e-12 ? ((mark - p.entry) * sgn) / risk : 0;
    // the stop has moved through entry: the trade can no longer lose
    const locked = risk > 1e-12 && (p.stop - p.entry) * sgn > 0;
    // no fixed target means the exit is a trail — there is no price to draw, so
    // the right edge becomes "3R and still running" and is marked as open.
    const trailing = !(p.tp > 0);
    const target = trailing ? p.entry + sgn * risk * 3 : p.tp;
    const span = target - p.stop;
    const fr = (v) => span === 0 ? 0 : (v - p.stop) / span;
    const padL = 10, padR = 10, tw = W - padL - padR;
    const X = (v) => padL + clamp(fr(v), -0.02, 1.02) * tw;
    const ty = y0 + rh - 16;

    g.textAlign = "left"; g.font = "700 10.5px " + mono;
    g.fillStyle = up ? cssVar("--good") : cssVar("--bad");
    g.fillText(`${shortSym(sym)} ${p.side} ${p.leverage}x`, padL, y0 + 13);
    g.font = "9.5px " + mono; g.fillStyle = muted;
    g.fillText(`from ${fmt.px(p.entry)}`, padL + 108, y0 + 13);
    g.textAlign = "right"; g.font = "10.5px " + mono;
    g.fillStyle = r >= 0 ? cssVar("--good") : cssVar("--bad");
    g.fillText(`${fmt.signed(r, 2)}R  ${fmt.signed(p.upnl || 0, 2)}`, W - padR, y0 + 13);

    g.fillStyle = cssVar("--surface-3"); g.fillRect(padL, ty - 3, tw, 6);
    // the travelled part: entry -> mark, coloured by which way it went
    const xe = X(p.entry), xm = X(mark);
    g.fillStyle = (r >= 0 ? cssVar("--good") : cssVar("--bad")) + "cc";
    g.fillRect(Math.min(xe, xm), ty - 3, Math.abs(xm - xe), 6);
    g.fillStyle = locked ? cssVar("--good") : cssVar("--bad");
    g.fillRect(padL, ty - 7, 2, 14);                                         // stop
    g.fillStyle = cssVar("--baseline"); g.fillRect(xe - 0.5, ty - 6, 1, 12); // entry
    if(trailing){                                                            // open end
      g.fillStyle = cssVar("--good") + "66";
      for(let x = W - padR - 8; x < W - padR; x += 4) g.fillRect(x, ty - 5, 2, 10);
    } else { g.fillStyle = cssVar("--good"); g.fillRect(W - padR - 2, ty - 7, 2, 14); }
    g.fillStyle = cssVar("--ink"); g.beginPath(); g.arc(xm, ty, 3.6, 0, 6.2832); g.fill();

    // only two labels under the track. A third (the entry price, centred on its
    // tick) collided with "stop ..." the moment a trailing stop pushed the
    // entry tick to the left edge — which is exactly when a winner is running.
    g.font = "8.5px " + mono;
    g.textAlign = "left";
    g.fillStyle = locked ? cssVar("--good") + "cc" : muted;
    g.fillText("stop " + fmt.px(p.stop) + (locked ? " · locked in" : ""), padL, ty + 15);
    g.textAlign = "right"; g.fillStyle = muted;
    g.fillText(trailing ? "trailing · no fixed target" : "tp " + fmt.px(p.tp), W - padR, ty + 15);
  });
}

/* -------------------------------------------------------------- 3. ORDER FLOW
   Book imbalance and trade-flow imbalance are the fastest-moving real numbers
   the system has, and the dashboard showed them as one line of text. Here they
   are a rolling ribbon of the last ~50 seconds, plus a per-symbol strip so the
   watchlist's pressure is comparable at a glance.
   The history is client-side by construction: the server ships the CURRENT
   reading, and the ribbon is just what we have been shown since page load. */
const FLOW_CAP = 220;
const _flowHist = new Map();
function flowPush(sym, m){
  if(!m) return;
  let h = _flowHist.get(sym);
  if(!h){ h = { obi: [], flow: [] }; _flowHist.set(sym, h); }
  const o = Number(m.obi), f = Number(m.flow);
  h.obi.push(Number.isFinite(o) ? clamp(o, -1, 1) : 0);
  h.flow.push(Number.isFinite(f) ? clamp(f, -1, 1) : 0);
  if(h.obi.length > FLOW_CAP){ h.obi.shift(); h.flow.shift(); }
}
function renderFlow(){
  const c = _liveCtx("viz-flow", "Engine idle."); if(!c) return;
  const { host, g, W, H } = c;
  const sym = curSymbol, s = S.engine.symbols?.[sym];
  const m = s?.micro || {};
  const mono = cssVar("--mono"), muted = cssVar("--muted");
  const cap = $("flow-cap");
  if(cap) cap.textContent = sym ? shortSym(sym) : "—";

  // readout strip
  const rd = [
    ["OBI", fmt.signed(m.obi ?? 0, 2), (m.obi ?? 0) >= 0],
    ["FLOW", fmt.signed(m.flow ?? 0, 2), (m.flow ?? 0) >= 0],
    ["CVD", fmt.signed(m.cvd_slope ?? 0, 2), (m.cvd_slope ?? 0) >= 0],
    ["SPRD", (m.spread_bps ?? 0).toFixed(1) + "bp", null],
    ["TICKS", (m.ticks_per_s ?? 0).toFixed(1) + "/s", null],
  ];
  const cw = W / rd.length;
  rd.forEach(([k, v, good], i) => {
    const x = i * cw + cw / 2;
    g.textAlign = "center"; g.font = "8px " + mono; g.fillStyle = muted;
    g.fillText(k, x, 9);
    g.font = "11px " + mono;
    g.fillStyle = good == null ? cssVar("--ink-2") : good ? cssVar("--good") : cssVar("--bad");
    g.fillText(v, x, 22);
  });

  const h = _flowHist.get(sym);
  const n = h ? h.obi.length : 0;
  const top = 32, botStrip = 52, bot = H - botStrip;
  const midY = (top + bot) / 2, half = (bot - top) / 2 - 2;
  // AUTO-SCALED. Both series are bounded at +/-1 but live around +/-0.2, so a
  // fixed +/-1 axis drew a flat line through the middle of an empty box and the
  // shape of the pressure — the entire point of the view — was invisible. The
  // axis follows the window's own peak and is labelled, so nothing is hidden.
  let peak = 0.12;
  for(let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(h.obi[i]), Math.abs(h.flow[i]));
  peak = Math.min(1, peak * 1.15);
  const amp = half / peak;

  g.strokeStyle = "rgba(255,255,255,.045)"; g.lineWidth = 1;
  for(const f of [-0.5, 0.5]){
    const y = Math.round(midY - f * half) + .5;
    g.beginPath(); g.moveTo(0, y); g.lineTo(W, y); g.stroke();
  }
  g.strokeStyle = cssVar("--baseline");
  g.beginPath(); g.moveTo(0, Math.round(midY) + .5); g.lineTo(W, Math.round(midY) + .5); g.stroke();
  g.fillStyle = muted; g.font = "8px " + mono; g.textAlign = "right";
  g.fillText("±" + peak.toFixed(2), W - 4, top + 8);
  if(n < 2){
    g.fillStyle = muted; g.font = "10px " + mono; g.textAlign = "center";
    g.fillText("listening to the tape…", W / 2, midY + 3);
  } else {
    // The window is "everything we have been shown", drawn across the full
    // width — anchoring x to the 220-sample capacity instead left a fresh page
    // with its ribbon crushed into the right eighth of the panel, which reads
    // as a broken chart rather than as a young one. The axis is labelled with
    // the real elapsed span so the compression stays honest.
    const X = (i) => n < 2 ? 0 : (i / (n - 1)) * W;
    const area = (col, positive) => {
      g.beginPath(); g.moveTo(X(0), midY);
      for(let i = 0; i < n; i++){
        const v = h.obi[i];
        g.lineTo(X(i), midY - (positive ? Math.max(0, v) : Math.min(0, v)) * amp);
      }
      g.lineTo(X(n - 1), midY); g.closePath();
      g.fillStyle = col; g.fill();
    };
    area(cssVar("--good") + "3a", true);
    area(cssVar("--bad") + "3a", false);
    const line = (arr, col, lw) => {
      g.strokeStyle = col; g.lineWidth = lw; g.beginPath();
      for(let i = 0; i < n; i++){
        const x = X(i), y = midY - arr[i] * amp;
        i ? g.lineTo(x, y) : g.moveTo(x, y);
      }
      g.stroke();
    };
    line(h.obi, cssVar("--accent"), 1.4);
    line(h.flow, cssVar("--warn") + "99", 1);
    const ly = clamp(midY - h.obi[n - 1] * amp, top, bot);
    g.fillStyle = cssVar("--accent"); g.beginPath(); g.arc(W - 3, ly, 3, 0, 6.2832); g.fill();
    // legend and the bid/ask sense, both anchored to the zero line so they
    // annotate the axis instead of floating in whatever half is empty
    g.font = "8px " + mono; g.textAlign = "left";
    g.fillStyle = cssVar("--accent"); g.fillText("book imbalance", 4, top + 8);
    g.fillStyle = cssVar("--warn") + "99"; g.fillText("trade flow", 82, top + 8);
    g.fillStyle = cssVar("--good") + "99"; g.fillText("bid", 4, midY - 4);
    g.fillStyle = cssVar("--bad") + "99";  g.fillText("ask", 4, midY + 11);
    g.textAlign = "right"; g.fillStyle = muted;
    g.fillText(`last ${fmt.dur(Math.round(n * 0.25))}`, W - 4, bot - 2);
  }

  // per-symbol pressure strip: same reading, every market, comparable
  const syms = engineSymbols(), sw = W / Math.max(1, syms.length);
  const sy = H - botStrip + 22;
  g.fillStyle = muted; g.font = "8px " + mono; g.textAlign = "left";
  g.fillText("BOOK PRESSURE — ALL MARKETS", 4, H - botStrip + 12);
  syms.forEach((sm, i) => {
    const o = clamp(S.engine.symbols?.[sm]?.micro?.obi ?? 0, -1, 1);
    const x0 = i * sw + 6, bw = sw - 12, cx = x0 + bw / 2;
    g.fillStyle = cssVar("--surface-3"); g.fillRect(x0, sy, bw, 6);
    g.fillStyle = (o >= 0 ? cssVar("--good") : cssVar("--bad")) + (sm === curSymbol ? "" : "88");
    g.fillRect(o >= 0 ? cx : cx + (o * bw) / 2, sy, Math.abs(o) * bw / 2, 6);
    g.fillStyle = cssVar("--baseline"); g.fillRect(cx, sy - 2, 1, 10);
    g.fillStyle = sm === curSymbol ? cssVar("--accent-2") : muted;
    g.font = "8.5px " + mono; g.textAlign = "center";
    g.fillText(shortSym(sm), cx, sy + 16);
  });
}

/* ------------------------------------------------------------- 4. RISK BUDGET
   How much room is left before the machine stops itself. The kill switch, the
   exposure cap and the slot cap are all real limits that end trading for the
   day, and nothing on screen showed the distance to any of them. */
function _arc(g, cx, cy, r, frac, col, big, sub, label){
  const A0 = Math.PI * 0.75, A1 = Math.PI * 2.25;
  const mono = cssVar("--mono");
  g.lineWidth = 9; g.lineCap = "butt";
  g.strokeStyle = cssVar("--surface-3");
  g.beginPath(); g.arc(cx, cy, r, A0, A1); g.stroke();
  if(frac > 0){
    g.strokeStyle = col;
    g.beginPath(); g.arc(cx, cy, r, A0, A0 + (A1 - A0) * clamp(frac, 0, 1)); g.stroke();
  }
  g.textAlign = "center";
  g.fillStyle = col; g.font = "700 16px " + mono; g.fillText(big, cx, cy + 2);
  g.fillStyle = cssVar("--muted"); g.font = "8.5px " + mono;
  g.fillText(sub, cx, cy + 15);
  g.fillStyle = cssVar("--ink-2"); g.font = "9px " + mono;
  g.fillText(label, cx, cy + r + 15);
}
function renderBudget(){
  const c = _liveCtx("viz-budget", "Engine idle."); if(!c) return;
  const { g, W, H } = c;
  const risk = S.engine.risk || {}, pf = S.engine.portfolio || {};
  const rc = S.config?.risk || {};
  const mono = cssVar("--mono"), muted = cssVar("--muted");
  const eq = pf.equity || 0;

  // Day-loss budget, measured the way the kill switch measures it: REALIZED
  // PnL against the day's opening equity (manager.py `dd = -day_realized /
  // day_start_equity`). Driving this off live equity instead would have been
  // smoother and wrong — an open winner would show headroom the switch does
  // not credit, and an open loser would show a halt that is not coming.
  const dayStart = risk.day_start_equity > 0 ? risk.day_start_equity : (pf.starting_balance || eq);
  const budget = (rc.max_daily_loss_pct || 0.05) * dayStart;
  const usedAbs = Math.max(0, -(risk.day_realized || 0));
  const usedFrac = budget > 0 ? clamp(usedAbs / budget, 0, 1) : 0;
  const leftFrac = 1 - usedFrac;

  const posns = Object.entries(pf.open_positions || {});
  let notional = 0;
  for(const [sym, p] of posns) notional += Math.abs((p.qty || 0) * (S.engine.symbols?.[sym]?.price || p.entry || 0));
  const lev = eq > 0 ? notional / eq : 0;
  const levCap = rc.max_net_exposure || 2.5;
  const slots = rc.max_open_positions || 3;

  const cap = $("budget-cap");
  if(cap) cap.textContent = risk.killed ? "HALTED" : `${fmt.signed(risk.day_realized || 0, 2)} today`;

  const r = Math.min(38, (H - 78) / 2), cy = 8 + r + 4, cw = W / 3;
  // the arc shows what is LEFT, not what is spent — a healthy day then reads as
  // a full ring draining toward the halt, which is the question being asked
  const dayCol = leftFrac <= 0.2 ? cssVar("--bad") : leftFrac <= 0.5 ? cssVar("--warn") : cssVar("--good");
  _arc(g, cw * 0.5, cy, r, leftFrac, dayCol, fmt.usd(budget - usedAbs, 0),
       Math.round(leftFrac * 100) + "% of " + fmt.usd(budget, 0), "day-loss headroom");
  const levCol = lev >= levCap * 0.9 ? cssVar("--bad") : lev > 0 ? cssVar("--accent") : muted;
  _arc(g, cw * 1.5, cy, r, levCap > 0 ? lev / levCap : 0, levCol, lev.toFixed(2) + "x",
       "cap " + levCap.toFixed(1) + "x", "net exposure");
  const slotCol = posns.length >= slots ? cssVar("--warn") : posns.length ? cssVar("--accent") : muted;
  _arc(g, cw * 2.5, cy, r, slots > 0 ? posns.length / slots : 0, slotCol,
       `${posns.length}/${slots}`, fmt.usd(notional, 0), "position slots");

  // bottom row: the two limits that pause trading rather than end the day
  const by = H - 26;
  g.strokeStyle = cssVar("--grid"); g.lineWidth = 1;
  g.beginPath(); g.moveTo(6, by - 12.5); g.lineTo(W - 6, by - 12.5); g.stroke();
  g.textAlign = "left"; g.font = "8.5px " + mono; g.fillStyle = muted;
  g.fillText("LOSS STREAK", 6, by);
  const maxL = Math.max(1, rc.max_consecutive_losses || 8), lost = risk.consecutive_losses || 0;
  const pw = Math.min(9, (W * 0.34 - 76) / maxL);
  for(let i = 0; i < maxL; i++){
    g.fillStyle = i < lost ? (lost >= maxL - 2 ? cssVar("--bad") : cssVar("--warn")) : cssVar("--surface-3");
    g.fillRect(74 + i * (pw + 2), by - 7, pw, 7);
  }
  g.textAlign = "right"; g.font = "9px " + mono;
  const cool = risk.cooldown_s || 0;
  g.fillStyle = cool > 0 ? cssVar("--warn") : muted;
  g.fillText(cool > 0 ? "cooldown " + fmt.dur(cool) : `${risk.trades_today || 0} trades today`, W - 6, by);
  g.fillStyle = muted; g.font = "8.5px " + mono;
  g.fillText("dd " + fmt.pct(risk.health?.drawdown || 0, 1)
    + " · risk " + Math.round((risk.health?.scalar ?? 1) * 100) + "%", W - 6, by + 11);
  if(risk.killed){
    g.fillStyle = "rgba(255,45,120,.12)"; g.fillRect(0, 0, W, H);
    g.fillStyle = cssVar("--bad"); g.font = "700 13px " + mono; g.textAlign = "center";
    g.fillText("KILL SWITCH ENGAGED", W / 2, H / 2 + 4);
  }
}

/* one entry point for the four, called from both render paths */
function renderLiveViews(){
  if(!S) return;
  renderBoard(); renderRunway(); renderFlow(); renderBudget();
}

/* board interaction — bound once */
(() => {
  const host = $("viz-board"); if(!host) return;
  const cv = host.querySelector("canvas");
  cv.style.cursor = "pointer";
  cv.addEventListener("pointermove", boardHover);
  cv.addEventListener("pointerleave", () => {
    host.querySelector(".viz-tip").classList.remove("on");
    if(_boardHit !== -1){ _boardHit = -1; renderBoard(); }
  });
  cv.addEventListener("click", (ev) => {
    if(!_boardGeom) return;
    const r = cv.getBoundingClientRect();
    const i = Math.floor((ev.clientY - r.top - _boardGeom.TOP) / _boardGeom.rh);
    const sym = (i >= 0 && i < _boardGeom.syms.length) ? _boardGeom.syms[i] : null;
    if(!sym) return;
    autoFollow = false; $("auto-follow-btn")?.classList.remove("on");
    setSymbol(sym); renderLiveViews();
  });
  // the four live views share one observer; any of them resizing means the
  // column resized, so invalidate and redraw the set
  const ro = new ResizeObserver(() => {
    document.querySelectorAll(".viz canvas").forEach(vizInvalidate);
    renderLiveViews();
  });
  for(const id of ["viz-board", "viz-runway", "viz-flow", "viz-budget"]){
    const el = $(id); if(el) ro.observe(el);
  }
})();
