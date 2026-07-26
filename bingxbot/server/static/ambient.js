/* ==========================================================================
   PULSE — ambient layer

   One WebGL2 fullscreen pass behind the whole terminal: a receding grid, drift,
   scanlines, grain, vignette and chromatic edge bleed, all driven by what the
   machine is actually doing (regime, conviction, armed, halted).

   Why a shader and not CSS or a DOM overlay:

     · The CRT scanlines and the header's hologram sweep were BOTH removed from
       this dashboard earlier because a fixed full-viewport element sits in
       every display list the browser rebuilds — profiling put that churn at the
       top of main-thread cost. In a fragment shader the same look is a few
       instructions on pixels the GPU was going to touch anyway, and the main
       thread never sees it.
     · It is ONE draw call. No geometry buffer, no vertex attributes: the vertex
       stage builds a single oversized triangle from gl_VertexID, which covers
       the viewport with no clipping seam and no per-frame uploads.
     · It renders at a FRACTION of device resolution and is stretched back up by
       CSS. Full-screen effects are fill-rate bound, so resolution is the cheap
       knob; at 0.55 scale we shade a third of the pixels and nobody can tell,
       because everything in here is low-frequency.

   The budget is enforced, not assumed: the layer measures its own frame times
   and walks the resolution down (and finally stops animating) if it cannot hold
   its share. It is decoration — it must yield to the terminal, never compete
   with it.
   ========================================================================== */
(() => {
  "use strict";

  const CANVAS_ID = "ambient";
  const cv = document.getElementById(CANVAS_ID);
  if (!cv) return;

  // Quality ladder, walked down when frames get expensive and back up when
  // they are cheap again. Resolution first, because that is where the cost is.
  // Wallpaper does not need 60fps. Every term in this shader is low-frequency
  // drift; at 30 half the frames are shaded for a look nobody can tell apart,
  // and the fill-rate bill halves with them. Resolution is the second knob and
  // the steeper one, since the pass is fill-rate bound end to end.
  const TIERS = [
    { scale: 0.55, fps: 30, grain: 1.0 },
    { scale: 0.45, fps: 30, grain: 0.8 },
    { scale: 0.36, fps: 20, grain: 0.0 },
    { scale: 0.28, fps: 12, grain: 0.0 },
  ];
  let tier = 0;

  const gl = cv.getContext("webgl2", {
    alpha: true, antialias: false, depth: false, stencil: false,
    powerPreference: "low-power",       // this is wallpaper, not the workload
    preserveDrawingBuffer: false,
  });
  if (!gl) return;                      // no WebGL2: the CSS background stands alone

  // A fullscreen fragment shader is nearly free on a GPU and ruinous without
  // one. Chrome silently falls back to SwiftShader — a CPU rasterizer — on
  // blocklisted drivers, in VMs, over remote desktop and in headless. Measured
  // there, this layer took the page from p50 24ms with zero long tasks to p50
  // 60ms with 137 of them: every shaded pixel was competing with the terminal
  // for the same cores.
  //
  // There is no adaptive-quality ladder steep enough to make software
  // rasterization of a full viewport a good idea, so this is not a tier — it is
  // a refusal. The CSS backdrop stands in and nobody loses a frame.
  // `?ambient=1` forces it on (a correct GPU reported under an unfamiliar name,
  // or taking a screenshot on a machine that has none); `?ambient=0` forces it
  // off. Absent the parameter the detection decides.
  const FORCE = new URLSearchParams(location.search).get("ambient");
  const dbg = gl.getExtension("WEBGL_debug_renderer_info");
  const RENDERER = dbg ? String(gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) || "") : "";
  const SOFTWARE = /swiftshader|llvmpipe|softpipe|software|basic render|microsoft basic/i
    .test(RENDERER);
  if (FORCE === "0" || (SOFTWARE && FORCE !== "1")) {
    cv.remove();
    document.documentElement.classList.add("no-ambient");
    if (SOFTWARE) console.info("ambient: software renderer (%s) — layer disabled", RENDERER);
    return;
  }

  /* ------------------------------------------------------------- shaders */

  // No attributes. gl_VertexID 0,1,2 -> a triangle that covers clip space.
  const VS = `#version 300 es
void main(){
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

  const FS = `#version 300 es
precision mediump float;
out vec4 frag;

uniform vec2  uRes;        // drawing-buffer size, px
uniform float uTime;       // seconds
uniform vec3  uTint;       // regime colour
uniform float uEdge;       // 0..1 conviction — drives grid brightness + drift
uniform float uArmed;      // 0..1 eased "a signal is at the gate"
uniform float uAlarm;      // 0..1 kill switch / halted
uniform float uGrain;      // 0..1 film grain amount

// cheap value noise — two texture-free lookups, no derivatives
float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float vnoise(vec2 p){
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1,0)), f.x),
             mix(hash(i + vec2(0,1)), hash(i + vec2(1,1)), f.x), f.y);
}

// perspective floor grid receding to a horizon — the y2k/arcade signature.
// Distance to the nearest line in each axis, widened by fwidth so the lines
// stay one pixel wide at every depth instead of aliasing into moire.
float grid(vec2 uv, float cell){
  vec2 g = abs(fract(uv * cell - 0.5) - 0.5) / fwidth(uv * cell);
  return 1.0 - min(min(g.x, g.y), 1.0);
}

// The horizon sits low, not centred. Mirroring the grid around the middle of
// the viewport made the two halves converge into a starburst that cut across
// the panels — it read as diagonal streaks, not as a floor. One floor below a
// low horizon and open sky above it is the composition this look actually is.
const float HORIZON = 0.34;

void main(){
  vec2 uv = gl_FragCoord.xy / uRes;
  vec2 p  = (gl_FragCoord.xy - 0.5 * uRes) / uRes.y;   // aspect-correct, centred

  vec3 col = vec3(0.0);
  float below = HORIZON - uv.y;                        // >0 on the floor

  // --- floor: a grid receding to the horizon, drifting toward the viewer at a
  //     speed set by how convinced the machine currently is
  if(below > 0.0){
    float depth = below + 0.012;
    vec2  gp    = vec2(p.x / depth * 0.55,
                       0.16 / depth + uTime * (0.04 + 0.13 * uEdge));
    float lines = grid(gp, 1.0);
    // the far field dissolves rather than aliasing into moire
    float fade  = smoothstep(0.0, 0.10, below) * smoothstep(0.34, 0.03, below);
    col += uTint * lines * fade * (0.95 + 1.20 * uEdge);
  }

  // --- sky: slow plasma drift, two octaves is plenty at this scale
  float sky = smoothstep(-0.02, 0.35, uv.y - HORIZON);
  float n = vnoise(p * 2.2 + vec2(uTime * 0.035, uTime * 0.021)) * 0.65
          + vnoise(p * 5.1 - vec2(uTime * 0.018, uTime * 0.030)) * 0.35;
  col += uTint * pow(n, 2.0) * 0.30 * sky;

  // --- the horizon itself, blooming
  col += uTint * exp(-abs(uv.y - HORIZON) * uRes.y * 0.028) * (0.55 + 0.95 * uArmed);

  // --- armed sweep: a band travelling up the page, replacing the header
  //     hologram that had to be deleted for costing a full-width repaint
  float sweep = fract(uTime * 0.13);
  col += uTint * exp(-abs(uv.y - sweep) * 30.0) * uArmed * 0.32;

  // --- CRT scanlines + a slow rolling bar
  float sl   = 0.5 + 0.5 * sin(gl_FragCoord.y * 1.55);
  float roll = 0.5 + 0.5 * sin(uv.y * 5.0 - uTime * 0.55);
  col *= 1.0 - 0.16 * sl - 0.05 * roll;

  // --- chromatic bleed at the edges only, where a CRT actually smears
  float r2 = dot(p, p);
  col.r *= 1.0 + 0.16 * r2;
  col.b *= 1.0 + 0.24 * r2;

  // --- halted: the whole room goes red and breathes
  col = mix(col, vec3(0.75, 0.06, 0.22) * (0.35 + 0.30 * sin(uTime * 2.6)),
            uAlarm * 0.55);

  // --- vignette, then grain (grain last so it is not tinted)
  col *= smoothstep(1.45, 0.25, r2 * 1.6);
  col += (hash(gl_FragCoord.xy + fract(uTime) * 91.7) - 0.5) * 0.030 * uGrain;

  // OPAQUE, and it carries the page's base colour itself.
  //
  // The first version used brightness as alpha over SRC_ALPHA blending, which
  // multiplies a dark colour by its own small alpha — 0.05 became 0.004 and the
  // entire layer was invisible. Owning the backdrop outright removes the whole
  // question, costs one less blend per pixel, and means the CSS gradients this
  // replaces can go.
  frag = vec4(vec3(0.016, 0.020, 0.039) + max(col, 0.0), 1.0);
}`;

  function compile(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error("ambient shader:", gl.getShaderInfoLog(s));
      gl.deleteShader(s);
      return null;
    }
    return s;
  }

  const vs = compile(gl.VERTEX_SHADER, VS), fs = compile(gl.FRAGMENT_SHADER, FS);
  if (!vs || !fs) return;
  const prog = gl.createProgram();
  gl.attachShader(prog, vs); gl.attachShader(prog, fs); gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    console.error("ambient link:", gl.getProgramInfoLog(prog));
    return;
  }
  gl.useProgram(prog);
  const U = {};
  for (const n of ["uRes", "uTime", "uTint", "uEdge", "uArmed", "uAlarm", "uGrain"]) {
    U[n] = gl.getUniformLocation(prog, n);
  }
  gl.disable(gl.DEPTH_TEST);
  gl.disable(gl.BLEND);       // the pass is opaque and owns the backdrop
  const vao = gl.createVertexArray();   // required in WebGL2 even with no attributes
  gl.bindVertexArray(vao);

  /* --------------------------------------------------------------- state */

  const REGIME_RGB = {
    TREND_UP:   [0.00, 0.88, 0.63],
    TREND_DOWN: [1.00, 0.24, 0.50],
    RANGE:      [0.62, 0.42, 1.00],
    VOLATILE:   [1.00, 0.79, 0.24],
  };
  const DEFAULT_RGB = [0.00, 0.82, 1.00];

  // eased targets — the background must never snap, it drifts toward the truth
  const S = { tint: DEFAULT_RGB.slice(), edge: 0, armed: 0, alarm: 0 };
  const T = { tint: DEFAULT_RGB.slice(), edge: 0, armed: 0, alarm: 0 };

  /** Called by app.js on every state push. Pure numbers — no DOM, no layout. */
  window.ambientState = (st) => {
    if (!st) return;
    const rgb = REGIME_RGB[st.regime] || DEFAULT_RGB;
    T.tint = rgb;
    T.edge = Math.max(0, Math.min(1, st.edge || 0));
    T.armed = st.armed ? 1 : 0;
    T.alarm = st.alarm ? 1 : 0;
  };

  let W = 0, H = 0, scale = TIERS[0].scale;
  function resize() {
    const d = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(window.innerWidth * d * scale));
    const h = Math.max(1, Math.round(window.innerHeight * d * scale));
    if (w === W && h === H) return;
    W = w; H = h;
    cv.width = W; cv.height = H;
    gl.viewport(0, 0, W, H);
    gl.uniform2f(U.uRes, W, H);
  }
  resize();
  addEventListener("resize", resize, { passive: true });

  /* ---------------------------------------------------------------- loop */

  // The layer polices its own budget — and the FIRST version of this policing
  // did not work, which is the whole reason the loop below looks like it does.
  //
  // It timed performance.now() around gl.drawArrays and backed off when that
  // exceeded 3ms. Measured A/B, the layer took the page from p50 24ms with zero
  // long tasks to p50 60ms with 137 long tasks — while its own metric read
  // 0.047ms and it happily stayed at full quality. WebGL submission is
  // asynchronous: drawArrays queues a command and returns. The shading lands
  // later, off this thread, and never appears in a timer wrapped around the
  // call. The meter was measuring the postman, not the parcel.
  //
  // So the signal is now the only one that cannot lie about total cost: the
  // page's actual frame cadence. If frames are not landing near the display's
  // rate, something is too expensive, and the decoration is what gives way —
  // whether or not the decoration is what caused it.
  const SLOW_MS = 22.0;           // sustained frame interval that means trouble
  const GOOD_MS = 18.0;           // ...and one that means there is room again
  let last = 0, frameAvg = 16.7, slow = 0, fast = 0, t0 = performance.now();
  let prevFrame = performance.now();

  function setTier(next) {
    tier = next; slow = 0; fast = 0;
    scale = TIERS[tier].scale; W = H = 0; resize();
  }

  function step(now) {
    requestAnimationFrame(step);

    // cadence is sampled on EVERY animation frame, not only the ones we draw,
    // so throttling the layer to 30fps does not blind it to a 60fps page
    const gap = now - prevFrame; prevFrame = now;
    if (gap > 0 && gap < 200) frameAvg = frameAvg * 0.92 + gap * 0.08;

    if (document.hidden) { last = now; return; }

    if (frameAvg > SLOW_MS) {
      if (++slow > 90 && tier < TIERS.length - 1) setTier(tier + 1);
      else if (slow > 90 && tier === TIERS.length - 1) {
        // the bottom tier is still not enough: stop drawing entirely and leave
        // the last frame on screen. A still gradient is a fine wallpaper; a
        // terminal that drops inputs is not.
        cv.style.opacity = "0.55";
        return;
      }
    } else if (frameAvg < GOOD_MS) {
      if (++fast > 900 && tier > 0) setTier(tier - 1);
    }

    const minGap = 1000 / TIERS[tier].fps;
    if (now - last < minGap - 0.5) return;
    const dt = Math.min(0.1, (now - last) / 1000) || 0.016;
    last = now;

    // ease state toward its target; k is per-second so it is frame-rate free
    const k = 1 - Math.exp(-dt * 3.2);
    for (let i = 0; i < 3; i++) S.tint[i] += (T.tint[i] - S.tint[i]) * k;
    S.edge += (T.edge - S.edge) * k;
    S.armed += (T.armed - S.armed) * k;
    S.alarm += (T.alarm - S.alarm) * k;

    gl.uniform1f(U.uTime, (performance.now() - t0) / 1000);
    gl.uniform3f(U.uTint, S.tint[0], S.tint[1], S.tint[2]);
    gl.uniform1f(U.uEdge, S.edge);
    gl.uniform1f(U.uArmed, S.armed);
    gl.uniform1f(U.uAlarm, S.alarm);
    gl.uniform1f(U.uGrain, TIERS[tier].grain);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
  requestAnimationFrame(step);

  // exposed for the profiler harness, not for the app
  window.__ambient = () => ({ tier, scale, frameAvg: +frameAvg.toFixed(2),
                              w: W, h: H, fps: TIERS[tier].fps, renderer: RENDERER });
})();
