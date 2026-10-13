"""An animated, interactive replay of the student's settling-column test (HTML/canvas component).

How the particles move (so the picture is the student's data, not a cartoon):
each particle has a fixed rank p, the percentage of particles that settle faster than it. A particle of rank p rides
the p % isoremoval curve: its depth at time t is its starting depth plus the depth of that curve at t. Because the
curves bend for flocculent settling, every particle speeds up as it falls, and fast particles overtake slow ones.
With particles spread evenly at the start, the share still above any port at any time equals 1 - removal there,
exactly (removal at depth z is the share of ranks whose curve has already passed z). Particles that reach the floor
join the sludge; the share on the floor is the mean over ranks of min(1, curve depth / H), which tracks the tool's
overall removal (area method). Sampling moments flash the ports and fill in the table; optional lines show where
each removal level is, and one highlighted particle can be followed along its curve.
"""
import json

import numpy as np


VERSION = "3-setup-intro"   # bump when the animation changes (part of the page cache key)
RANKS = np.arange(0.5, 100, 1.0)   # one particle "rank" per percent of the particles


def rank_depths(cv, grid):
    """Depth of each rank's curve at each grid time; values past the floor are capped just below 1.5 H."""
    out = []
    for p in RANKS:
        row = []
        for t in grid:
            z = cv.depth_at(p, t)
            if np.isnan(z):
                z = 0.0
            row.append(round(float(min(z, 1.49 * cv.H)), 4))
        out.append(np.maximum.accumulate(row).tolist())   # a particle never moves back up
    return out


def floor_share(D, H):
    """Share of particles on the floor at each grid time (evenly spread start)."""
    D = np.asarray(D, float)
    return (np.minimum(D / H, 1.0).mean(axis=0) * 100).round(2).tolist()


def build(cv, depths, times, readings, c0, unit, munit, levels=()):
    step = 0.5
    grid = np.arange(0, cv.t_last + 1e-9, step)
    D = rank_depths(cv, grid)
    rem = [[round(float(np.interp(t, cv.t_aug, row)), 2) for t in grid] for row in cv.R_aug]
    lv = {}
    for p in levels:
        zs = []
        for t in grid:
            z = cv.depth_at(p, t)
            zs.append(round(float(z), 4) if np.isfinite(z) and 0 <= z <= cv.H else None)
        lv[str(p)] = zs
    data = {"grid": grid.round(3).tolist(), "step": step, "rem": rem, "depths": [float(d) for d in depths],
            "times": [float(t) for t in times], "readings": np.round(readings, 3).tolist(), "c0": float(c0),
            "unit": unit, "munit": munit, "H": float(cv.H), "levels": lv, "tlast": float(cv.t_last),
            "D": D, "floor": floor_share(D, cv.H),
            "overall": [round(cv.overall_removal(t), 2) if t > 0 else 0.0 for t in grid]}
    return HTML.replace("__DATA__", json.dumps(data))


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><style>
:root{--ink:#1b1a17;--ink2:#4d4a43;--rule:#d4ccbb;--focus:#0072b2}
*{box-sizing:border-box}
html,body{margin:0;background:#fff;color:var(--ink);font:16px/1.4 "Open Sans",system-ui,sans-serif}
.wrap{border:1px solid var(--rule);border-radius:8px;padding:12px 14px 10px;background:#fff}
canvas{display:block;width:100%}
.bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-top:8px}
button,select{font:inherit;min-height:44px;padding:0 14px;border:1.5px solid var(--ink);background:#fff;color:var(--ink);border-radius:6px;cursor:pointer}
button.primary{background:var(--ink);color:#fff}
input[type=range]{flex:1 1 220px;min-width:180px;height:44px;accent-color:#385F96}
label.ck{display:inline-flex;gap:6px;align-items:center;min-height:44px}
:focus-visible{outline:3px solid var(--focus);outline-offset:2px}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}
.note{font-size:14px;color:var(--ink2);margin-top:4px}
</style></head><body><div class="wrap">
<canvas id="cv" role="img" aria-label="Animated settling column. Use the controls below to play it or move through time."></canvas>
<div class="bar">
  <button id="play" class="primary" type="button" aria-pressed="false">▶ Play</button>
  <button id="restart" type="button">↺ Start over</button>
  <button id="setup" type="button">⏭ Skip the set-up</button>
  <label class="sr" for="time">Time since the test started (minutes)</label>
  <input id="time" type="range" min="0" step="0.5" value="0">
  <label class="sr" for="speed">Playback speed</label>
  <select id="speed"><option value="4">Slow</option><option value="8" selected>Normal</option><option value="16">Fast</option></select>
  <label class="ck"><input id="lv" type="checkbox" checked> Show removal levels</label>
  <label class="ck" for="fol">Follow one particle:</label>
  <select id="fol"></select>
</div>
<p class="note" id="live" aria-live="polite"></p>
</div>
<script>
/* Load the app's font from wherever the app is served (locally at /, on Streamlit Cloud at /~/+/). */
(function(){ try {
  var p = window.parent.location.pathname.replace(/accessibility\/?$/, ""); if (p.slice(-1) !== "/") p += "/";
  [["400","open-sans-latin-400-normal"],["700","open-sans-latin-700-normal"]].forEach(function(a){
    var f = new FontFace("Open Sans", "url(" + p + "app/static/fonts/" + a[1] + ".woff2)", {weight: a[0]});
    f.load().then(function(x){ document.fonts.add(x); if (window.__redraw) window.__redraw(); }).catch(function(){});
  });
} catch (e) {} })();
</script>
<script>
const D = __DATA__;
const cv = document.getElementById("cv"), g = cv.getContext("2d");
const C = {ink:"#1b1a17", ink2:"#4d4a43", blue:"#385F96", orange:"#B8481A", water:"#dbe6f3", rule:"#d4ccbb", hi:"#fff3c4", sludge:"#8c6b5a"};
const LVCOL = (()=>{ const stops=["#7a978d","#608698","#46729d","#2b5693","#143277","#010e51"]; const ks=Object.keys(D.levels);
  return Object.fromEntries(ks.map((k,i)=>[k, stops[Math.round(i*(stops.length-1)/Math.max(1,ks.length-1))]])); })();
let W = 800, H = 470, t = 0, playing = false, last = null, flash = {};
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
function rng(s){return ()=>{s|=0;s=s+0x6D2B79F5|0;let t=Math.imul(s^s>>>15,1|s);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
/* Each particle: a rank k (it rides the (k + 0.5) % isoremoval curve), a starting depth z0, and a place across the
   column. Faster particles (low rank) are drawn a little bigger, like larger flocs. */
const NR = D.D.length;
const R = rng(5), PARTS = Array.from({length: 800}, () => { const k = Math.min(NR-1, Math.floor(R()*NR));
  return {x: R(), z0: R()*D.H, k, ph: R()*6.28, s: 1.1 + 1.9*(1 - k/(NR-1))}; });
const lerp=(a,b,u)=>a+(b-a)*u, clamp=(v,a,b)=>Math.min(b,Math.max(a,v));
function portRem(tt){ const f = clamp(tt/D.step, 0, D.grid.length-1), i = Math.min(Math.floor(f), D.grid.length-2), u = f-i;
  return D.rem.map(r => lerp(r[i], r[i+1], u)); }
function levelAt(k, tt){ const a = D.levels[k], f = clamp(tt/D.step,0,a.length-1), i=Math.min(Math.floor(f),a.length-2), u=f-i;
  if (a[i]==null || a[i+1]==null) return a[Math.round(f)]; return lerp(a[i], a[i+1], u); }
function curveDepth(k, tt){ const a = D.D[k], f = clamp(tt/D.step, 0, a.length-1), i = Math.min(Math.floor(f), a.length-2), u = f-i;
  return lerp(a[i], a[i+1], u); }
function series(a, tt){ const f = clamp(tt/D.step, 0, a.length-1), i = Math.min(Math.floor(f), a.length-2), u = f-i; return lerp(a[i], a[i+1], u); }
function speed(k, tt){ const a = Math.max(0, tt-1), b = Math.min(D.tlast, tt+1); if (b <= a) return 0;
  return (curveDepth(k, b) - curveDepth(k, a)) / (b - a) * 1440; }   // m/d (or ft/d)
const folSel = document.getElementById("fol");
folSel.innerHTML = '<option value="">Off</option>' + Object.keys(D.levels).map(k => `<option value="${k}">the ${k} % curve</option>`).join("");
{ const ks = Object.keys(D.levels); folSel.value = ks.includes("50") ? "50" : (ks[Math.floor(ks.length/2)] || ""); }
function fnt(sz, w=400){ g.font = `${w} ${sz}px "Open Sans", system-ui, sans-serif`; }
function txt(s,x,y,o={}){ g.save(); fnt(o.size||14,o.w||400); g.fillStyle=o.c||C.ink; g.textAlign=o.a||"left"; g.textBaseline=o.b||"alphabetic"; g.fillText(s,x,y); g.restore(); }
/* ---- Set-up before t = 0: fill with turbid water, mix, mixer off, wait for the swirl to die down ---- */
const IN = {fill: 4.5, mix: 4.5, coast: 4.2, hold: 1.4}; IN.total = IN.fill + IN.mix + IN.coast + IN.hold;
let phase = "ready", it = 0, ilast = null;
const SW = (() => { const dt = 0.02, n = Math.ceil(IN.total/dt) + 1;
  const w = s => s < IN.fill ? 0.35 + 0.6*s/IN.fill : s < IN.fill+IN.mix ? 4.0
    : s < IN.fill+IN.mix+IN.coast ? 4.0*Math.pow(1 - (s-IN.fill-IN.mix)/IN.coast, 2) : 0;
  const th = [0]; for (let i = 1; i < n; i++) th.push(th[i-1] + w(i*dt)*dt);
  const end = th[n-1], k = Math.max(1, Math.round(end/6.2832)), sc = k*6.2832/end;   // end exactly where it began
  return {dt, th: th.map(v => v*sc), w: s => w(s)*sc}; })();
const swirl = s => { const f = clamp(s/SW.dt, 0, SW.th.length-1), i = Math.min(Math.floor(f), SW.th.length-2); return lerp(SW.th[i], SW.th[i+1], f-i); };
const smooth = u => { u = clamp(u, 0, 1); return u*u*(3 - 2*u); };
/* Each particle circulates in one of three stacked eddies and comes back exactly to its starting place. */
function introPos(p, s){ const NC = 3, hc = D.H/NC, c = Math.min(NC-1, Math.floor(p.z0/hc)), zc = (c+0.5)*hc;
  const a = (c % 2 ? -1 : 1) * swirl(s), u = (p.x-0.5)/0.5, v = (p.z0-zc)/(hc/2), cu = Math.cos(a), su = Math.sin(a);
  const om = SW.w(s), jx = 0.025*om/4*Math.sin(p.ph*7 + s*5), jz = 0.02*hc*om/4*Math.cos(p.ph*5 + s*4);
  return {x: clamp(0.5 + 0.5*(u*cu - v*su) + jx, 0, 1), z: clamp(zc + (u*su + v*cu)*hc/2 + jz, 0.004, D.H-0.004)}; }
function introText(s){ return s < IN.fill ? "Filling the column with the turbid water…"
  : s < IN.fill+IN.mix ? "Mixing, so the particles are spread evenly…"
  : s < IN.fill+IN.mix+IN.coast ? "Mixer off: waiting for the water to go still…"
  : "Still. Start the clock: t = 0 min."; }
function tint(k){ const a = [219,230,243], b = [214,203,176]; return `rgb(${a.map((x,i)=>Math.round(lerp(x, b[i], k))).join(",")})`; }
function resize(){ const w = cv.parentElement.clientWidth - 28; const dpr = window.devicePixelRatio||1;
  W = Math.max(300, w); H = W < 640 ? 390 : 470; cv.width = W*dpr; cv.height = H*dpr; cv.style.height = H+"px"; g.setTransform(dpr,0,0,dpr,0,0); draw(); }
function drawPorts(colX, colY, colW, pxm){
  D.depths.forEach((d,i) => { const y = colY + d*pxm, f = flash[i] || 0;
    g.strokeStyle = f ? C.orange : C.ink; g.lineWidth = 6; g.beginPath(); g.moveTo(colX+colW+5, y); g.lineTo(colX+colW+34, y); g.stroke();
    g.fillStyle = f ? C.orange : C.ink; g.fillRect(colX+colW+30, y-9, 14, 18);
    txt(`${d} ${D.unit}`, colX+colW+52, y+5, {size:13, c:C.ink2}); }); }
function drawIntro(colX, colY, colW, colH, pxm){
  const s = it, narrow = W < 640, fillF = smooth(s/IN.fill), topZ = D.H*(1 - fillF), wy = colY + colH*(1 - fillF);
  // water rising from the floor
  if (fillF > 0) { g.fillStyle = tint(0.8); g.fillRect(colX, wy, colW, colY + colH - wy);
    if (fillF > 0.995) { g.strokeStyle = C.blue; g.lineWidth = 2.5; g.beginPath(); g.moveTo(colX, colY); g.lineTo(colX+colW, colY); g.stroke(); } }
  // the mixer: a shaft with three paddles, one in each eddy; it spins only while switched on
  const ms = s - IN.fill, spin = 7*(clamp(ms, 0, IN.mix) + 0.25*clamp(ms - IN.mix, 0, 0.5)), cx = colX + colW/2;
  const lift = smooth((s - IN.fill - IN.mix - IN.coast)/IN.hold), ma = 1 - lift;
  if (ma > 0.01) { g.save(); g.globalAlpha = ma; g.translate(0, -lift*40);
    g.strokeStyle = "#5d5a52"; g.lineWidth = 4; g.beginPath(); g.moveTo(cx, colY - 22); g.lineTo(cx, colY + colH - 14); g.stroke();
    for (let c = 0; c < 3; c++) { const y = colY + (c + 0.5)*colH/3, hw = Math.max(3, 0.34*colW*Math.abs(Math.cos(spin + c*0.7)));
      g.fillStyle = "#5d5a52"; g.fillRect(cx - hw, y - 4, 2*hw, 8); }
    g.restore(); }
  // particles: carried round by the eddies, and only where there is water yet
  g.fillStyle = "rgba(77,74,67,0.78)";
  PARTS.forEach(p => { const q = introPos(p, s); if (q.z < topZ) return;
    g.beginPath(); g.arc(colX+5+q.x*(colW-10), colY+q.z*pxm, p.s, 0, 6.283); g.fill(); });
  // the inflow: a stream of turbid water from above while filling
  const sa = 1 - smooth((s - IN.fill + 0.4)/0.4);
  if (sa > 0.01 && fillF < 1) { g.save(); g.globalAlpha = sa; g.fillStyle = tint(1.0);
    g.fillRect(cx - 4, 0, 8, Math.max(0, wy - 0)); g.restore(); }
  txt("0", colX-14, colY+5, {a:"right", size:13, c:C.ink2}); txt(`${D.H} ${D.unit}`, colX-14, colY+colH+5, {a:"right", size:13, c:C.ink2});
  drawPorts(colX, colY, colW, pxm);
  if (s >= IN.fill + IN.mix + IN.coast) txt("t = 0 min", colX+colW/2, colY-30, {a:"center", size:16, w:700});
  const msg = introText(s);
  if (narrow) { g.save(); fnt(13, 700); g.fillStyle = C.ink; const words = msg.split(" "); let line = "", yy = H - 26;
      const lines = []; words.forEach(w => { const tt = line ? line + " " + w : w; if (g.measureText(tt).width > W - 24 && line) { lines.push(line); line = w; } else line = tt; });
      lines.push(line); yy = H - 9 - 17*(lines.length - 1); lines.forEach(l => { g.fillText(l, 12, yy); yy += 17; }); g.restore(); return; }
  txt(msg, colX - 40, H - 12, {size:13, w:700});
  const tx = colX + colW + 140, maxW = W - tx - 10;
  txt("Setting up the test", tx, 60, {size:14, w:700});
  g.save(); fnt(13); g.fillStyle = C.ink2; let line = "", yy = 86;
  ("Before any readings are taken, the column is filled with the water to be treated and mixed so the particles " +
   "are spread evenly from top to bottom. When the mixer is switched off the water keeps circling for a while, so " +
   "the test waits until it is still. Only then does the clock start at t = 0, and the particles begin to settle.")
    .split(" ").forEach(w => { const tt = line ? line + " " + w : w; if (g.measureText(tt).width > maxW && line) { g.fillText(line, tx, yy); yy += 19; line = w; } else line = tt; });
  if (line) g.fillText(line, tx, yy); g.restore();
}
function draw(){
  g.clearRect(0,0,W,H);
  const narrow = W < 640;
  const colX = narrow ? Math.max(64, W/2 - 100) : 64, colY = 46, colW = 140, colH = H - 90;
  const pxm = colH / D.H, pr = portRem(t);
  // column
  g.fillStyle = "#fbfaf7"; g.strokeStyle = C.ink; g.lineWidth = 2.5; g.beginPath(); g.roundRect(colX-5, colY-20, colW+10, colH+26, 10); g.fill(); g.stroke();
  const intro = phase !== "settle";
  const fl = intro ? 0 : series(D.floor, t), band = fl/100 * 38;
  if (intro) { drawIntro(colX, colY, colW, colH, pxm); return; }
  g.fillStyle = tint(0.8*(1 - fl/100)); g.fillRect(colX, colY, colW, colH);   // the water clears as particles leave it
  g.strokeStyle = C.blue; g.lineWidth = 2.5; g.beginPath(); g.moveTo(colX, colY); g.lineTo(colX+colW, colY); g.stroke();
  g.fillStyle = "rgba(77,74,67,0.78)";
  PARTS.forEach(p => { const z = p.z0 + curveDepth(p.k, t); if (z >= D.H) return;   // on the floor: part of the sludge
    const x = clamp(p.x + 0.006*Math.sin(p.ph + t*0.4), 0, 1);
    g.beginPath(); g.arc(colX+5+x*(colW-10), colY+Math.max(z, 0.004)*pxm, p.s, 0, 6.283); g.fill(); });
  // sludge: its thickness grows with the share of particles on the floor (drawn thicker than true scale)
  if (band > 0.3) { g.fillStyle = C.sludge; g.fillRect(colX, colY+colH-band, colW, band);
    if (band > 15) txt("sludge", colX+colW*0.27, colY+colH-band/2+4, {a:"center", size:12, c:"#fff", w:700}); }
  // depth scale
  txt("0", colX-14, colY+5, {a:"right", size:13, c:C.ink2}); txt(`${D.H} ${D.unit}`, colX-14, colY+colH+5, {a:"right", size:13, c:C.ink2});
  // removal levels
  if (document.getElementById("lv").checked) { let lastY = -99;
    Object.keys(D.levels).map(k => [k, levelAt(k, t)]).filter(([k, z]) => z != null && z > 0.005)
      .map(([k, z]) => [k, colY + z*pxm]).sort((a, b) => a[1] - b[1]).forEach(([k, y]) => {
      g.strokeStyle = LVCOL[k]; g.lineWidth = 2.5; g.setLineDash([6,4]); g.beginPath(); g.moveTo(colX-2, y); g.lineTo(colX+colW+2, y); g.stroke(); g.setLineDash([]);
      if (y - lastY >= 13) { txt(`${k} %`, colX-8, y+4, {a:"right", size:12, w:700, c:LVCOL[k]}); lastY = y; } }); }
  drawPorts(colX, colY, colW, pxm);
  // the followed particle starts at the surface and rides its curve (it sits on that curve's dashed line)
  const fk = folSel.value;
  if (fk) { const k = clamp(Math.round(+fk - 0.5), 0, NR-1), z = curveDepth(k, t), down = z >= D.H;
    const y = down ? colY+colH-Math.max(band, 4)/2 : colY+Math.max(z, 0.004)*pxm, x = colX+colW*0.5;
    g.fillStyle = C.orange; g.strokeStyle = "#fff"; g.lineWidth = 2; g.beginPath(); g.arc(x, y, 6.5, 0, 6.283); g.fill(); g.stroke();
    const msg = down ? `The followed particle (${fk} % curve) is on the floor.`
      : `Followed particle (${fk} % curve): ${z.toFixed(2)} ${D.unit} down, sinking at ${speed(k, t).toFixed(0)} ${D.unit}/d.`;
    if (narrow && msg.includes(": ")) { const [a, b] = msg.split(": "); txt(a + ":", 12, H-26, {size:13, w:700, c:C.orange}); txt(b, 12, H-9, {size:13, w:700, c:C.orange}); }
    else txt(msg, narrow ? 12 : colX-40, H-12, {size:13, w:700, c:C.orange}); }
  txt(`t = ${t.toFixed(0)} min`, colX+colW/2, colY-30, {a:"center", size:16, w:700});
  if (narrow) return;   // on a phone the readings are read out in the text below instead
  // table of readings collected so far
  const tx = narrow ? 16 : colX + colW + 140, ty = narrow ? colY + colH + 70 : 60, tw = W - tx - 8;
  const nT = D.times.length, cw = Math.min(96, (tw - 70) / nT), rh = narrow ? 44 : Math.min(54, (H - 150) / D.depths.length);
  txt(`Your readings (${D.munit}) and percent removal`, tx, ty - 26, {size:14, w:700});
  txt(`Depth`, tx, ty, {size:13, w:700, c:C.ink2});
  const cur = D.times.reduce((a,tm,j) => tm <= t + 1e-9 ? j : a, -1);
  D.times.forEach((tm,j) => { const x = tx + 66 + j*cw;
    if (j === cur) { g.fillStyle = C.hi; g.fillRect(x+2, ty-18, cw-4, rh*D.depths.length + 26); }
    txt(`${tm} min`, x + cw/2, ty, {a:"center", size:13, w:700, c:C.ink2}); });
  D.depths.forEach((d,i) => { const y = ty + 14 + i*rh;
    txt(`${d}`, tx, y + rh/2 + 2, {size:13, c:C.ink2});
    D.times.forEach((tm,j) => { if (tm > t + 1e-9) return; const x = tx + 66 + j*cw + cw/2;
      txt(String(+D.readings[i][j].toFixed(2)), x, y + rh/2 - 2, {a:"center", size: narrow ? 13 : 15});
      const r = (D.c0 - D.readings[i][j]) / D.c0 * 100; txt(`${r.toFixed(0)} %`, x, y + rh/2 + 14, {a:"center", size:11, c:C.blue}); }); });
  g.strokeStyle = C.rule; g.lineWidth = 1; g.strokeRect(tx - 6, ty - 22, 66 + nT*cw + 10, rh*D.depths.length + 32);
  const nextT = D.times.find(tm => tm > t + 1e-9);
  const msg = nextT != null ? `Next samples at ${nextT} min.` : "All samples taken. Beyond this point there are no readings.";
  if (!narrow) { g.save(); fnt(13); g.fillStyle = C.ink2;
    const ov = series(D.overall, t);
    const words = (`Each dot is a particle. Each one sinks along its own isoremoval curve and speeds up as it falls, as flocs grow; bigger dots are the faster ones. ` +
      `On the floor: ${fl.toFixed(0)} % of the particles (your overall removal from the curves: ${ov.toFixed(0)} %). ${msg}`).split(" ");
    let line = "", yy = ty + rh*D.depths.length + 34; const maxW = W - tx - 10;
    words.forEach(w => { const test = line ? line + " " + w : w; if (g.measureText(test).width > maxW && line) { g.fillText(line, tx, yy); yy += 18; line = w; } else line = test; });
    if (line) g.fillText(line, tx, yy); g.restore(); }
}
const slider = document.getElementById("time"), playBtn = document.getElementById("play"), live = document.getElementById("live");
slider.max = D.tlast;
function setT(v, fromUser){ const prev = t; t = clamp(v, 0, D.tlast); slider.value = t;
  D.times.forEach(tm => { if (prev < tm && t >= tm && !fromUser) D.depths.forEach((_,i)=> flash[i] = 1); });
  draw(); }
function announce(){ const pr = portRem(t); live.textContent = `At ${t.toFixed(0)} minutes, removal is ` +
  D.depths.map((d,i)=>`${pr[i].toFixed(0)} % at ${d} ${D.unit}`).join(", ") + `; ${series(D.floor, t).toFixed(0)} % of the particles are on the floor.`; }
function tick(ts){ if (!playing) { last = null; return; } if (last != null) { const dt = (ts - last)/1000;
    setT(t + dt * (+document.getElementById("speed").value)); Object.keys(flash).forEach(k => { flash[k] = Math.max(0, flash[k] - dt*1.5); if (!flash[k]) delete flash[k]; }); }
  last = ts; if (t >= D.tlast) { stop(); announce(); return; } requestAnimationFrame(tick); }
function start(){ if (phase !== "settle") return; if (t >= D.tlast) setT(0); playing = true; playBtn.textContent = "❚❚ Pause"; playBtn.setAttribute("aria-pressed","true"); requestAnimationFrame(tick); }
function stop(){ playing = false; playBtn.textContent = "▶ Play"; playBtn.setAttribute("aria-pressed","false"); }
const setupBtn = document.getElementById("setup"), locked = ["play","restart","time","speed","lv","fol"].map(id => document.getElementById(id));
function lock(on){ locked.forEach(el => el.disabled = on); setupBtn.textContent = on ? "⏭ Skip the set-up" : "↻ Watch the set-up again"; }
let lastMsg = "";
function introTick(ts){ if (phase !== "intro") { ilast = null; return; }
  if (ilast != null) it += (ts - ilast)/1000; ilast = ts;
  const m = introText(it); if (m !== lastMsg) { live.textContent = m; lastMsg = m; }
  if (it >= IN.total) { endIntro(true); return; }
  draw(); requestAnimationFrame(introTick); }
function startIntro(){ stop(); phase = "intro"; it = 0; ilast = null; lastMsg = ""; lock(true); requestAnimationFrame(introTick); }
function endIntro(play){ phase = "settle"; ilast = null; lock(false); setT(0, true); announce();
  live.textContent = "Set-up done: the column was filled, mixed and left until still. " + live.textContent;
  if (play && !reduce) start(); }
setupBtn.onclick = () => phase === "intro" ? endIntro(false) : startIntro();
playBtn.onclick = () => playing ? (stop(), announce()) : start();
document.getElementById("restart").onclick = () => { stop(); setT(0, true); announce(); };
slider.oninput = () => { stop(); setT(+slider.value, true); };
slider.onchange = announce;
document.getElementById("lv").onchange = draw;
folSel.onchange = draw;
window.addEventListener("resize", resize);
window.__redraw = resize;
/* hook for recording: deterministic frame at time tt */
window.__anim = { intro: s => { phase = "intro"; it = clamp(s, 0, IN.total); draw(); }, introDuration: IN.total,
  render: tt => { phase = "settle"; flash = {}; D.times.forEach(tm => { if (tt >= tm && tt - tm < 0.6*8) D.depths.forEach((_,i)=> flash[i] = 1 - (tt-tm)/(0.6*8)); }); t = clamp(tt,0,D.tlast); slider.value = t; draw(); }, duration: D.tlast };
document.fonts && document.fonts.load('400 14px "Open Sans"').finally(resize);
if (reduce) { phase = "settle"; lock(false); resize(); announce(); }
else { lock(true); resize(); live.textContent = "The set-up (filling, mixing, waiting for the water to go still) plays when this comes into view.";
  if ("IntersectionObserver" in window) { const io = new IntersectionObserver(es => { if (es[0].isIntersecting) { io.disconnect(); setTimeout(startIntro, 300); } }, {threshold: 0.5}); io.observe(cv); }
  else startIntro(); }
</script></body></html>"""
