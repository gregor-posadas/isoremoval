"""An animated, interactive replay of the student's settling-column test (HTML/canvas component).

Particles are drawn wherever the tool's own interpolated removal says they are still in suspension, so the picture
is the student's data, not a cartoon: the water clears from the top down exactly as fast as their readings say.
Sampling moments flash the ports and fill in the table; optional lines show where each removal level is.
"""
import json

import numpy as np


def build(cv, depths, times, readings, c0, unit, munit, levels=()):
    step = 0.5
    grid = np.arange(0, cv.t_last + 1e-9, step)
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
            "unit": unit, "munit": munit, "H": float(cv.H), "levels": lv, "tlast": float(cv.t_last)}
    return HTML.replace("__DATA__", json.dumps(data))


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><style>
@font-face{font-family:"Open Sans";font-weight:400;src:url("/app/static/fonts/open-sans-latin-400-normal.woff2") format("woff2")}
@font-face{font-family:"Open Sans";font-weight:700;src:url("/app/static/fonts/open-sans-latin-700-normal.woff2") format("woff2")}
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
  <label class="sr" for="time">Time since the test started (minutes)</label>
  <input id="time" type="range" min="0" step="0.5" value="0">
  <label class="sr" for="speed">Playback speed</label>
  <select id="speed"><option value="4">Slow</option><option value="8" selected>Normal</option><option value="16">Fast</option></select>
  <label class="ck"><input id="lv" type="checkbox" checked> Show removal levels</label>
</div>
<p class="note" id="live" aria-live="polite"></p>
</div>
<script>
const D = __DATA__;
const cv = document.getElementById("cv"), g = cv.getContext("2d");
const C = {ink:"#1b1a17", ink2:"#4d4a43", blue:"#385F96", orange:"#B8481A", water:"#dbe6f3", rule:"#d4ccbb", hi:"#fff3c4"};
const LVCOL = (()=>{ const stops=["#7a978d","#608698","#46729d","#2b5693","#143277","#010e51"]; const ks=Object.keys(D.levels);
  return Object.fromEntries(ks.map((k,i)=>[k, stops[Math.round(i*(stops.length-1)/Math.max(1,ks.length-1))]])); })();
let W = 800, H = 470, t = 0, playing = false, last = null, flash = {};
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
function rng(s){return ()=>{s|=0;s=s+0x6D2B79F5|0;let t=Math.imul(s^s>>>15,1|s);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
const R = rng(5), PARTS = Array.from({length: 900}, () => ({x: R(), z: R()*D.H, u: R(), ph: R()*6.28, s: 1.6 + R()*2}));
const lerp=(a,b,u)=>a+(b-a)*u, clamp=(v,a,b)=>Math.min(b,Math.max(a,v));
function portRem(tt){ const f = clamp(tt/D.step, 0, D.grid.length-1), i = Math.min(Math.floor(f), D.grid.length-2), u = f-i;
  return D.rem.map(r => lerp(r[i], r[i+1], u)); }
function remAt(z, pr){ const zs=[0,...D.depths], rs=[100,...pr]; if (z<=0) return 100;
  for (let i=1;i<zs.length;i++) if (z<=zs[i]) return lerp(rs[i-1], rs[i], (z-zs[i-1])/(zs[i]-zs[i-1])); return rs[rs.length-1]; }
function levelAt(k, tt){ const a = D.levels[k], f = clamp(tt/D.step,0,a.length-1), i=Math.min(Math.floor(f),a.length-2), u=f-i;
  if (a[i]==null || a[i+1]==null) return a[Math.round(f)]; return lerp(a[i], a[i+1], u); }
function fnt(sz, w=400){ g.font = `${w} ${sz}px "Open Sans", system-ui, sans-serif`; }
function txt(s,x,y,o={}){ g.save(); fnt(o.size||14,o.w||400); g.fillStyle=o.c||C.ink; g.textAlign=o.a||"left"; g.textBaseline=o.b||"alphabetic"; g.fillText(s,x,y); g.restore(); }
function resize(){ const w = cv.parentElement.clientWidth - 28; const dpr = window.devicePixelRatio||1;
  W = Math.max(300, w); H = 470; cv.width = W*dpr; cv.height = H*dpr; cv.style.height = H+"px"; g.setTransform(dpr,0,0,dpr,0,0); draw(); }
function draw(){
  g.clearRect(0,0,W,H);
  const narrow = W < 640;
  const colX = narrow ? Math.max(64, W/2 - 100) : 64, colY = 46, colW = 140, colH = H - 90;
  const pxm = colH / D.H, pr = portRem(t);
  // column
  g.fillStyle = "#fbfaf7"; g.strokeStyle = C.ink; g.lineWidth = 2.5; g.beginPath(); g.roundRect(colX-5, colY-20, colW+10, colH+26, 10); g.fill(); g.stroke();
  g.fillStyle = C.water; g.fillRect(colX, colY, colW, colH);
  g.strokeStyle = C.blue; g.lineWidth = 2.5; g.beginPath(); g.moveTo(colX, colY); g.lineTo(colX+colW, colY); g.stroke();
  g.fillStyle = "rgba(77,74,67,0.8)";
  PARTS.forEach(p => { const z = clamp(p.z + 0.01*Math.sin(p.ph + t*0.3), 0.005, D.H-0.005);
    if (p.u*100 < remAt(z, pr)) return; g.beginPath(); g.arc(colX+5+p.x*(colW-10), colY+z*pxm, p.s, 0, 6.283); g.fill(); });
  // depth scale
  txt("0", colX-14, colY+5, {a:"right", size:13, c:C.ink2}); txt(`${D.H} ${D.unit}`, colX-14, colY+colH+5, {a:"right", size:13, c:C.ink2});
  // removal levels
  if (document.getElementById("lv").checked) Object.keys(D.levels).forEach(k => { const z = levelAt(k, t); if (z == null || !(z > 0.005)) return;
    const y = colY + z*pxm; g.strokeStyle = LVCOL[k]; g.lineWidth = 2.5; g.setLineDash([6,4]); g.beginPath(); g.moveTo(colX-2, y); g.lineTo(colX+colW+2, y); g.stroke(); g.setLineDash([]);
    txt(`${k} %`, colX-8, y+4, {a:"right", size:12, w:700, c:LVCOL[k]}); });
  // ports
  D.depths.forEach((d,i) => { const y = colY + d*pxm, f = flash[i] || 0;
    g.strokeStyle = f ? C.orange : C.ink; g.lineWidth = 6; g.beginPath(); g.moveTo(colX+colW+5, y); g.lineTo(colX+colW+34, y); g.stroke();
    g.fillStyle = f ? C.orange : C.ink; g.fillRect(colX+colW+30, y-9, 14, 18);
    txt(`${d} ${D.unit}`, colX+colW+52, y+5, {size:13, c:C.ink2}); });
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
    const words = `Dots are particles still in suspension. The water clears from the top first, because removal is 100 % at the surface. ${msg}`.split(" ");
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
  D.depths.map((d,i)=>`${pr[i].toFixed(0)} % at ${d} ${D.unit}`).join(", ") + "."; }
function tick(ts){ if (!playing) { last = null; return; } if (last != null) { const dt = (ts - last)/1000;
    setT(t + dt * (+document.getElementById("speed").value)); Object.keys(flash).forEach(k => { flash[k] = Math.max(0, flash[k] - dt*1.5); if (!flash[k]) delete flash[k]; }); }
  last = ts; if (t >= D.tlast) { stop(); announce(); return; } requestAnimationFrame(tick); }
function start(){ if (t >= D.tlast) setT(0); playing = true; playBtn.textContent = "❚❚ Pause"; playBtn.setAttribute("aria-pressed","true"); requestAnimationFrame(tick); }
function stop(){ playing = false; playBtn.textContent = "▶ Play"; playBtn.setAttribute("aria-pressed","false"); }
playBtn.onclick = () => playing ? (stop(), announce()) : start();
document.getElementById("restart").onclick = () => { stop(); setT(0, true); announce(); };
slider.oninput = () => { stop(); setT(+slider.value, true); };
slider.onchange = announce;
document.getElementById("lv").onchange = draw;
window.addEventListener("resize", resize);
/* hook for recording: deterministic frame at time tt */
window.__anim = { render: tt => { flash = {}; D.times.forEach(tm => { if (tt >= tm && tt - tm < 0.6*8) D.depths.forEach((_,i)=> flash[i] = 1 - (tt-tm)/(0.6*8)); }); t = clamp(tt,0,D.tlast); slider.value = t; draw(); }, duration: D.tlast };
document.fonts && document.fonts.load('400 14px "Open Sans"').finally(resize);
resize(); announce();
if (!reduce && "IntersectionObserver" in window) { const io = new IntersectionObserver(es => { if (es[0].isIntersecting) { io.disconnect(); setTimeout(start, 400); } }, {threshold: 0.6}); io.observe(cv); }
</script></body></html>"""
