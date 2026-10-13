/* A small animation engine in the spirit of 3Blue1Brown's Manim, plus the eight scenes of the by-hand method.
   Everything is a function of time: every actor has keyframed tracks, so the scene can be played, paused and
   scrubbed, and reduced-motion users get the finished picture. Numbers move from where they come from to where
   they are used, and change in place, so each transformation can be followed with the eye. */
"use strict";
const COL = {bg: "#0f1115", fg: "#ececec", muted: "#9a9a9a", dim: "#5c5f66", grid: "#3a3d44", blue: "#9EB8DB",
             gold: "#E7B800", orange: "#E8834F", panel: "#171a20"};
const FONT = '"Open Sans", system-ui, sans-serif';

/* ---------- easing ---------- */
const EASE = {
  smooth: u => u*u*u*(u*(u*6 - 15) + 10),
  linear: u => u,
  out: u => 1 - Math.pow(1 - u, 3),
  tab: u => Math.sin(Math.PI*u),        // there and back (for arcs and pulses)
};
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const lerp = (a, b, u) => a + (b - a)*u;
function hex2rgb(h){ h = h.replace("#", ""); return [0, 2, 4].map(i => parseInt(h.substr(i, 2), 16)); }
function mixColor(a, b, u){ const A = hex2rgb(a), B = hex2rgb(b);
  return "#" + A.map((x, i) => Math.round(lerp(x, B[i], u)).toString(16).padStart(2, "0")).join(""); }

/* ---------- actors and scenes ---------- */
const DEF = {x: 0, y: 0, oy: 0, opacity: 1, scale: 1, color: COL.fg, size: 18, weight: 400, align: "center", text: "",
             reveal: 1, lw: 2, draw: 1, fill: null, stroke: null, w: 0, h: 0, r: 0, x1: 0, y1: 0, x2: 0, y2: 0,
             dash: null, pts: [], italic: false, glow: 0, head: 0};
class Actor {
  constructor(type, props){ this.type = type; this.p0 = Object.assign({}, DEF, props); this.tracks = {}; this.z = props.z || 0; }
  val(k, t){
    const tr = this.tracks[k]; let v = this.p0[k];
    if (!tr) return v;
    for (const s of tr){
      if (t >= s.t1) { v = s.ease === "tab" ? s.a : s.num ? s.fmt(s.b) : s.b; continue; }
      if (t > s.t0) { const u = (EASE[s.ease] || EASE.smooth)((t - s.t0)/(s.t1 - s.t0)); return blend(s, s.a, s.b, u); }
      break;
    }
    return v;
  }
}
function blend(s, a, b, u){
  if (s.num) return s.fmt(lerp(a, b, u));
  if (typeof a === "number" && typeof b === "number") return s.ease === "tab" ? lerp(a, b, u) : lerp(a, b, u);
  if (typeof a === "string" && typeof b === "string" && a[0] === "#" && b[0] === "#" && a.length === 7 && b.length === 7) return mixColor(a, b, u);
  if (Array.isArray(a) && Array.isArray(b) && a.length === b.length) return a.map((p, i) => [lerp(p[0], b[i][0], u), lerp(p[1], b[i][1], u)]);
  if (typeof a === "string" || typeof b === "string") return {from: a, to: b, u};
  return u < 0.5 ? a : b;
}
class Scene {
  constructor(ctx, mul = 1){ this.actors = []; this.cues = []; this.dur = 0; this.ctx = ctx; this.mul = mul; }
  /* text sizes are given for the wide stage; on the tall (phone) stage they are scaled up so they stay readable */
  add(type, props, raw = false){ props = Object.assign({}, props || {});
    if (!raw && props.size != null) props.size *= this.mul;
    const a = new Actor(type, props); this.actors.push(a); return a; }
  to(a, props, t0, dur = 0.8, ease = "smooth", raw = false){
    if (!raw && props.size != null) props = Object.assign({}, props, {size: props.size*this.mul});
    for (const k in props){
      const seg = {t0, t1: t0 + dur, a: a.val(k, t0), b: props[k], ease};
      (a.tracks[k] = a.tracks[k] || []).push(seg); a.tracks[k].sort((x, y) => x.t0 - y.t0);
    }
    this.dur = Math.max(this.dur, t0 + dur); return t0 + dur;
  }
  set(a, props, t){ return this.to(a, props, t, 0); }
  count(a, from, to, t0, dur, fmt, ease = "smooth"){
    const seg = {t0, t1: t0 + dur, a: from, b: to, ease, num: true, fmt};
    (a.tracks.text = a.tracks.text || []).push(seg); a.tracks.text.sort((x, y) => x.t0 - y.t0);
    // after the count the text stays at the final value
    this.dur = Math.max(this.dur, t0 + dur); return t0 + dur;
  }
  /* appear: fade in while sliding up a little (Manim's FadeIn(shift=UP)) */
  fadeIn(a, t0, dur = 0.6, dy = 10){ a.p0.opacity = 0; const y = a.val("y", t0);
    this.set(a, {y: y + dy}, t0); this.to(a, {opacity: 1, y}, t0, dur); return t0 + dur; }
  fadeOut(a, t0, dur = 0.5){ return this.to(a, {opacity: 0}, t0, dur); }
  /* write: reveal from left to right (Manim's Write) */
  write(a, t0, dur = 0.8){ a.p0.opacity = 0; this.set(a, {opacity: 1, reveal: 0}, t0); return this.to(a, {reveal: 1}, t0, dur, "linear"); }
  /* draw a line or path on (Manim's Create) */
  create(a, t0, dur = 0.8){ a.p0.draw = 0; return this.to(a, {draw: 1}, t0, dur); }
  /* move along an arc (Manim's path_arc) */
  fly(a, x, y, t0, dur = 0.9, arc = 40){ this.to(a, {x, y}, t0, dur); this.to(a, {oy: -arc}, t0, dur, "tab"); return t0 + dur; }
  /* flash a highlight around a point (Manim's Indicate/Flash) */
  pulse(a, t0, dur = 0.5, k = 1.25){ this.to(a, {scale: k}, t0, dur, "tab"); return t0 + dur; }
  copy(a, t, props){ props = Object.assign({}, props || {}); if (props.size != null) props.size *= this.mul;
    const c = this.add(a.type, Object.assign({}, a.p0, {x: a.val("x", t), y: a.val("y", t), text: textAt(a, t),
      color: a.val("color", t), size: a.val("size", t), opacity: 0, reveal: 1, z: (a.z || 0) + 5}, props), true);
    this.set(c, {opacity: 1}, t); return c; }
  cue(t, s){ this.cues.push([t, s]); }
  measure(s, size, weight = 400){ const g = this.ctx; g.save(); g.font = `${weight} ${size*this.mul}px ${FONT}`; const w = g.measureText(s).width; g.restore(); return w; }
}
function textAt(a, t){ const v = a.val("text", t); return typeof v === "object" && v ? v.to : v; }

/* ---------- drawing ---------- */
function render(g, S, t, L){
  g.save(); g.fillStyle = COL.bg; g.fillRect(0, 0, L.W, L.H);
  const list = S.actors.slice().sort((a, b) => a.z - b.z);
  for (const a of list){
    const op = a.val("opacity", t); if (op <= 0.004) continue;
    const x = a.val("x", t), y = a.val("y", t) + a.val("oy", t), sc = a.val("scale", t);
    g.save(); g.globalAlpha = clamp(op, 0, 1);
    if (a.type === "text"){
      const v = a.val("text", t), size = a.val("size", t), color = a.val("color", t), wt = a.p0.weight;
      g.translate(x, y); g.scale(sc, sc);
      g.font = `${a.p0.italic ? "italic " : ""}${wt} ${size}px ${FONT}`; g.textAlign = a.p0.align; g.textBaseline = "middle";
      const rv = a.val("reveal", t);
      const drawT = (s, alpha) => { if (!s) return; g.save(); g.globalAlpha *= alpha;
        if (rv < 1){ const w = g.measureText(s).width, x0 = a.p0.align === "center" ? -w/2 : a.p0.align === "right" ? -w : 0;
          g.beginPath(); g.rect(x0 - 2, -size, (w + 4)*rv, size*2); g.clip(); }
        const gl = a.val("glow", t); if (gl > 0){ g.shadowColor = COL.gold; g.shadowBlur = 14*gl; }
        g.fillStyle = color; g.fillText(s, 0, 0); g.restore(); };
      if (typeof v === "object" && v){ drawT(String(v.from), 1 - v.u); drawT(String(v.to), v.u); } else drawT(String(v), 1);
    } else if (a.type === "rect"){
      const w = a.val("w", t), h = a.val("h", t), r = a.p0.r, fill = a.val("fill", t), stroke = a.val("stroke", t);
      g.translate(x + w/2, y + h/2); g.scale(sc, sc); g.beginPath(); g.roundRect(-w/2, -h/2, w, h, r);
      if (fill) { g.fillStyle = fill; g.fill(); }
      if (stroke) { g.strokeStyle = stroke; g.lineWidth = a.val("lw", t); g.stroke(); }
    } else if (a.type === "circle"){
      const r = a.val("r", t)*sc, fill = a.val("fill", t), stroke = a.val("stroke", t);
      g.beginPath(); g.arc(x, y, Math.max(0, r), 0, 6.2832);
      if (fill) { g.fillStyle = fill; g.fill(); } if (stroke) { g.strokeStyle = stroke; g.lineWidth = a.val("lw", t); g.stroke(); }
    } else if (a.type === "line" || a.type === "arrow"){
      const x1 = a.val("x1", t), y1 = a.val("y1", t), x2 = a.val("x2", t), y2 = a.val("y2", t), d = a.val("draw", t);
      const ex = lerp(x1, x2, d), ey = lerp(y1, y2, d);
      g.strokeStyle = a.val("color", t); g.lineWidth = a.val("lw", t); if (a.p0.dash) g.setLineDash(a.p0.dash);
      g.beginPath(); g.moveTo(x1, y1); g.lineTo(ex, ey); g.stroke(); g.setLineDash([]);
      if (a.type === "arrow" && d > 0.05){ const ang = Math.atan2(y2 - y1, x2 - x1), hs = a.p0.head || 9; g.fillStyle = a.val("color", t);
        g.beginPath(); g.moveTo(ex, ey); g.lineTo(ex - hs*Math.cos(ang - 0.4), ey - hs*Math.sin(ang - 0.4));
        g.lineTo(ex - hs*Math.cos(ang + 0.4), ey - hs*Math.sin(ang + 0.4)); g.closePath(); g.fill(); }
    } else if (a.type === "path"){
      const pts = a.val("pts", t), d = a.val("draw", t); if (pts.length < 2) { g.restore(); continue; }
      let total = 0; const seg = []; for (let i = 1; i < pts.length; i++){ const l = Math.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1]); seg.push(l); total += l; }
      let left = total*d; g.beginPath(); g.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < pts.length && left > 0; i++){ const l = seg[i-1], u = Math.min(1, left/l);
        g.lineTo(lerp(pts[i-1][0], pts[i][0], u), lerp(pts[i-1][1], pts[i][1], u)); left -= l; }
      const fill = a.val("fill", t);
      if (fill && d >= 1){ g.closePath(); g.fillStyle = fill; g.fill(); }
      else { g.strokeStyle = a.val("color", t); g.lineWidth = a.val("lw", t); g.lineJoin = "round"; g.lineCap = "round";
        if (a.p0.dash) g.setLineDash(a.p0.dash); g.stroke(); }
    }
    g.restore();
  }
  g.restore();
}

/* ---------- helpers shared by scenes ---------- */
const f2 = (x, nd = 2) => { if (x == null || !isFinite(x)) return "–"; let s = x.toFixed(nd); if (s.includes(".")) s = s.replace(/0+$/, "").replace(/\.$/, ""); return s === "-0" ? "0" : s; };
const pct = x => f2(x, 1) + " %";

/* A table: header row + header column + cells. Returns geometry; actors are made by the caller. */
function grid(x, y, w, h, nr, nc, opts = {}){
  const hw = opts.headW || w/(nc + 1), cw = (w - hw)/nc, rh = h/(nr + 1);
  return {x, y, w, h, hw, cw, rh, nr, nc,
    cell: (i, j) => [x + hw + cw*(j + 0.5), y + rh*(i + 1.5)],
    colHead: j => [x + hw + cw*(j + 0.5), y + rh*0.5],
    rowHead: i => [x + hw*0.5, y + rh*(i + 1.5)],
    font: Math.min(opts.maxFont || 20, cw*0.32/MUL, rh*0.5/MUL)};
}
function drawGridLines(S, G, t0, dur = 1.0){
  const lines = [];
  const hl = S.add("line", {x1: G.x, y1: G.y + G.rh, x2: G.x + G.w, y2: G.y + G.rh, color: COL.grid, lw: 1.5});
  const vl = S.add("line", {x1: G.x + G.hw, y1: G.y, x2: G.x + G.hw, y2: G.y + G.h, color: COL.grid, lw: 1.5});
  S.create(hl, t0, dur); S.create(vl, t0 + 0.15, dur); lines.push(hl, vl); return lines;
}
/* Axes for time (right) and depth (down). Returns mapping functions. */
function axes(S, x, y, w, h, tmax, H, t0, opts = {}){
  const X = t => x + w*t/tmax, Y = z => y + h*z/H;
  const ax = S.add("arrow", {x1: x, y1: y, x2: x + w + 12, y2: y, color: COL.muted, lw: 1.5, head: 8});
  const ay = S.add("arrow", {x1: x, y1: y, x2: x, y2: y + h + 12, color: COL.muted, lw: 1.5, head: 8});
  S.create(ax, t0, 0.8); S.create(ay, t0, 0.8);
  const fs = opts.fs || 14, items = [ax, ay];
  const tl = S.add("text", {x: x + w, y: y - 16, text: opts.xlabel || "time (min)", size: fs, color: COL.muted, align: "right"});
  const zl = S.add("text", {x: x + 8, y: y + h + 22, text: opts.ylabel || `depth (${D.unit})`, size: fs, color: COL.muted, align: "left"});
  S.fadeIn(tl, t0 + 0.4); S.fadeIn(zl, t0 + 0.4); items.push(tl, zl);
  (opts.zticks || []).filter(z => z < H*0.97).forEach(z => { const a = S.add("text", {x: x - 8, y: Y(z), text: f2(z), size: fs - 1, color: COL.muted, align: "right"}); S.fadeIn(a, t0 + 0.5); items.push(a); });
  (opts.tticks || []).forEach(tt => { if (tt / tmax >= 0.86) return; const a = S.add("text", {x: X(tt), y: y - 16, text: f2(tt, 0), size: fs - 1, color: COL.muted}); S.fadeIn(a, t0 + 0.5); items.push(a); });
  return {X, Y, items};
}
function niceTicks(max, n = 5){ const raw = max/n, p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw/p;
  const step = (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10)*p; const out = []; for (let v = step; v < max*0.999; v += step) out.push(+v.toFixed(6)); return out; }

/* ---------- the eight scenes ---------- */
let D = null;     // data from Python
let MUL = 1;      // text scale of the current stage (bigger on the tall phone stage)
const SCENES = {};

/* 1. The raw table */
SCENES.raw = (S, L) => {
  const nr = D.depths.length, nc = D.times.length;
  const G = L.tall ? grid(30, 110, L.W - 60, Math.min(520, 70*(nr + 1)), nr, nc, {maxFont: 24})
                   : grid(120, 80, L.W - 240, Math.min(360, 58*(nr + 1)), nr, nc, {maxFont: 22});
  const fs = G.font;
  const ttl = S.add("text", {x: L.W/2, y: L.tall ? 50 : 36, text: `Your readings (${D.munit})`, size: fs + 4, weight: 700});
  S.write(ttl, 0.2, 0.9);
  drawGridLines(S, G, 0.6);
  const dh = S.add("text", {x: G.x + G.hw/2, y: G.y + G.rh/2, text: `depth`, size: fs*0.8, color: COL.muted}); S.fadeIn(dh, 1.2);
  D.times.forEach((t, j) => { const [x, y] = G.colHead(j); const a = S.add("text", {x, y, text: `${f2(t, 0)} min`, size: fs*0.85, color: COL.blue, weight: 700}); S.fadeIn(a, 1.3 + 0.08*j); });
  D.depths.forEach((z, i) => { const [x, y] = G.rowHead(i); const a = S.add("text", {x, y, text: `${f2(z)} ${D.unit}`, size: fs*0.85, color: COL.blue, weight: 700}); S.fadeIn(a, 2.1 + 0.08*i); });
  S.cue(0, "This is the table you bring back from the lab: one row for each sampling port, one column for each sampling time.");
  let t = 3.0;
  for (let j = 0; j < nc; j++) for (let i = 0; i < nr; i++){ const [x, y] = G.cell(i, j);
    const a = S.add("text", {x, y, text: f2(D.conc[i][j]), size: fs}); S.write(a, t + 0.11*(j*nr + i), 0.35); }
  t += 0.11*nr*nc + 0.4;
  S.cue(3.0, `Each cell is one sample: what was measured at that depth and time, in ${D.munit}.`);
  const c0 = S.add("text", {x: L.W/2, y: G.y + G.h + (L.tall ? 60 : 44), text: `C₀ = ${f2(D.c0)} ${D.munit}  (the well-mixed water at t = 0)`, size: fs*0.95, color: COL.gold, weight: 700});
  S.fadeIn(c0, t);
  S.cue(t, "C₀ is the reading of the well-mixed water before anything settled. Every percent removal is measured against it.");
  S.dur = t + 3;
};

/* 2. Percent removal */
SCENES.convert = (S, L) => {
  const nr = D.depths.length, nc = D.times.length;
  const G = L.tall ? grid(30, 70, L.W - 60, Math.min(400, 62*(nr + 1)), nr, nc, {maxFont: 22})
                   : grid(24, 70, 480, Math.min(340, 56*(nr + 1)), nr, nc, {maxFont: 20});
  const fs = G.font;
  drawGridLines(S, G, 0, 0.01);
  D.times.forEach((t, j) => { const [x, y] = G.colHead(j); S.add("text", {x, y, text: `${f2(t, 0)} min`, size: fs*0.85, color: COL.blue, weight: 700}); });
  D.depths.forEach((z, i) => { const [x, y] = G.rowHead(i); S.add("text", {x, y, text: `${f2(z)} ${D.unit}`, size: fs*0.85, color: COL.blue, weight: 700}); });
  const cells = D.conc.map((row, i) => row.map((c, j) => { const [x, y] = G.cell(i, j); return S.add("text", {x, y, text: f2(c), size: fs}); }));
  // the formula area
  const fx = L.tall ? L.W/2 : 520 + (L.W - 520)/2, fy = L.tall ? G.y + G.h + 70 : 150, ffs = L.tall ? 24 : 21;
  const box = S.add("rect", {x: cells[0][0].p0.x - G.cw/2 + 3, y: cells[0][0].p0.y - G.rh/2 + 3, w: G.cw - 6, h: G.rh - 6, stroke: COL.gold, lw: 2.5, r: 6, opacity: 0});
  S.to(box, {opacity: 1}, 0.6, 0.4);
  S.cue(0, `Take one reading: ${f2(D.conc[0][0])} ${D.munit} at ${f2(D.depths[0])} ${D.unit} after ${f2(D.times[0], 0)} minutes.`);
  const l1 = S.add("text", {x: fx, y: fy, text: "removal = (C₀ − C) ÷ C₀ × 100", size: ffs});
  S.write(l1, 1.2, 1.2);
  S.cue(1.2, "Percent removal compares it with the starting concentration: (C₀ − C) ÷ C₀ × 100.");
  // line 2: = (20 − 14) ÷ 20 × 100, with 14 flying in from the table and 20 from C₀
  const parts = ["= (", f2(D.c0), " − ", f2(D.conc[0][0]), ") ÷ ", f2(D.c0), " × 100"];
  const widths = parts.map(p => S.measure(p, ffs)), tot = widths.reduce((a, b) => a + b, 0);
  let xx = fx - tot/2; const y2 = fy + ffs*1.9, slots = [];
  parts.forEach((p, k) => { slots.push(xx + widths[k]/2); xx += widths[k]; });
  const p2 = parts.map((p, k) => { if (k === 3) return null; const a = S.add("text", {x: slots[k], y: y2, text: p, size: ffs, color: k === 1 || k === 5 ? COL.gold : COL.fg}); S.fadeIn(a, 2.8 + 0.1*k, 0.5, 6); return a; });
  const fly = S.copy(cells[0][0], 2.6); S.to(fly, {color: COL.gold, size: ffs}, 2.6, 0.9); S.fly(fly, slots[3], y2, 2.6, 1.0, 60);
  const res = 100*(D.c0 - D.conc[0][0])/D.c0;
  const l3 = S.add("text", {x: fx, y: fy + ffs*3.8, text: `= ${pct(res)}`, size: ffs + 6, weight: 700, color: COL.blue});
  S.fadeIn(l3, 4.4);
  S.cue(2.6, `Put the numbers in: (${f2(D.c0)} − ${f2(D.conc[0][0])}) ÷ ${f2(D.c0)} × 100 = ${pct(res)}.`);
  // the answer flies back into the table, and the cell changes
  const back = S.copy(l3, 5.6, {text: pct(res)}); S.fly(back, cells[0][0].p0.x, cells[0][0].p0.y, 5.6, 1.0, -50); S.to(back, {size: fs, color: COL.blue}, 5.6, 1.0);
  S.set(back, {opacity: 0}, 6.6); S.set(cells[0][0], {text: pct(res), color: COL.blue}, 6.6);
  [l1, l3, fly, ...p2.filter(Boolean)].forEach(a => S.fadeOut(a, 7.0));
  S.to(box, {opacity: 0}, 7.0, 0.4);
  // every other cell, in a diagonal wave
  const counter = S.add("text", {x: fx, y: L.tall ? fy + 10 : fy + 40, text: "1", size: 54, weight: 700, color: COL.gold, opacity: 0});
  const clab = S.add("text", {x: fx, y: (L.tall ? fy + 10 : fy + 40) + 44, text: "calculations like that", size: ffs*0.85, color: COL.muted});
  S.fadeIn(counter, 7.4); S.fadeIn(clab, 7.4);
  let n = 1, t = 7.8; const order = [];
  for (let s = 0; s < nr + nc; s++) for (let i = 0; i < nr; i++){ const j = s - i; if (j >= 0 && j < nc && !(i === 0 && j === 0)) order.push([i, j]); }
  order.forEach(([i, j], k) => { const tt = t + 0.12*k, r = D.R[i][j];
    S.to(cells[i][j], {text: pct(r), color: COL.blue}, tt, 0.4); S.pulse(cells[i][j], tt, 0.4, 1.18); });
  S.count(counter, 1, nr*nc, t, 0.12*order.length + 0.3, v => String(Math.round(v)), "linear");
  S.cue(7.4, `Now the same calculation for every cell: ${nr*nc} of them, each one by hand.`);
  t += 0.12*order.length + 0.8;
  S.cue(t, "The table now says what share of the starting concentration is gone at each port and time.");
  S.dur = t + 3;
};

/* 3. Flip the question: time down the side, removal levels across the top */
SCENES.flip = (S, L) => {
  const nr = D.depths.length, nc = D.times.length, nl = D.levels.length, rows = D.rows;
  const GA = L.tall ? grid(20, 40, L.W - 40, 300, nr, nc, {maxFont: 18}) : grid(16, 70, 430, Math.min(330, 52*(nr + 1)), nr, nc, {maxFont: 17});
  const GB = L.tall ? grid(20, 400, L.W - 40, Math.min(400, 50*(rows.length + 1)), rows.length, nl, {maxFont: 19})
                    : grid(470, 70, L.W - 486, Math.min(440, 52*(rows.length + 1)), rows.length, nl, {maxFont: 18});
  const fa = GA.font, fb = GB.font;
  drawGridLines(S, GA, 0, 0.01);
  const aHeads = D.times.map((t, j) => { const [x, y] = GA.colHead(j); return S.add("text", {x, y, text: `${f2(t, 0)} min`, size: fa*0.85, color: COL.blue, weight: 700}); });
  const aRows = D.depths.map((z, i) => { const [x, y] = GA.rowHead(i); return S.add("text", {x, y, text: `${f2(z)} ${D.unit}`, size: fa*0.85, color: COL.blue, weight: 700}); });
  const aCells = D.R.map((row, i) => row.map((r, j) => { const [x, y] = GA.cell(i, j); return S.add("text", {x, y, text: pct(r), size: fa, color: COL.blue}); }));
  S.cue(0, "Here's the turn that makes it confusing. The curves need the depth of a removal level at each time, but the table gives removal at each depth.");
  // the new table
  drawGridLines(S, GB, 1.6);
  const bt = S.add("text", {x: GB.x + GB.w/2, y: GB.y - 18, text: `depth (${D.unit}) where removal reaches…`, size: fb*0.85, color: COL.muted}); S.fadeIn(bt, 1.8);
  const bLev = D.levels.map((p, k) => { const [x, y] = GB.colHead(k); const a = S.add("text", {x, y, text: `${f2(p, 0)} %`, size: fb*0.9, color: COL.gold, weight: 700}); S.fadeIn(a, 2.0 + 0.08*k); return a; });
  S.cue(1.6, "So start a new table: the removal levels you want to plot go across the top…");
  // the sampling times move from across the top of the old table to down the side of the new one
  const t0row = S.add("text", {x: GB.rowHead(0)[0], y: GB.rowHead(0)[1], text: "0 min", size: fb*0.85, color: COL.blue, weight: 700}); S.fadeIn(t0row, 3.4);
  D.times.forEach((t, j) => { const c = S.copy(aHeads[j], 3.6 + 0.18*j); const [x, y] = GB.rowHead(j + 1); S.fly(c, x, y, 3.6 + 0.18*j, 1.1, 40); S.to(c, {size: fb*0.85}, 3.6 + 0.18*j, 1.1); });
  S.cue(3.4, "…and the sampling times move from across the top to down the side. This is the switch of rows and columns.");
  let t = 3.6 + 0.18*nc + 1.4;
  const bCells = rows.map((_, i) => D.levels.map((_, k) => { const [x, y] = GB.cell(i, k); return S.add("text", {x, y, text: "", size: fb, opacity: 0}); }));
  D.levels.forEach((_, k) => { S.set(bCells[0][k], {text: "0", color: COL.fg}, t); S.fadeIn(bCells[0][k], t + 0.06*k, 0.4, 6); });
  S.cue(t, "At time zero nothing has settled, so every level is at the surface: depth 0.");
  t += 1.6;
  // easy cells: the percent table has the level exactly, so copy that port's depth
  S.cue(t, "Easy cells: where the percent table shows a level exactly, copy that port's depth straight across.");
  let nEasy = 0;
  D.table.forEach((row, i) => row.forEach((c, k) => { if (c.k !== "easy") return;
    const j = i - 1, ii = D.depths.findIndex(z => Math.abs(z - c.d) < 1e-9);
    if (j < 0 || ii < 0) return;
    const hit = aCells[ii][j];
    S.to(hit, {color: COL.gold}, t, 0.3); S.pulse(hit, t, 0.5, 1.3); S.to(aRows[ii], {color: COL.gold}, t, 0.3);
    const cp = S.copy(aRows[ii], t + 0.4, {text: f2(c.d), color: COL.gold}); const [x, y] = GB.cell(i, k);
    S.fly(cp, x, y, t + 0.4, 0.9, 50); S.to(cp, {size: fb}, t + 0.4, 0.9);
    S.to(hit, {color: COL.blue}, t + 1.2, 0.4); S.to(aRows[ii], {color: COL.blue}, t + 1.2, 0.4);
    t += 0.95; nEasy++; }));
  t += 0.6;
  // what's left: hard cells (?) and levels already past the bottom (–)
  D.table.forEach((row, i) => row.forEach((c, k) => { if (i === 0 || c.k === "easy") return;
    S.set(bCells[i][k], {text: c.k === "below" ? "–" : "?", color: c.k === "below" ? COL.dim : COL.gold}, t);
    S.fadeIn(bCells[i][k], t + 0.03*(i*nl + k), 0.4, 6); }));
  const nHard = D.tally.hard;
  S.cue(t, `${nEasy === 0 ? "None of the cells were" : nEasy === 1 ? "Only 1 cell was" : `Only ${nEasy} cells were`} that easy. Each "?" (${nHard} of them) needs an interpolation; a dash means the level has already passed the bottom port.`);
  S.dur = t + 4;
};

/* 4. Interpolate one hard cell, then all the rest */
SCENES.interp = (S, L) => {
  const rows = D.rows, nl = D.levels.length, ex = D.example.hard, [ei, ek] = ex, cell = D.table[ei][ek];
  const GB = L.tall ? grid(20, 500, L.W - 40, Math.min(310, 44*(rows.length + 1)), rows.length, nl, {maxFont: 18})
                    : grid(500, 80, L.W - 516, Math.min(420, 50*(rows.length + 1)), rows.length, nl, {maxFont: 17});
  const fb = GB.font;
  drawGridLines(S, GB, 0, 0.01);
  D.levels.forEach((p, k) => { const [x, y] = GB.colHead(k); S.add("text", {x, y, text: `${f2(p, 0)} %`, size: fb*0.9, color: COL.gold, weight: 700}); });
  rows.forEach((t, i) => { const [x, y] = GB.rowHead(i); S.add("text", {x, y, text: `${f2(t, 0)} min`, size: fb*0.85, color: COL.blue, weight: 700}); });
  const bCells = D.table.map((row, i) => row.map((c, k) => { const [x, y] = GB.cell(i, k);
    const txt = c.k === "start" || c.k === "easy" ? f2(c.d) : c.k === "below" ? "–" : "?";
    return S.add("text", {x, y, text: txt, size: fb, color: c.k === "below" ? COL.dim : txt === "?" ? COL.gold : COL.fg}); }));
  const target = bCells[ei][ek];
  const hb = S.add("rect", {x: GB.cell(ei, ek)[0] - GB.cw/2 + 3, y: GB.cell(ei, ek)[1] - GB.rh/2 + 3, w: GB.cw - 6, h: GB.rh - 6, stroke: COL.gold, lw: 2.5, r: 6, opacity: 0});
  S.to(hb, {opacity: 1}, 0.4, 0.4); S.pulse(target, 0.4, 0.6, 1.4);
  const p = D.levels[ek], t = rows[ei];
  S.cue(0, `Take one "?": how deep is the ${f2(p, 0)} % level at ${f2(t, 0)} minutes?`);
  // the column of removals at that time, laid out along a depth line
  const ax0 = L.tall ? 70 : 90, ay0 = L.tall ? 70 : 90, ah = L.tall ? 300 : 300;
  const Hm = D.H, Y = z => ay0 + ah*z/Hm;
  const axis = S.add("arrow", {x1: ax0, y1: ay0, x2: ax0, y2: ay0 + ah + 14, color: COL.muted, lw: 1.5}); S.create(axis, 1.2, 0.8);
  const axl = S.add("text", {x: ax0, y: ay0 - 22, text: `removal at ${f2(t, 0)} min`, size: 15, color: COL.muted, align: "left"}); S.fadeIn(axl, 1.3);
  const zs = [0, ...D.depths], rs = [100, ...D.depths.map((_, i) => D.R[i][ei - 1])];
  const dots = [], rl = [], zl = [];
  zs.forEach((z, i) => { const y = Y(z), tt = 1.6 + 0.25*i;
    dots.push(S.add("circle", {x: ax0, y, r: 5, fill: COL.blue, opacity: 0})); S.to(dots[i], {opacity: 1}, tt, 0.3);
    zl.push(S.add("text", {x: ax0 - 12, y, text: `${f2(z)} ${D.unit}`, size: 15, color: COL.muted, align: "right"})); S.fadeIn(zl[i], tt);
    rl.push(S.add("text", {x: ax0 + 14, y, text: i === 0 ? "100 % (surface)" : pct(rs[i]), size: 16, color: COL.blue, align: "left"})); S.fadeIn(rl[i], tt); });
  S.cue(1.4, "Lay that time's removals along the depth. The surface counts as 100 %: everything has left it.");
  // the level slides down until it sits between two ports
  const ia = zs.findIndex((z, i) => i > 0 && rs[i - 1] > p && p > rs[i]) - 1;
  const za = zs[ia], zb = zs[ia + 1], ra = rs[ia], rb = rs[ia + 1], d = cell.d;
  const lev = S.add("text", {x: ax0 + 150, y: Y(0), text: `${f2(p, 0)} %`, size: 18, weight: 700, color: COL.gold, opacity: 0, align: "left"});
  let tt = 1.6 + 0.25*zs.length + 0.5;
  S.fadeIn(lev, tt); S.to(lev, {y: Y((za + zb)/2)}, tt + 0.5, 1.6);
  S.cue(tt, `${f2(p, 0)} % falls between ${pct(ra)} at ${f2(za)} ${D.unit} and ${pct(rb)} at ${f2(zb)} ${D.unit}.`);
  tt += 2.3;
  zs.forEach((_, i) => { if (i !== ia && i !== ia + 1){ S.to(rl[i], {opacity: 0.25}, tt, 0.4); S.to(zl[i], {opacity: 0.25}, tt, 0.4); S.to(dots[i], {opacity: 0.25}, tt, 0.4); } });
  [ia, ia + 1].forEach(i => { S.to(rl[i], {color: COL.gold}, tt, 0.4); S.pulse(dots[i], tt, 0.5, 1.6); });
  // the interpolation itself, built from the numbers in the picture
  const fx = L.tall ? L.W/2 : ax0 + 210, fy = L.tall ? ay0 + ah + 38 : ay0 + 120, ffs = 19;
  const parts = ["depth = ", f2(za), " + (", f2(ra, 1), " − ", f2(p, 0), ") ÷ (", f2(ra, 1), " − ", f2(rb, 1), ") × (", f2(zb), " − ", f2(za), ")"];
  const src = {1: zl[ia], 3: rl[ia], 5: lev, 7: rl[ia], 9: rl[ia + 1], 11: zl[ia + 1], 13: zl[ia]};
  const fsz = Math.min(ffs, ffs*(L.tall ? (L.W - 30) : (L.W - fx - 20 + 200))/Math.max(1, parts.reduce((a, s) => a + S.measure(s, ffs), 0)));
  const widths = parts.map(s => S.measure(s, fsz)), tot = widths.reduce((a, b) => a + b, 0);
  let xx = (L.tall ? L.W/2 : Math.min(fx + tot/2, L.W - 470) ) - tot/2; xx = Math.max(12, xx);
  const fyy = L.tall ? fy : ay0 + ah + 40;
  const slots = parts.map((s, k) => { const c = xx + widths[k]/2; xx += widths[k]; return c; });
  tt += 0.8;
  parts.forEach((s, k) => { if (src[k]){ const c = S.copy(src[k], tt + 0.15*k, {text: s, align: "center"}); S.fly(c, slots[k], fyy, tt + 0.15*k, 0.9, 40); S.to(c, {size: fsz, color: COL.gold}, tt + 0.15*k, 0.9); }
    else { const a = S.add("text", {x: slots[k], y: fyy, text: s, size: fsz}); S.fadeIn(a, tt + 0.15*k, 0.4, 6); } });
  S.cue(tt, "Linear interpolation: start at the upper depth, and go down the same fraction of the gap that the level is between the two removals.");
  tt += 0.15*parts.length + 1.2;
  const res = S.add("text", {x: L.tall ? L.W/2 : slots[0] + tot/2 - widths[0]/2, y: fyy + fsz*1.9, text: `= ${f2(za)} + ${f2(cell.frac, 3)} × ${f2(zb - za)} = ${f2(d)} ${D.unit}`, size: fsz + 2, weight: 700, color: COL.blue});
  S.fadeIn(res, tt);
  S.cue(tt, `That gives ${f2(d)} ${D.unit}. Three subtractions, a division, a multiplication and an addition, for one cell.`);
  tt += 1.6;
  const ans = S.copy(res, tt, {text: f2(d)}); S.fly(ans, GB.cell(ei, ek)[0], GB.cell(ei, ek)[1], tt, 1.1, 60); S.to(ans, {size: fb, color: COL.fg}, tt, 1.1);
  S.set(ans, {opacity: 0}, tt + 1.1); S.set(target, {text: f2(d), color: COL.fg}, tt + 1.1); S.to(hb, {opacity: 0}, tt + 1.2, 0.4);
  tt += 1.8;
  // all the others
  const counter = S.add("text", {x: L.tall ? L.W - 70 : 380, y: L.tall ? 40 : 200, text: "1", size: 40, weight: 700, color: COL.gold, opacity: 0});
  const cl = S.add("text", {x: L.tall ? L.W - 70 : 380, y: L.tall ? 66 : 232, text: "interpolations", size: 14, color: COL.muted, opacity: 0});
  S.fadeIn(counter, tt); S.fadeIn(cl, tt);
  let n = 1; const todo = [];
  D.table.forEach((row, i) => row.forEach((c, k) => { if ((c.k === "hard" || c.k === "surface") && !(i === ei && k === ek)) todo.push([i, k, c]); }));
  todo.forEach(([i, k, c], m) => { const t1 = tt + 0.32*m; S.to(bCells[i][k], {text: f2(c.d), color: COL.fg}, t1, 0.35); S.pulse(bCells[i][k], t1, 0.35, 1.35); });
  S.count(counter, 1, todo.length + 1, tt, 0.32*todo.length + 0.3, v => String(Math.round(v)), "linear");
  S.cue(tt, `Now repeat that for every other "?" — ${todo.length} more interpolations for this table.`);
  tt += 0.32*todo.length + 1;
  S.cue(tt, "The flipped table is finally full. Each column is one isoremoval curve.");
  S.dur = tt + 3;
};

/* 5. Plot the curves */
SCENES.plot = (S, L) => {
  const rows = D.rows, nl = D.levels.length;
  const GB = L.tall ? grid(20, 30, L.W - 40, Math.min(300, 40*(rows.length + 1)), rows.length, nl, {maxFont: 16})
                    : grid(16, 70, 400, Math.min(400, 48*(rows.length + 1)), rows.length, nl, {maxFont: 16});
  const fb = GB.font;
  drawGridLines(S, GB, 0, 0.01);
  const heads = D.levels.map((p, k) => { const [x, y] = GB.colHead(k); return S.add("text", {x, y, text: `${f2(p, 0)} %`, size: fb*0.9, color: D.colors[k], weight: 700}); });
  rows.forEach((t, i) => { const [x, y] = GB.rowHead(i); S.add("text", {x, y, text: `${f2(t, 0)}`, size: fb*0.85, color: COL.blue, weight: 700}); });
  const cells = D.table.map((row, i) => row.map((c, k) => { const [x, y] = GB.cell(i, k); return S.add("text", {x, y, text: c.d == null ? "–" : f2(c.d), size: fb, color: c.d == null ? COL.dim : COL.fg}); }));
  const tmax = D.rows[D.rows.length - 1]*1.08;
  const ax = L.tall ? {x: 70, y: 380, w: L.W - 130, h: 370} : {x: 490, y: 90, w: L.W - 540, h: 370};
  const A = axes(S, ax.x, ax.y, ax.w, ax.h, tmax, D.H, 0.3, {zticks: [D.H/2, D.H], tticks: niceTicks(tmax)});
  S.cue(0, "Plot each column: time across, depth down (the surface is at the top, like the column).");
  let t = 1.4;
  D.levels.forEach((p, k) => {
    const pts = [];
    D.table.forEach((row, i) => { const c = row[k]; if (c.d == null) return; const x = A.X(rows[i]), y = A.Y(c.d);
      const cp = S.copy(cells[i][k], t + 0.06*i, {color: D.colors[k]}); S.fly(cp, x, y, t + 0.06*i, 0.7, 25); S.to(cp, {opacity: 0}, t + 0.06*i + 0.7, 0.2);
      const dot = S.add("circle", {x, y, r: 4, fill: D.colors[k], opacity: 0}); S.set(dot, {opacity: 1}, t + 0.06*i + 0.7); pts.push([x, y]); });
    S.to(heads[k], {scale: 1.3}, t, 0.5, "tab");
    const path = S.add("path", {pts, color: D.colors[k], lw: 3, draw: 0}); S.to(path, {draw: 1}, t + 0.9, 0.7);
    const last = pts[pts.length - 1]; const lab = S.add("text", {x: last[0] + 8, y: last[1], text: `${f2(p, 0)} %`, size: 14, color: D.colors[k], align: "left", weight: 700});
    S.fadeIn(lab, t + 1.5, 0.4, 0);
    t += L.reduced ? 0.6 : 1.5;
  });
  S.cue(1.4, "Each column of numbers becomes one line: the depth of that removal level as time goes on.");
  S.cue(t, "These are the isoremoval curves. By hand, each one is plotted point by point and joined with a ruler.");
  S.dur = t + 3;
};

/* 6. When does each curve reach the bottom? */
SCENES.bottom = (S, L) => {
  const tbs = D.levels.map(p => D.bottom[p] && D.bottom[p].t).filter(v => v != null);
  const tmax = Math.min(Math.max(...tbs, D.rows[D.rows.length - 1])*1.05, D.rows[D.rows.length - 1]*2.6);
  const ax = L.tall ? {x: 70, y: 70, w: L.W - 130, h: 360} : {x: 80, y: 80, w: 520, h: 380};
  const A = axes(S, ax.x, ax.y, ax.w, ax.h, tmax, D.H, 0, {zticks: [D.H/2, D.H], tticks: niceTicks(tmax)});
  const bl = S.add("line", {x1: ax.x, y1: A.Y(D.H), x2: ax.x + ax.w, y2: A.Y(D.H), color: COL.orange, lw: 2, dash: [6, 5]}); S.create(bl, 0.6, 0.8);
  const blab = S.add("text", {x: ax.x + ax.w, y: A.Y(D.H) + 20, text: `bottom (${f2(D.H)} ${D.unit})`, size: 14, color: COL.orange, align: "right"}); S.fadeIn(blab, 0.8);
  const curves = D.levels.map((p, k) => { const pts = [];
    D.table.forEach((row, i) => { const c = row[k]; if (c.d != null) pts.push([A.X(D.rows[i]), A.Y(c.d)]); });
    const pa = S.add("path", {pts, color: D.colors[k], lw: 2.5}); S.fadeIn(pa, 0.2 + 0.05*k, 0.5, 0); return pa; });
  S.cue(0, "Detention time for each curve is when it reaches the bottom of the column.");
  const px = L.tall ? L.W/2 : ax.x + ax.w + 30 + (L.W - ax.x - ax.w - 40)/2, py = L.tall ? ax.y + ax.h + 60 : 120;
  let t = 1.8, shownI = false, shownE = false;
  const list = [];
  D.levels.forEach((p, k) => { const b = D.bottom[p]; if (!b || b.t == null) return;
    const x = A.X(Math.min(b.t, tmax)), y = A.Y(D.H);
    if (b.kind === "extrap"){ const sx = A.X(b.tb), sy = A.Y(b.zb);
      const ext = S.add("line", {x1: sx, y1: sy, x2: x, y2: b.t <= tmax ? y : A.Y(b.zb + (D.H - b.zb)*(tmax - b.tb)/(b.t - b.tb)), color: D.colors[k], lw: 2.5, dash: [3, 5]});
      S.create(ext, t, 0.9); }
    if (b.kind !== "extrap"){ const col = D.table.map((row, i) => [D.rows[i], row[k].d]).filter(c => c[1] != null && c[0] < b.t - 1e-9);
      const lp = col[col.length - 1]; if (lp){ const jn = S.add("line", {x1: A.X(lp[0]), y1: A.Y(lp[1]), x2: x, y2: y, color: D.colors[k], lw: 2.5}); S.create(jn, t, 0.6); } }
    const dot = S.add("circle", {x, y, r: 6, fill: D.colors[k], stroke: COL.bg, lw: 2, opacity: 0}); S.to(dot, {opacity: 1}, t + 0.6, 0.3); S.pulse(dot, t + 0.6, 0.5, 1.6);
    if ((b.kind === "interp" && !shownI) || (b.kind === "extrap" && !shownE)){
      const lines = b.kind === "interp"
        ? [`${f2(p, 0)} % curve: the bottom port reads`, `${pct(b.r0)} at ${f2(b.t0, 0)} min and ${pct(b.r1)} at ${f2(b.t1, 0)} min`,
           `t = ${f2(b.t0, 0)} + (${f2(p, 0)} − ${f2(b.r0, 1)}) ÷ (${f2(b.r1, 1)} − ${f2(b.r0, 1)}) × (${f2(b.t1, 0)} − ${f2(b.t0, 0)})`, `= ${f2(b.t, 1)} min`]
        : [`${f2(p, 0)} % curve: not at the bottom by ${f2(b.tb, 0)} min,`, `so extend its last straight piece:`,
           `t = ${f2(b.tb, 0)} + (${f2(D.H)} − ${f2(b.zb)}) ÷ [(${f2(b.zb)} − ${f2(b.za)}) ÷ (${f2(b.tb, 0)} − ${f2(b.ta, 0)})]`, `= ${f2(b.t, 1)} min`];
      const fs = L.tall ? 16 : 15, mw = Math.max(...lines.map(s => S.measure(s, fs))), maxw = L.tall ? L.W - 30 : L.W - ax.x - ax.w - 40;
      const fz = Math.min(fs, fs*maxw/mw);
      const items = lines.map((s, i) => { const a = S.add("text", {x: px, y: py + (b.kind === "extrap" ? 150 : 0) + i*fz*1.6, text: s, size: fz, color: i === 3 ? COL.gold : i === 2 ? COL.fg : COL.muted, weight: i === 3 ? 700 : 400}); S.fadeIn(a, t + 0.3 + 0.35*i); return a; });
      S.cue(t, b.kind === "interp" ? `The ${f2(p, 0)} % curve reaches the bottom between two sampling times, so interpolate in time along the bottom row.`
                                   : `The ${f2(p, 0)} % curve hasn't reached the bottom by the last reading, so extend it in a straight line. This is an extrapolation, so it's less certain.`);
      t += 3.8; if (b.kind === "interp") shownI = true; else shownE = true;
    } else t += 0.9;
    const lab = S.add("text", {x, y: y - 16, text: b.t > tmax ? `${f2(b.t, 0)} →` : f2(b.t, 0), size: 13, color: D.colors[k], weight: 700}); S.fadeIn(lab, t - 0.3, 0.3, 0);
    list.push([p, b]);
  });
  S.cue(t, `Each curve now has a detention time. ${D.tally.bottom_interp} needed an interpolation and ${D.tally.bottom_extrap} an extrapolation.`);
  S.dur = t + 3;
};

/* 7. Overall removal for one curve, by trapezoids */
SCENES.overall = (S, L) => {
  const p = D.example.curve, o = D.overall[p], k0 = D.levels.indexOf(p), tb = o.t;
  const tmax = Math.max(tb, D.rows[D.rows.length - 1])*1.08;
  const ax = L.tall ? {x: 64, y: 60, w: L.W - 120, h: 300} : {x: 56, y: 80, w: 320, h: 360};
  const A = axes(S, ax.x, ax.y, ax.w, ax.h, tmax, D.H, 0, {zticks: [D.H/2, D.H], tticks: niceTicks(tmax, 4)});
  D.levels.forEach((q, k) => { const pts = []; D.table.forEach((row, i) => { const c = row[k]; if (c.d != null) pts.push([A.X(D.rows[i]), A.Y(c.d)]); });
    if (tb > D.rows[D.rows.length - 1] && q >= p && pts.length >= 2){ const a = pts[pts.length - 2], b = pts[pts.length - 1]; const sl = (b[1] - a[1])/(b[0] - a[0]); pts.push([A.X(tmax), b[1] + sl*(A.X(tmax) - b[0])]); }
    const pa = S.add("path", {pts: pts.map(([x, y]) => [x, Math.min(y, A.Y(D.H))]), color: D.colors[k], lw: q === p ? 3.5 : 2, opacity: 0}); S.to(pa, {opacity: q < p ? 0.3 : 1}, 0.1, 0.5); });
  const vl = S.add("line", {x1: A.X(tb), y1: ax.y, x2: A.X(tb), y2: A.Y(D.H), color: COL.gold, lw: 2, dash: [5, 5]}); S.create(vl, 0.8, 0.8);
  const vlab = S.add("text", {x: A.X(tb), y: A.Y(D.H) + 20, text: `t = ${f2(tb, 1)} min`, size: 15, color: COL.gold, weight: 700}); S.fadeIn(vlab, 1.0);
  S.cue(0, `Take the ${f2(p, 0)} % curve. It reaches the bottom at ${f2(tb, 1)} min: that's its detention time.`);
  // depths of the higher curves at that time
  const pts = o.points;      // [z, removal, how]
  const dots = pts.map(([z, r], i) => { const a = S.add("circle", {x: A.X(tb), y: A.Y(z), r: 5, fill: COL.gold, opacity: 0}); S.to(a, {opacity: 1}, 2.0 + 0.3*i, 0.3); S.pulse(a, 2.0 + 0.3*i, 0.4, 1.6); return a; });
  S.cue(2.0, "At that moment, read how deep each higher curve is. Together with 100 % at the surface, that's the removal at every depth.");
  // a profile: removal (across) against depth (down)
  const pr = L.tall ? {x: 60, y: 430, w: 170, h: 300} : {x: 440, y: 90, w: 170, h: 330};
  const PX = r => pr.x + pr.w*r/100, PY = z => pr.y + pr.h*z/D.H;
  const pax = S.add("arrow", {x1: pr.x, y1: pr.y, x2: pr.x + pr.w + 10, y2: pr.y, color: COL.muted, lw: 1.5}); S.create(pax, 3.4, 0.6);
  const pay = S.add("arrow", {x1: pr.x, y1: pr.y, x2: pr.x, y2: pr.y + pr.h + 10, color: COL.muted, lw: 1.5}); S.create(pay, 3.4, 0.6);
  const prl = S.add("text", {x: pr.x + pr.w, y: pr.y - 16, text: "removal (%)", size: 13, color: COL.muted, align: "right"}); S.fadeIn(prl, 3.5);
  let t = 4.2;
  const pp = pts.map(([z, r]) => [PX(r), PY(z)]);
  pts.forEach(([z, r], i) => { const c = S.add("circle", {x: A.X(tb), y: A.Y(z), r: 5, fill: COL.gold, opacity: 0}); S.set(c, {opacity: 1}, t + 0.2*i);
    S.fly(c, pp[i][0], pp[i][1], t + 0.2*i, 0.9, 30);
    const lab = S.add("text", {x: pp[i][0] + 10, y: pp[i][1], text: L.tall ? `${f2(r, 0)} %` : `${f2(r, 0)} % at ${f2(z)}`, size: 13, color: COL.fg, align: "left"}); S.fadeIn(lab, t + 0.2*i + 0.8, 0.3, 0); });
  t += 0.2*pts.length + 1.1;
  const prof = S.add("path", {pts: pp, color: COL.blue, lw: 2.5}); S.create(prof, t, 0.8);
  S.cue(4.2, "Move those readings onto a removal-against-depth chart and join them.");
  t += 1.0;
  // strips and their areas
  const sx = L.tall ? 305 : 720, sy0 = L.tall ? 450 : 100, fsz = 14;
  const sums = [];
  o.strips.forEach((s, i) => { const poly = [[pr.x, PY(s.z0)], [PX(s.r0), PY(s.z0)], [PX(s.r1), PY(s.z1)], [pr.x, PY(s.z1)]];
    const sh = S.add("path", {pts: poly, fill: i % 2 ? "#3c5a80" : "#4f6f99", opacity: 0, z: -1}); S.to(sh, {opacity: 0.85}, t, 0.5);
    const ar = S.add("text", {x: (pr.x + Math.min(PX(s.r0), PX(s.r1)))/2, y: PY((s.z0 + s.z1)/2), text: f2(s.area, 1), size: 13, color: COL.fg, weight: 700}); S.fadeIn(ar, t + 0.3, 0.3, 0);
    const line = S.add("text", {x: sx, y: sy0 + i*fsz*1.7, text: `(${f2(s.r0, 0)} + ${f2(s.r1, 0)}) ÷ 2 × ${f2(s.z1 - s.z0)} = ${f2(s.area, 1)}`, size: fsz, align: "left", color: COL.fg});
    S.fadeIn(line, t + 0.4, 0.4, 6); sums.push(line); t += 1.1; });
  S.cue(t - 1.1*o.strips.length, "Split the area into trapezoid strips: average height × width for each, then add them up.");
  const tot = S.add("text", {x: sx, y: sy0 + o.strips.length*fsz*1.7 + 8, text: `area = ${f2(o.area, 1)} %·${D.unit}`, size: fsz + 1, align: "left", weight: 700}); S.fadeIn(tot, t);
  const rr = S.add("text", {x: sx, y: sy0 + o.strips.length*fsz*1.7 + 8 + fsz*2, text: `R = ${f2(o.area, 1)} ÷ ${f2(D.H)} = ${pct(o.R)}`, size: L.tall ? fsz + 1 : fsz + 4, align: "left", weight: 700, color: COL.gold});
  S.fadeIn(rr, t + 1.0);
  S.cue(t, `Divide by the depth: the ${f2(p, 0)} % curve gives ${pct(o.R)} overall removal. Then do all of this again for every other curve.`);
  t += 2.2;
  S.dur = t + 3;
};

/* 8. Design values, and the tally */
SCENES.design = (S, L) => {
  const rows = D.design, n = rows.length;
  const G = L.tall ? grid(16, 40, L.W - 32, Math.min(330, 38*(n + 1)), n, 4, {maxFont: 17, headW: 70})
                   : grid(16, 70, 470, Math.min(380, 44*(n + 1)), n, 4, {maxFont: 17, headW: 70});
  const fs = G.font;
  drawGridLines(S, G, 0, 0.6);
  ["detention (min)", "overall R (%)", `overflow (${D.unit}/d)`, "detention (h)"].forEach((h, j) => { const [x, y] = G.colHead(j); const a = S.add("text", {x, y, text: h, size: fs*0.72, color: COL.muted}); S.fadeIn(a, 0.3 + 0.1*j); });
  const cellsV = [], cellsH = [];
  rows.forEach((r, i) => { const [hx, hy] = G.rowHead(i); const k = D.levels.indexOf(r.p);
    const h = S.add("text", {x: hx, y: hy, text: `${f2(r.p, 0)} %`, size: fs*0.9, color: D.colors[k], weight: 700}); S.fadeIn(h, 0.5 + 0.06*i);
    const a = S.add("text", {x: G.cell(i, 0)[0], y: hy, text: f2(r.t, 1), size: fs}); S.fadeIn(a, 0.6 + 0.06*i);
    const b = S.add("text", {x: G.cell(i, 1)[0], y: hy, text: f2(r.R, 1), size: fs}); S.fadeIn(b, 0.7 + 0.06*i);
    cellsV.push(S.add("text", {x: G.cell(i, 2)[0], y: hy, text: "", size: fs, color: COL.blue}));
    cellsH.push(S.add("text", {x: G.cell(i, 3)[0], y: hy, text: "", size: fs, color: COL.blue})); });
  S.cue(0, "Collect each curve's detention time and overall removal.");
  const fx = L.tall ? L.W/2 : 500 + (L.W - 500)/2;
  const eq = S.add("text", {x: L.tall ? L.W/2 : G.x + G.w/2, y: G.y + G.h + 34, text: `overflow rate = H ÷ t × 1440 = ${f2(D.H)} ÷ t × 1440`, size: fs, color: COL.fg}); S.fadeIn(eq, 1.6);
  let t = 2.4;
  rows.forEach((r, i) => { S.count(cellsV[i], 0, r.v, t + 0.25*i, 0.6, v => f2(v, 1)); S.count(cellsH[i], 0, r.th, t + 0.25*i, 0.6, v => f2(v, 2)); });
  S.cue(1.6, "Overflow rate is the column depth divided by the time (×1440 to turn minutes into days). Detention time is just the time in hours.");
  t += 0.25*n + 1;
  // the two design plots
  const pw = L.tall ? (L.W - 80)/2 : (L.W - 560), ph = L.tall ? 230 : 150;
  const boxes = L.tall ? [{x: 40, y: 480}, {x: 60 + pw, y: 480}] : [{x: 530, y: 90}, {x: 530, y: 320}];
  const vmax = Math.max(...rows.map(r => r.v))*1.08, hmax = Math.max(...rows.map(r => r.th))*1.08;
  const Rlo = Math.max(0, Math.min(...rows.map(r => r.R)) - 8), Rhi = Math.min(100, Math.max(...rows.map(r => r.R)) + 5);
  [[`overflow rate (${D.unit}/d)`, "v", vmax], ["detention time (h)", "th", hmax]].forEach(([lab, key, mx], q) => {
    const b = boxes[q], X = v => b.x + pw*v/mx, Y = R => b.y + ph - ph*(R - Rlo)/(Rhi - Rlo);
    const ax1 = S.add("line", {x1: b.x, y1: b.y + ph, x2: b.x + pw, y2: b.y + ph, color: COL.muted, lw: 1.5}); S.create(ax1, t, 0.6);
    const ax2 = S.add("line", {x1: b.x, y1: b.y + ph, x2: b.x, y2: b.y, color: COL.muted, lw: 1.5}); S.create(ax2, t, 0.6);
    const xl = S.add("text", {x: b.x + pw/2, y: b.y + ph + 18, text: lab, size: 13, color: COL.muted}); S.fadeIn(xl, t + 0.2);
    const yl = S.add("text", {x: b.x + 4, y: b.y - 12, text: "overall R (%)", size: 13, color: COL.muted, align: "left"}); S.fadeIn(yl, t + 0.2);
    const pts = [];
    rows.slice().sort((a, c) => a[key] - c[key]).forEach((r, m) => { const i = rows.indexOf(r), k = D.levels.indexOf(r.p);
      const src = key === "v" ? cellsV[i] : cellsH[i];
      const c = S.add("circle", {x: src.p0.x, y: src.p0.y, r: 5, fill: D.colors[k], opacity: 0}); S.set(c, {opacity: 1}, t + 0.6 + 0.12*m);
      S.fly(c, X(r[key]), Y(r.R), t + 0.6 + 0.12*m, 0.9, 30); pts.push([X(r[key]), Y(r.R)]); });
    const pa = S.add("path", {pts, color: COL.muted, lw: 1.5, z: -1}); S.create(pa, t + 0.8 + 0.12*n, 0.8); });
  S.cue(t, "Plot overall removal against each, and read off the overflow rate and detention time for your target removal.");
  t += 2.6 + 0.12*n;
  // the tally
  const T = D.tally;
  const veil = S.add("rect", {x: 0, y: 0, w: L.W, h: L.H, fill: "#0f1115", opacity: 0, z: 20}); S.to(veil, {opacity: 0.93}, t, 0.8);
  const lines = [[`${T.conversions}`, "percent-removal conversions"], [`${T.easy}`, "cells copied straight across"],
                 [`${T.interpolations}`, "interpolations"], [`${T.bottom_extrap}`, "extrapolations"],
                 [`${T.trapezoids}`, "trapezoid strips"], [`≈ ${T.arithmetic}`, "arithmetic steps in all"]];
  const ty = L.tall ? 190 : 110, gap = L.tall ? 70 : 52, nx = L.tall ? 190 : L.W/2 - 16;
  const head = S.add("text", {x: L.W/2, y: ty - 60, text: "By hand, for this one dataset:", size: 22, weight: 700, z: 21}); S.fadeIn(head, t + 0.6);
  lines.forEach(([num, txt], i) => { const yy = ty + i*gap;
    const a = S.add("text", {x: nx, y: yy, text: "0", size: 30, weight: 700, color: COL.gold, align: "right", z: 21}); S.set(a, {opacity: 0}, 0); S.to(a, {opacity: 1}, t + 1 + 0.3*i, 0.3);
    const v = parseInt(num.replace(/[^0-9]/g, ""), 10); S.count(a, 0, v, t + 1 + 0.3*i, 0.8, x => (num.startsWith("≈") ? "≈ " : "") + Math.round(x), "out");
    const b = S.add("text", {x: nx + 16, y: yy, text: txt, size: L.tall ? 15 : 17, align: "left", z: 21}); S.fadeIn(b, t + 1 + 0.3*i, 0.4, 0); });
  const toolLines = L.tall ? ["The tool: all of it, more precisely,", "in well under a second."] : ["The tool: all of it, more precisely, in well under a second."];
  toolLines.forEach((s2, i) => { const tool = S.add("text", {x: L.W/2, y: ty + lines.length*gap + 20 + i*32, text: s2, size: L.tall ? 16 : 19, weight: 700, color: COL.blue, z: 21});
    S.fadeIn(tool, t + 1 + 0.3*lines.length + 0.8); });
  S.cue(t + 0.6, `That's ${T.conversions} conversions, ${T.interpolations} interpolations, ${T.bottom_extrap} extrapolations and ${T.trapezoids} trapezoid strips — about ${T.arithmetic} arithmetic steps — for one dataset. The tool does it in well under a second.`);
  S.dur = t + 1 + 0.3*lines.length + 4;
};

/* ---------- player ---------- */
function boot(data, sceneName){
  D = data;
  const stage = document.getElementById("stage"), cv = document.getElementById("cv"), g = cv.getContext("2d");
  const cap = document.getElementById("cap"), playB = document.getElementById("play"), rst = document.getElementById("restart");
  const sl = document.getElementById("t"), spd = document.getElementById("speed"), all = document.getElementById("all");
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let S = null, L = null, t = 0, playing = false, last = null, mode = null, lastCue = "";
  function build(){
    const w = stage.clientWidth, h = stage.clientHeight; const tall = w/h < 1.15;
    if (S && tall === (mode === "tall")) return;
    mode = tall ? "tall" : "wide";
    L = tall ? {W: 600, H: 820, tall: true} : {W: 960, H: 540, tall: false};
    L.reduced = reduce;
    MUL = tall ? 1.3 : 1; S = new Scene(g, MUL); SCENES[sceneName](S, L);
    sl.max = S.dur; sl.step = 0.05;
    all.innerHTML = S.cues.map(c => `<li>${c[1]}</li>`).join("");
  }
  function fit(){
    build();
    const dpr = window.devicePixelRatio || 1, w = stage.clientWidth, h = stage.clientHeight;
    const s = Math.min(w/L.W, h/L.H), cw = L.W*s, ch = L.H*s;
    cv.style.width = cw + "px"; cv.style.height = ch + "px"; cv.width = Math.round(cw*dpr); cv.height = Math.round(ch*dpr);
    g.setTransform(s*dpr, 0, 0, s*dpr, 0, 0); frame();
  }
  function frame(){ render(g, S, t, L); sl.value = t;
    let c = ""; for (const [ct, s] of S.cues) if (ct <= t + 1e-6) c = s;
    if (c !== lastCue){ cap.textContent = c; lastCue = c; } }
  function tick(ts){ if (!playing) return; if (last != null) t = Math.min(S.dur, t + (ts - last)/1000*(+spd.value)); last = ts; frame();
    if (t >= S.dur) { stop(); return; } requestAnimationFrame(tick); }
  function play(){ if (t >= S.dur) t = 0; playing = true; last = null; playB.textContent = "❚❚ Pause"; playB.setAttribute("aria-pressed", "true"); requestAnimationFrame(tick); }
  function stop(){ playing = false; playB.textContent = "▶ Play"; playB.setAttribute("aria-pressed", "false"); }
  playB.onclick = () => playing ? stop() : play();
  rst.onclick = () => { t = 0; frame(); play(); };
  sl.oninput = () => { stop(); t = +sl.value; frame(); };
  window.addEventListener("resize", fit);
  window.__anim = {render: tt => { t = clamp(tt, 0, S.dur); frame(); }, get duration(){ return S.dur; }};
  const go = () => { fit(); if (reduce) { t = S.dur; frame(); cap.textContent = "Showing the finished picture (reduced motion is on). Press Play to watch it."; return; }
    if ("IntersectionObserver" in window){ const io = new IntersectionObserver(es => { if (es[0].isIntersecting){ io.disconnect(); setTimeout(play, 250); } }, {threshold: 0.6}); io.observe(stage); }
    else play(); };
  (document.fonts ? document.fonts.load(`400 16px ${FONT}`).then(go, go) : go());
}
