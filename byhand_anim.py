"""HTML components for the by-hand page: one animated scene per step (engine and scenes in byhand_anim.js)."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VERSION = "1"
SCENES = ["raw", "convert", "flip", "interp", "plot", "bottom", "overall", "design"]


def _js():
    with open(os.path.join(HERE, "byhand_anim.js"), encoding="utf-8") as f:
        return f.read()


def level_colors(n):
    """Davos, light part, so every curve stands out on the dark stage (at least 3:1): low removal darker."""
    try:
        import cmcrameri.cm as cmc
        import matplotlib.colors as mc
        return [mc.to_hex(cmc.davos(x)) for x in np.linspace(0.32, 0.86, max(n, 2))[:n]]
    except Exception:
        return ["#9EB8DB"] * n


def pick_examples(A):
    """A typical hard cell (between two ports, mid-table) and a curve for the trapezoid example."""
    table, levels = A["table"], A["levels"]
    hard = [(i, k) for i, row in enumerate(table) for k, c in enumerate(row) if c.kind == "hard"]
    if not hard:
        hard = [(i, k) for i, row in enumerate(table) for k, c in enumerate(row) if c.kind == "surface"]
    mid = len(levels) / 2
    hard.sort(key=lambda ik: (abs(ik[1] - mid + 0.5), ik[0]))
    ex_hard = list(hard[0]) if hard else [1, 0]
    have = [p for p in levels if p in A["overall"]]
    ok = [p for p in have if A["bottom"][p]["kind"] != "extrap"] or have
    curve = 50.0 if 50.0 in ok else ok[len(ok) // 2]
    return {"hard": ex_hard, "curve": curve}


def payload(A, unit, munit):
    tbl = [[{"k": c.kind, "d": c.depth, **{k: c.work[k] for k in ("za", "zb", "ra", "rb", "frac") if k in c.work}}
            for c in row] for row in A["table"]]
    num = lambda v: None if v is None else float(v)
    return {
        "depths": A["depths"].tolist(), "times": A["times"].tolist(), "rows": A["rows"].tolist(),
        "conc": np.round(A["conc"], 4).tolist(), "c0": A["c0"], "H": A["H"], "levels": A["levels"],
        "R": np.round(A["R"], 4).tolist(), "unit": unit, "munit": munit, "table": tbl,
        "bottom": {f"{p:g}": {k: (num(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in b.items()
                            if k not in ("why", "eq")} for p, b in A["bottom"].items()},
        "overall": {f"{p:g}": {"t": o["t"], "area": o["area"], "R": o["R"],
                             "points": [[float(z), float(r), h] for z, r, h in o["points"]],
                             "strips": o["strips"]} for p, o in A["overall"].items()},
        "design": A["design"], "tally": A["tally"], "colors": level_colors(len(A["levels"])),
        "example": pick_examples(A),
    }


def build(scene, data_json):
    return (HTML.replace("__JS__", _js()).replace("__DATA__", data_json).replace("__SCENE__", json.dumps(scene)))


HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><style>
:root{--ink:#1b1a17;--ink2:#4d4a43;--rule:#d4ccbb;--focus:#0072b2}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:#fff;color:var(--ink);font:16px/1.45 "Open Sans",system-ui,sans-serif}
.wrap{display:flex;flex-direction:column;height:100%;border:1px solid var(--rule);border-radius:8px;overflow:hidden;background:#fff}
#stage{flex:1 1 auto;min-height:0;background:#0f1115;display:flex;align-items:center;justify-content:center}
canvas{display:block}
.under{padding:8px 12px 10px}
#cap{margin:0 0 8px;min-height:3em;font-size:15.5px;color:var(--ink)}
.bar{display:flex;gap:8px;align-items:center}
button,select{font:inherit;min-height:44px;padding:0 12px;border:1.5px solid var(--ink);background:#fff;color:var(--ink);border-radius:6px;cursor:pointer;white-space:nowrap}
button.primary{background:var(--ink);color:#fff}
input[type=range]{flex:1 1 auto;min-width:80px;height:44px;accent-color:#385F96}
:focus-visible{outline:3px solid var(--focus);outline-offset:2px}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}
details{margin-top:6px;font-size:14px;color:var(--ink2)} details ol{margin:4px 0 0;padding-left:1.3em}
</style></head><body><div class="wrap">
<div id="stage"><canvas id="cv" role="img" aria-label="Animation of this step. The captions below describe each moment; the list under 'Read this step as text' has all of them."></canvas></div>
<div class="under">
<p id="cap" aria-live="polite"></p>
<div class="bar">
  <button id="play" class="primary" type="button" aria-pressed="false">▶ Play</button>
  <button id="restart" type="button" aria-label="Play from the start">↺</button>
  <label class="sr" for="t">Position in the animation</label><input id="t" type="range" min="0" value="0">
  <label class="sr" for="speed">Speed</label>
  <select id="speed"><option value="0.6">0.6×</option><option value="1" selected>1×</option><option value="1.5">1.5×</option></select>
</div>
<details><summary>Read this step as text</summary><ol id="all"></ol></details>
</div></div>
<script>
(function(){ try {
  var p = window.parent.location.pathname.replace(/(accessibility|by-hand)\/?$/, ""); if (p.slice(-1) !== "/") p += "/";
  [["400","open-sans-latin-400-normal"],["700","open-sans-latin-700-normal"]].forEach(function(a){
    var f = new FontFace("Open Sans", "url(" + p + "app/static/fonts/" + a[1] + ".woff2)", {weight: a[0]});
    f.load().then(function(x){ document.fonts.add(x); }).catch(function(){});
  });
} catch (e) {} })();
</script>
<script>
__JS__
boot(__DATA__, __SCENE__);
</script></body></html>"""
