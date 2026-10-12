"""Small pictures and further-reading links for the explainer terms (shared by both CE 323 tools).

Each picture is a tiny inline SVG drawn for these tools (no outside images), shown on a white card at the top of
the explainer box. They are decorative: the explainer text says the same thing in words, so screen readers skip
them. Links were checked when added; each opens in a new tab.
"""

INK, BLUE, ORANGE, WATER, SLUDGE, GREY, PALE = "#1b1a17", "#385F96", "#B8481A", "#dbe6f3", "#8c6b5a", "#857e6e", "#9EB8DB"

# ---------------------------------------------------------------------------
# Further reading
# ---------------------------------------------------------------------------
SLIDES = ("CE 323 lecture slides", "https://sedimentationanalysis.my.canva.site/")
TUD_LAB = ("TU Delft lab module: flocculent settling",
           "https://ocw.tudelft.nl/wp-content/uploads/Sedimentation-Flocculent-settling.pdf")
TUD_SED = ("TU Delft lecture: sedimentation", "https://ocw.tudelft.nl/wp-content/uploads/Sedimentation.pdf")
NPTEL = ("NPTEL: primary sedimentation tanks (PDF)",
         "https://archive.nptel.ac.in/content/storage2/courses/105105048/M16L21.pdf")
WIKI_SED = ("Wikipedia: sedimentation (water treatment)", "https://en.wikipedia.org/wiki/Sedimentation_(water_treatment)")


def _w(title, slug):
    return (f"Wikipedia: {title}", f"https://en.wikipedia.org/wiki/{slug}")


LINKS = {
    "column": [SLIDES, TUD_LAB],
    "port": [SLIDES],
    "turbidity": [("USGS: turbidity and water",
                   "https://www.usgs.gov/special-topics/water-science-school/science/turbidity-and-water")],
    "tss": [_w("total suspended solids", "Total_suspended_solids")],
    "removal": [SLIDES],
    "iso": [NPTEL, TUD_LAB],
    "interp": [_w("linear interpolation", "Linear_interpolation")],
    "extrap": [_w("extrapolation", "Extrapolation")],
    "discrete": [_w("Stokes' law", "Stokes%27_law"), WIKI_SED],
    "floc": [_w("flocculation", "Flocculation"), TUD_LAB],
    "detention": [_w("hydraulic retention time", "Hydraulic_retention_time")],
    "overflow": [WIKI_SED, TUD_SED],
    "overall": [NPTEL],
    "trapezoid": [_w("trapezoidal rule", "Trapezoidal_rule")],
    "hazen": [TUD_SED, WIKI_SED],
    "lw": [TUD_SED],
    "weir": [_w("weir", "Weir"), _w("clarifier", "Clarifier")],
    "launder": [_w("clarifier", "Clarifier")],
    "vh": [TUD_SED],
    "weirload": [("Virginia Waterworks Regulations, 12VAC5-590-872",
                  "https://law.lis.virginia.gov/admincode/title12/agency5/chapter590/section872/")],
    "re": [_w("Reynolds number", "Reynolds_number")],
    "fr": [_w("Froude number", "Froude_number"), TUD_SED],
    "scale": [NPTEL],
    "sf": [_w("factor of safety", "Factor_of_safety")],
}


# ---------------------------------------------------------------------------
# Pictures (viewBox 200 x 100)
# ---------------------------------------------------------------------------
def _svg(body, h=100):
    return (f'<svg class="pic" viewBox="0 0 200 {h}" aria-hidden="true" focusable="false" '
            f'xmlns="http://www.w3.org/2000/svg">{body}</svg>')


def _t(x, y, s, anchor="middle", size=9, color=INK, weight=400):
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}">{s}</text>')


def _dots(pts, r=2.2, color=BLUE):
    return "".join(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}"/>' for x, y in pts)


def _arrow(x1, y1, x2, y2, color=INK, w=1.5):
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    hx1, hy1 = x2 - 6 * math.cos(a - 0.45), y2 - 6 * math.sin(a - 0.45)
    hx2, hy2 = x2 - 6 * math.cos(a + 0.45), y2 - 6 * math.sin(a + 0.45)
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{w}"/>'
            f'<path d="M{hx1:.1f},{hy1:.1f} L{x2},{y2} L{hx2:.1f},{hy2:.1f}" fill="none" stroke="{color}" '
            f'stroke-width="{w}"/>')


def _dim(x, y1, y2, label, side="left"):
    """Vertical dimension line with a label."""
    tx = x - 4 if side == "left" else x + 4
    return (_arrow(x, (y1 + y2) / 2, x, y1, w=1) + _arrow(x, (y1 + y2) / 2, x, y2, w=1)
            + _t(tx, (y1 + y2) / 2 + 3, label, "end" if side == "left" else "start"))


def _scatter(x0, x1, y0, y1, n, seed, r=2):
    import random
    g = random.Random(seed)
    return _dots([(round(g.uniform(x0, x1), 1), round(g.uniform(y0, y1), 1)) for _ in range(n)], r)


def _column(x=80, w=30, top=8, bot=92, ports=(28, 48, 68, 88), fill=True):
    s = f'<rect x="{x}" y="{top}" width="{w}" height="{bot - top}" fill="{WATER if fill else "#fff"}" stroke="{INK}" stroke-width="1.5"/>'
    for y in ports:
        s += (f'<line x1="{x + w}" y1="{y}" x2="{x + w + 9}" y2="{y}" stroke="{INK}" stroke-width="1.5"/>'
              f'<circle cx="{x + w + 11}" cy="{y}" r="2.2" fill="#fff" stroke="{INK}" stroke-width="1.2"/>')
    return s


def _axes(x0=34, y0=12, x1=190, y1=86, xl="time", yl="depth"):
    return (_arrow(x0, y0, x1, y0, GREY, 1) + _arrow(x0, y0, x0, y1, GREY, 1)
            + _t(x1 - 2, y0 - 3, xl, "end", 8, GREY) + _t(x0 - 4, y1 - 2, yl, "end", 8, GREY))


def _side_basin(x=20, y=22, w=160, h=56, fb=8, sludge=True):
    s = (f'<rect x="{x}" y="{y + fb}" width="{w}" height="{h - fb}" fill="{WATER}"/>'
         f'<line x1="{x}" y1="{y + fb}" x2="{x + w}" y2="{y + fb}" stroke="{BLUE}" stroke-width="1.5"/>')
    if sludge:
        s += f'<rect x="{x}" y="{y + h - 5}" width="{w}" height="5" fill="{SLUDGE}"/>'
    s += f'<path d="M{x},{y} V{y + h} H{x + w} V{y}" fill="none" stroke="{INK}" stroke-width="1.8"/>'
    return s


def _pic_column():
    return _svg(_column() + _scatter(83, 107, 12, 90, 26, 1)
                + _dim(70, 8, 88, "2.5 m") + _t(124, 31, "ports", "start", 8)
                + _t(158, 60, "still water,", "middle", 8, GREY) + _t(158, 70, "sampled", "middle", 8, GREY)
                + _t(158, 80, "over time", "middle", 8, GREY))


def _pic_port():
    body = (f'<rect x="40" y="6" width="44" height="88" fill="{WATER}" stroke="{INK}" stroke-width="1.5"/>'
            + _scatter(44, 80, 10, 90, 18, 2)
            + f'<rect x="84" y="44" width="30" height="7" fill="#fff" stroke="{INK}" stroke-width="1.5"/>'
            + f'<rect x="96" y="38" width="6" height="6" fill="{INK}"/>'
            + f'<path d="M112,51 q2,8 0,14" fill="none" stroke="{BLUE}" stroke-width="2"/>'
            + f'<path d="M100,70 h26 l-3,24 h-20 z" fill="#fff" stroke="{INK}" stroke-width="1.5"/>'
            + f'<rect x="103.5" y="82" width="19" height="11" fill="{PALE}"/>'
            + _t(150, 46, "tap at a", "start", 8) + _t(150, 56, "known depth", "start", 8)
            + _t(150, 86, "sample", "start", 8))
    return _svg(body)


def _pic_turbidity():
    body = (f'<rect x="6" y="40" width="24" height="18" rx="3" fill="{GOLD_SOFT}" stroke="{INK}" stroke-width="1.3"/>'
            + _t(18, 70, "light", "middle", 8)
            + f'<rect x="78" y="22" width="40" height="56" fill="{WATER}" stroke="{INK}" stroke-width="1.5"/>'
            + _scatter(82, 114, 26, 74, 16, 3)
            + _arrow(30, 49, 78, 49, ORANGE, 2) + _arrow(118, 49, 150, 49, ORANGE, 1)
            + _arrow(98, 48, 98, 12, ORANGE, 1.4) + _arrow(100, 51, 126, 76, ORANGE, 1) + _arrow(96, 51, 72, 82, ORANGE, 1)
            + f'<rect x="88" y="2" width="20" height="9" fill="{INK}"/>' + _t(112, 10, "detector at 90°", "start", 8)
            + _t(158, 44, "more", "start", 8) + _t(158, 54, "particles,", "start", 8)
            + _t(158, 64, "more light", "start", 8) + _t(158, 74, "scattered", "start", 8))
    return _svg(body)


def _pic_tss():
    body = (f'<path d="M40,10 h60 l-24,34 v14 h-12 v-14 z" fill="{WATER}" stroke="{INK}" stroke-width="1.5"/>'
            + f'<path d="M48,22 h44" stroke="{SLUDGE}" stroke-width="3"/>' + _dots([(55, 20), (64, 19), (73, 20), (83, 20)], 2, SLUDGE)
            + f'<path d="M52,64 h36 v28 h-36 z" fill="#fff" stroke="{INK}" stroke-width="1.5"/>'
            + f'<rect x="53" y="76" width="34" height="15" fill="{WATER}"/>'
            + _t(102, 24, "filter keeps", "start", 8) + _t(102, 34, "the particles", "start", 8)
            + f'<rect x="132" y="66" width="44" height="6" fill="{INK}"/><rect x="150" y="72" width="8" height="14" fill="{INK}"/>'
            + f'<rect x="138" y="58" width="32" height="8" fill="{SLUDGE}"/>' + _t(154, 98, "dry and weigh", "middle", 8)
            + _t(70, 99, "known volume", "middle", 8))
    return _svg(body, 102)


def _pic_c0():
    import random
    g = random.Random(5)
    pts = [(round(84 + (i % 6) * 4.6 + g.uniform(-1, 1), 1), round(12 + (i // 6) * 6.2 + g.uniform(-1, 1), 1)) for i in range(78)]
    return _svg(_column(ports=()) + _dots(pts, 1.7) + _t(150, 44, "t = 0:", "middle", 9, INK, 700)
                + _t(150, 56, "evenly mixed,", "middle", 8) + _t(150, 66, "reading = C₀", "middle", 8))


def _pic_removal():
    import random
    g = random.Random(7)
    full = [(round(g.uniform(33, 57), 1), round(g.uniform(12, 90), 1)) for _ in range(40)]
    later = [(x + 90, y) for x, y in full if y > 30 + (y - 12) * 0.0 and g.random() < (0.25 if y < 50 else 0.75)]
    body = (_column(30, 30, ports=(50,)) + _column(120, 30, ports=(50,)) + _dots(full, 1.8) + _dots(later, 1.8)
            + _t(45, 99, "t = 0: C₀", "middle", 8) + _t(135, 99, "later: C", "middle", 8)
            + f'<rect x="28" y="44" width="56" height="12" fill="none" stroke="{ORANGE}" stroke-width="1.3" stroke-dasharray="3 2"/>'
            + f'<rect x="118" y="44" width="56" height="12" fill="none" stroke="{ORANGE}" stroke-width="1.3" stroke-dasharray="3 2"/>'
            + _arrow(88, 50, 114, 50, GREY, 1))
    return _svg(body, 102)


def _pic_iso():
    body = _axes(34, 12, 190, 90)
    for (c, w, ex, lab) in [("#7a9a91", 1.5, 52, "20 %"), (BLUE, 2.6, 104, "50 %"), ("#1a2a5c", 1.5, 156, "80 %")]:
        body += f'<path d="M34,12 Q{34 + ex * 0.95},{16 + ex * 0.06} {34 + ex},78" fill="none" stroke="{c}" stroke-width="{w}"/>'
        body += _t(34 + ex, 88, lab, "middle", 8, c, 700 if lab == "50 %" else 400)
    return _svg(body, 100)


def _pic_interp():
    body = (_arrow(30, 88, 190, 88, GREY, 1) + _arrow(30, 88, 30, 8, GREY, 1)
            + _t(188, 98, "depth", "end", 8, GREY) + _t(34, 10, "removal", "start", 8, GREY)
            + f'<line x1="60" y1="24" x2="160" y2="70" stroke="{BLUE}" stroke-width="2"/>'
            + _dots([(60, 24), (160, 70)], 4) + f'<circle cx="114" cy="48.8" r="4" fill="#fff" stroke="{ORANGE}" stroke-width="2"/>'
            + f'<line x1="114" y1="52" x2="114" y2="88" stroke="{ORANGE}" stroke-dasharray="3 2"/>'
            + f'<line x1="30" y1="48.8" x2="110" y2="48.8" stroke="{ORANGE}" stroke-dasharray="3 2"/>'
            + _t(64, 18, "50 % at 0.5 m", "start", 8) + _t(160, 82, "35 % at 1.0 m", "middle", 8)
            + _t(120, 44, "40 % ≈ 0.83 m", "start", 8, ORANGE, 700))
    return _svg(body, 100)


def _pic_extrap():
    body = (_axes(30, 12, 190, 90)
            + f'<polyline points="30,12 60,26 90,38 120,50" fill="none" stroke="{BLUE}" stroke-width="2"/>'
            + _dots([(60, 26), (90, 38), (120, 50)], 3.5)
            + f'<line x1="120" y1="50" x2="176" y2="72" stroke="{BLUE}" stroke-width="2" stroke-dasharray="2 3"/>'
            + f'<circle cx="176" cy="72" r="4" fill="#fff" stroke="{BLUE}" stroke-width="2"/>'
            + _t(122, 40, "last reading", "start", 8) + _t(170, 86, "guess", "end", 8, GREY))
    return _svg(body, 100)


def _pic_discrete():
    body = ""
    for i in range(6):
        body += f'<circle cx="60" cy="{12 + i * 14}" r="4" fill="{BLUE}" opacity="{0.25 + 0.15 * i:.2f}"/>'
    body += (_arrow(80, 14, 80, 86, GREY, 1) + _t(92, 40, "same size,", "start", 9)
             + _t(92, 52, "same speed:", "start", 9) + _t(92, 64, "equal steps,", "start", 9)
             + _t(92, 76, "straight path", "start", 9))
    return _svg(body)


def _pic_floc():
    body = _dots([(40, 12), (48, 14), (56, 11), (66, 13), (74, 12)], 1.8)
    body += _dots([(48, 26), (52, 27), (68, 27), (71, 26)], 2.4) + _dots([(56, 44), (61, 45), (65, 43)], 3.4)
    body += f'<circle cx="60" cy="72" r="7" fill="{BLUE}"/>'
    body += (_arrow(92, 14, 92, 88, GREY, 1) + _t(104, 30, "particles stick", "start", 9)
             + _t(104, 42, "together, grow", "start", 9) + _t(104, 54, "and speed up:", "start", 9)
             + _t(104, 66, "gaps get longer", "start", 9))
    return _svg(body)


def _pic_detention():
    body = (_side_basin() + _arrow(2, 40, 22, 40, BLUE, 2) + _arrow(178, 40, 198, 40, BLUE, 2)
            + f'<circle cx="100" cy="54" r="13" fill="#fff" stroke="{INK}" stroke-width="1.5"/>'
            + f'<path d="M100,54 V45 M100,54 L107,58" stroke="{INK}" stroke-width="1.5"/>'
            + _t(100, 94, "time in the basin = volume ÷ flow", "middle", 9) + _t(10, 34, "Q", "middle", 9, BLUE, 700))
    return _svg(body)


def _pic_overflow():
    body = (f'<path d="M30,62 L70,34 H180 L140,62 Z" fill="{WATER}" stroke="{INK}" stroke-width="1.5"/>'
            + _t(105, 52, "surface area A", "middle", 9))
    for x in (60, 90, 120, 150):
        body += _arrow(x, 44, x, 20, BLUE, 1.2)
    body += (_t(100, 12, "flow Q leaves through the surface", "middle", 8, BLUE)
             + _t(100, 84, "overflow rate = Q ÷ A", "middle", 9, INK, 700)
             + _t(100, 96, "(a speed, in m/d)", "middle", 8, GREY))
    return _svg(body)


def _pic_overall():
    body = (_arrow(40, 10, 190, 10, GREY, 1) + _arrow(40, 10, 40, 92, GREY, 1)
            + f'<path d="M40,10 L170,10 L132,34 L120,58 L112,86 L40,86 Z" fill="{PALE}" opacity="0.7"/>'
            + f'<polyline points="170,10 132,34 120,58 112,86" fill="none" stroke="{BLUE}" stroke-width="2"/>'
            + _t(188, 7, "removal", "end", 8, GREY) + _t(36, 90, "H", "end", 9)
            + _t(78, 52, "area", "middle", 9, INK, 700) + _t(150, 70, "overall =", "start", 8)
            + _t(150, 80, "area ÷ H", "start", 8))
    return _svg(body)


def _pic_trapezoid():
    body = (_arrow(20, 88, 190, 88, GREY, 1) + _arrow(20, 88, 20, 8, GREY, 1))
    xs, ys = [30, 75, 120, 165], [30, 44, 52, 70]
    fills = [PALE, "#c9d8ec", PALE]
    for i in range(3):
        body += (f'<path d="M{xs[i]},88 V{ys[i]} L{xs[i + 1]},{ys[i + 1]} V88 Z" fill="{fills[i]}" stroke="#fff" stroke-width="1.5"/>')
    body += f'<polyline points="{" ".join(f"{x},{y}" for x, y in zip(xs, ys))}" fill="none" stroke="{BLUE}" stroke-width="2"/>'
    body += _t(52, 70, "strip", "middle", 8) + _t(105, 18, "area = width × average height", "middle", 8)
    return _svg(body)


def _pic_strat():
    import random
    g = random.Random(11)
    pts = [(round(g.uniform(83, 107), 1), round(y, 1)) for y in [g.triangular(12, 90, 90) for _ in range(46)]]
    return _svg(f'<defs><linearGradient id="gst" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#f4f8fc"/>'
                f'<stop offset="1" stop-color="#b9cbe2"/></linearGradient></defs>'
                f'<rect x="60" y="8" width="30" height="84" fill="url(#gst)" stroke="{INK}" stroke-width="1.5"/>'
                + _dots([(x - 20, y) for x, y in pts], 1.8) + _t(104, 18, "clearer at the top", "start", 8)
                + _t(104, 88, "cloudier at the bottom", "start", 8))


def _pic_h():
    return _svg(_column(ports=(28, 48, 68, 88)) + _dim(70, 8, 88, "H") + _t(124, 91, "deepest port", "start", 8)
                + _t(124, 12, "water surface", "start", 8))


def _pic_section(hl):
    """Basin cross-section with side-water depth and freeboard; hl picks the one to highlight."""
    c_swd, c_fb = (ORANGE, INK) if hl == "swd" else (INK, ORANGE)
    body = (f'<rect x="60" y="30" width="90" height="62" fill="{WATER}"/>'
            f'<line x1="60" y1="30" x2="150" y2="30" stroke="{BLUE}" stroke-width="1.5"/>'
            f'<path d="M60,12 V92 H150 V12" fill="none" stroke="{INK}" stroke-width="2"/>'
            + _arrow(48, 21, 48, 12, c_fb, 1) + _arrow(48, 21, 48, 30, c_fb, 1)
            + _arrow(48, 61, 48, 30, c_swd, 1) + _arrow(48, 61, 48, 92, c_swd, 1)
            + _t(42, 24, "freeboard", "end", 8, c_fb, 700 if hl == "fb" else 400)
            + _t(42, 64, "side-water", "end", 8, c_swd, 700 if hl == "swd" else 400)
            + _t(42, 74, "depth", "end", 8, c_swd, 700 if hl == "swd" else 400)
            + _t(156, 30, "water", "start", 8, BLUE) + _t(156, 40, "surface", "start", 8, BLUE))
    return _svg(body)


def _pic_lw():
    body = (f'<rect x="20" y="24" width="160" height="40" fill="{WATER}" stroke="{INK}" stroke-width="1.5"/>'
            + _arrow(70, 44, 130, 44, BLUE, 2) + _t(100, 39, "flow", "middle", 8, BLUE)
            + _arrow(100, 78, 20, 78, INK, 1) + _arrow(100, 78, 180, 78, INK, 1) + _t(100, 92, "length L", "middle", 9)
            + _arrow(190, 44, 190, 24, INK, 1) + _arrow(190, 44, 190, 64, INK, 1) + _t(194, 18, "W", "end", 9)
            + _t(100, 14, "L ÷ W of 3 to 5: long and narrow", "middle", 8))
    return _svg(body)


def _pic_weir():
    body = (f'<rect x="10" y="36" width="80" height="56" fill="{WATER}"/>'
            f'<line x1="10" y1="36" x2="90" y2="36" stroke="{BLUE}" stroke-width="1.5"/>'
            f'<path d="M10,92 H90 V30" fill="none" stroke="{INK}" stroke-width="2"/>'
            f'<path d="M90,34 q14,4 18,30" fill="none" stroke="{BLUE}" stroke-width="2.4"/>'
            f'<path d="M96,62 V92 H176 V62" fill="none" stroke="{INK}" stroke-width="1.5"/>'
            f'<rect x="97" y="78" width="78" height="13" fill="{WATER}"/>'
            + _arrow(36, 50, 78, 50, BLUE, 1.4) + _t(50, 72, "basin", "middle", 9) + _t(136, 72, "channel", "middle", 9)
            + _t(92, 22, "water spills over the weir edge", "middle", 8))
    return _svg(body)


def _pic_launder():
    body = (f'<rect x="20" y="16" width="160" height="64" fill="{WATER}" stroke="{INK}" stroke-width="1.5"/>'
            f'<rect x="132" y="16" width="12" height="64" fill="#fff" stroke="{INK}" stroke-width="1.3"/>'
            + _arrow(112, 30, 130, 30, BLUE, 1.3) + _arrow(112, 48, 130, 48, BLUE, 1.3) + _arrow(112, 66, 130, 66, BLUE, 1.3)
            + _arrow(164, 30, 146, 30, BLUE, 1.3) + _arrow(164, 48, 146, 48, BLUE, 1.3) + _arrow(164, 66, 146, 66, BLUE, 1.3)
            + _arrow(138, 80, 138, 96, BLUE, 1.6) + _t(70, 50, "plan view", "middle", 8, GREY)
            + _t(126, 12, "launder: weir on both sides", "middle", 8))
    return _svg(body)


def _pic_vh():
    body = _side_basin(20, 14, 160, 60, 6)
    for y in (30, 44, 58):
        body += _arrow(60, y, 140, y, BLUE, 1.4)
    body += (_t(100, 88, "v = Q ÷ (W × depth)", "middle", 9, INK, 700)
             + _t(100, 99, "too fast stirs the sludge back up", "middle", 8, GREY))
    return _svg(body, 102)


def _pic_weirload():
    body = (f'<line x1="20" y1="40" x2="180" y2="40" stroke="{INK}" stroke-width="3"/>'
            + "".join(_arrow(x, 22, x, 56, BLUE, 1.2) for x in range(30, 180, 16))
            + _arrow(100, 72, 20, 72, INK, 1) + _arrow(100, 72, 180, 72, INK, 1)
            + _t(100, 86, "weir length", "middle", 9) + _t(100, 14, "flow shared along the weir", "middle", 8)
            + _t(100, 99, "loading = Q ÷ length", "middle", 8, GREY))
    return _svg(body, 102)


def _pic_re():
    body = "".join(f'<line x1="10" y1="{y}" x2="90" y2="{y}" stroke="{BLUE}" stroke-width="1.5"/>' for y in (26, 38, 50, 62))
    body += (f'<path d="M110,30 c10,-10 18,10 28,0 s14,16 22,4 s10,-8 26,2" fill="none" stroke="{BLUE}" stroke-width="1.5"/>'
             f'<path d="M110,48 c8,8 14,-12 26,0 c6,6 2,14 10,10 c12,-6 16,-14 30,-4" fill="none" stroke="{BLUE}" stroke-width="1.5"/>'
             f'<path d="M110,64 c14,-6 12,10 24,4 c10,-5 10,-14 22,-6 s12,6 20,0" fill="none" stroke="{BLUE}" stroke-width="1.5"/>'
             + _t(50, 84, "small Re: smooth", "middle", 8) + _t(148, 84, "large Re: turbulent", "middle", 8))
    return _svg(body)


def _pic_fr():
    body = (_side_basin(20, 20, 160, 56, 10, False)
            + f'<path d="M20,30 q20,-5 40,0 t40,0 t40,0 t40,0" fill="none" stroke="{BLUE}" stroke-width="1.5"/>'
            + _arrow(40, 12, 90, 12, GREY, 1.4) + _t(96, 15, "wind", "start", 8, GREY)
            + f'<path d="M150,64 C120,72 80,48 50,60" fill="none" stroke="{ORANGE}" stroke-width="1.4" stroke-dasharray="3 2"/>'
            + _t(100, 92, "slow flow is easily pushed around", "middle", 8))
    return _svg(body)


def _pic_scale():
    body = (_column(24, 18, 10, 88, ports=()) + _scatter(26, 40, 14, 86, 10, 13, 1.6)
            + _t(33, 98, "lab: still", "middle", 8)
            + _side_basin(70, 30, 120, 50, 6)
            + f'<path d="M92,52 c0,-10 16,-10 16,0 s-12,8 -10,0" fill="none" stroke="{ORANGE}" stroke-width="1.4"/>'
            + f'<path d="M150,60 c0,-10 16,-10 16,0 s-12,8 -10,0" fill="none" stroke="{ORANGE}" stroke-width="1.4"/>'
            + _arrow(80, 22, 120, 22, GREY, 1.2) + _t(124, 25, "wind", "start", 8, GREY)
            + _t(130, 98, "real basin: currents and eddies", "middle", 8))
    return _svg(body, 102)


def _pic_sf():
    body = (f'<rect x="30" y="20" width="110" height="52" fill="{WATER}" stroke="{INK}" stroke-width="1.5"/>'
            f'<rect x="140" y="20" width="16" height="52" fill="{PALE}" stroke="{INK}" stroke-width="1.5" stroke-dasharray="3 2"/>'
            + _t(85, 50, "area needed", "middle", 9) + _t(148, 88, "+10 %", "middle", 9, INK, 700)
            + _t(100, 12, "safety factor 1.1", "middle", 8))
    return _svg(body)


def _pic_hazen():
    body = (_side_basin(20, 16, 160, 60, 6) + _arrow(2, 30, 20, 30, BLUE, 2) + _arrow(180, 30, 198, 30, BLUE, 2)
            + f'<line x1="22" y1="23" x2="178" y2="70" stroke="{INK}" stroke-width="1.5" stroke-dasharray="4 3"/>'
            + f'<circle cx="22" cy="23" r="3" fill="{BLUE}"/><circle cx="178" cy="70" r="3" fill="{BLUE}"/>'
            + _t(100, 90, "caught if it sinks H before it crosses L", "middle", 8)
            + _t(100, 100, "needs a speed ≥ Q ÷ A", "middle", 8, INK, 700))
    return _svg(body, 104)


GOLD_SOFT = "#f3d97a"

PICS = {
    "column": _pic_column(), "port": _pic_port(), "turbidity": _pic_turbidity(), "tss": _pic_tss(), "c0": _pic_c0(),
    "removal": _pic_removal(), "iso": _pic_iso(), "interp": _pic_interp(), "extrap": _pic_extrap(),
    "discrete": _pic_discrete(), "floc": _pic_floc(), "detention": _pic_detention(), "overflow": _pic_overflow(),
    "overall": _pic_overall(), "trapezoid": _pic_trapezoid(), "strat": _pic_strat(), "h": _pic_h(),
    "swd": _pic_section("swd"), "freeboard": _pic_section("fb"), "lw": _pic_lw(), "weir": _pic_weir(),
    "launder": _pic_launder(), "vh": _pic_vh(), "weirload": _pic_weirload(), "re": _pic_re(), "fr": _pic_fr(),
    "scale": _pic_scale(), "sf": _pic_sf(), "hazen": _pic_hazen(),
}
