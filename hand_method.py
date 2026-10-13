"""The settling-column analysis done the way it is done by hand, step by step, keeping every calculation.

This mirrors the CE 323 lecture's manual method, using only the sampling times as rows:
  1. raw readings table (depth x time)
  2. percent removal for every cell: (C0 - C) / C0 x 100
  3. flip the question: a table with time down the side and removal levels across the top, each cell asking
     "how deep is this removal level at this time?"
  4. fill it in: copy a depth where the level appears exactly in the percent table (easy cell), otherwise
     interpolate down the column between the two depths that bracket it (hard cell). The surface counts as
     100 % removal. A level that is already below the bottom port is left blank.
  5. plot each column of that table as a curve, depth against time
  6. find when each curve reaches the bottom: interpolate in time along the bottom row, or extend the curve in a
     straight line through its last two points when it has not reached the bottom by the last reading
  7. overall removal for each curve: at that time, read the depth of every higher curve (interpolating between
     the table's rows, or extending past the last row), then add up trapezoid strips from the surface (100 %)
     to the bottom and divide by the depth
  8. overflow rate = H / t x 1440 and detention time = t / 60 for each curve

Every number keeps the working that produced it, so the page can show (and animate) any calculation.
"""
from dataclasses import dataclass, field

import numpy as np


def fmt(x, nd=2):
    """Short number for display: drop trailing zeros."""
    if x is None or not np.isfinite(x):
        return "–"
    s = f"{x:.{nd}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


@dataclass
class Cell:
    t: float
    p: float
    kind: str                 # "start", "easy", "hard", "surface" (hard, between the surface and the top port), "below"
    depth: float | None = None
    work: dict = field(default_factory=dict)


def percent_table(depths, times, conc, c0):
    R = (c0 - np.asarray(conc, float)) / c0 * 100
    work = [[f"({fmt(c0)} − {fmt(conc[i][j])}) ÷ {fmt(c0)} × 100 = {fmt(R[i, j], 1)} %"
             for j in range(len(times))] for i in range(len(depths))]
    return R, work


def flipped_cell(t, p, depths, Rcol):
    """Depth of removal level p in one column of the percent table (removal at each port at time t)."""
    if t == 0:
        return Cell(t, p, "start", 0.0, {"why": "At time zero nothing has settled yet, so every level starts at the surface."})
    z = np.concatenate([[0.0], depths])
    r = np.concatenate([[100.0], Rcol])
    for i in range(1, len(z)):
        if np.isclose(r[i], p):
            return Cell(t, p, "easy", float(z[i]),
                        {"why": f"The percent table has exactly {fmt(p)} % at {fmt(z[i])} at {fmt(t)} min, so copy the depth."})
        if r[i - 1] > p > r[i]:
            za, zb, ra, rb = float(z[i - 1]), float(z[i]), float(r[i - 1]), float(r[i])
            frac = (ra - p) / (ra - rb)
            d = za + frac * (zb - za)
            kind = "surface" if i == 1 else "hard"
            return Cell(t, p, kind, d, {
                "za": za, "zb": zb, "ra": ra, "rb": rb, "frac": frac,
                "why": (f"{fmt(p)} % lies between {fmt(ra, 1)} % at {fmt(za)} and {fmt(rb, 1)} % at {fmt(zb)}"
                        + (" (the surface counts as 100 %)" if i == 1 else "") + "."),
                "eq": (f"depth = {fmt(za)} + ({fmt(ra, 1)} − {fmt(p)}) ÷ ({fmt(ra, 1)} − {fmt(rb, 1)}) × "
                       f"({fmt(zb)} − {fmt(za)}) = {fmt(d)}")})
    if r[-1] > p:
        return Cell(t, p, "below", None, {"why": f"Even the bottom port has more than {fmt(p)} % removal at {fmt(t)} min, "
                                                 "so this level has already passed the bottom."})
    return Cell(t, p, "below", None, {"why": "Not reached in this column."})


def analyse(depths, times, conc, c0, levels):
    depths = np.asarray(depths, float)
    times = np.asarray(times, float)
    levels = [float(p) for p in levels]
    H = float(depths.max())
    R, conv_work = percent_table(depths, times, conc, c0)
    rows = np.concatenate([[0.0], times])
    Rfull = np.hstack([np.zeros((len(depths), 1)), R])        # removal is 0 everywhere at t = 0
    table = [[flipped_cell(t, p, depths, Rfull[:, j]) for p in levels] for j, t in enumerate(rows)]

    # 6. time each curve reaches the bottom
    rB = Rfull[-1]
    bottom = {}
    for k, p in enumerate(levels):
        hit = np.where(rB >= p - 1e-9)[0]
        if len(hit) and hit[0] > 0:
            j = hit[0]
            t0, t1, r0, r1 = rows[j - 1], rows[j], rB[j - 1], rB[j]
            if np.isclose(r1, p):
                bottom[p] = {"t": float(t1), "kind": "read", "why": f"The bottom port reads exactly {fmt(p)} % at {fmt(t1)} min."}
            else:
                tb = t0 + (t1 - t0) * (p - r0) / (r1 - r0)
                bottom[p] = {"t": float(tb), "kind": "interp", "t0": float(t0), "t1": float(t1), "r0": float(r0), "r1": float(r1),
                             "why": f"The bottom port goes from {fmt(r0, 1)} % at {fmt(t0)} min to {fmt(r1, 1)} % at {fmt(t1)} min.",
                             "eq": f"t = {fmt(t0)} + ({fmt(p)} − {fmt(r0, 1)}) ÷ ({fmt(r1, 1)} − {fmt(r0, 1)}) × ({fmt(t1)} − {fmt(t0)}) = {fmt(tb, 1)} min"}
        else:
            col = [(rows[j], table[j][k].depth) for j in range(len(rows)) if table[j][k].depth is not None]
            if len(col) >= 2 and col[-1][1] > col[-2][1]:
                (ta, za), (tb_, zb) = col[-2], col[-1]
                slope = (zb - za) / (tb_ - ta)
                tb = tb_ + (H - zb) / slope
                bottom[p] = {"t": float(tb), "kind": "extrap", "ta": float(ta), "za": float(za), "tb": float(tb_), "zb": float(zb),
                             "why": f"The {fmt(p)} % curve hasn't reached the bottom by {fmt(tb_)} min, so extend the line through its last two points.",
                             "eq": f"t = {fmt(tb_)} + ({fmt(H)} − {fmt(zb)}) ÷ [({fmt(zb)} − {fmt(za)}) ÷ ({fmt(tb_)} − {fmt(ta)})] = {fmt(tb, 1)} min"}
            else:
                bottom[p] = {"t": None, "kind": "none", "why": "Not enough of this curve to extend it to the bottom."}

    # 7. overall removal for each curve
    def depth_of(k, t):
        col = [(rows[j], table[j][k].depth) for j in range(len(rows)) if table[j][k].depth is not None]
        ts = np.array([c[0] for c in col])
        zs = np.array([c[1] for c in col])
        if len(ts) == 0:
            return None, "none"
        if t <= ts[-1]:
            return float(np.interp(t, ts, zs)), "interp"
        if len(ts) >= 2 and zs[-1] > zs[-2]:
            return float(zs[-1] + (zs[-1] - zs[-2]) / (ts[-1] - ts[-2]) * (t - ts[-1])), "extrap"
        return float(zs[-1]), "flat"

    overall = {}
    for k, p in enumerate(levels):
        tb = bottom[p]["t"]
        if tb is None:
            continue
        pts = [(0.0, 100.0, "surface")]
        for k2, q in enumerate(levels):
            if q <= p:
                continue
            z, how = depth_of(k2, tb)
            if z is not None and 0 < z < H:
                pts.append((z, q, how))
        pts.append((H, p, "bottom"))
        pts.sort(key=lambda a: a[0])
        strips = []
        for (z0, r0, _), (z1, r1, _) in zip(pts[:-1], pts[1:]):
            strips.append({"z0": z0, "z1": z1, "r0": r0, "r1": r1, "area": (r0 + r1) / 2 * (z1 - z0)})
        area = sum(s["area"] for s in strips)
        terms = " + ".join(f"({fmt(s['r0'])} + {fmt(s['r1'])}) ÷ 2 × {fmt(s['z1'] - s['z0'])}" for s in strips)
        overall[p] = {"t": tb, "points": pts, "strips": strips, "area": area, "R": area / H,
                      "eq": f"R = (1 ÷ {fmt(H)}) × [{terms}] = {fmt(area / H, 1)} %"}

    design = []
    for p in levels:
        if p in overall:
            tb = overall[p]["t"]
            design.append({"p": p, "t": tb, "R": overall[p]["R"], "v": H / tb * 1440, "th": tb / 60,
                           "extrap": bottom[p]["kind"] == "extrap"})

    # tally of hand work
    flat = [c for row in table for c in row]
    n_readings = depths.size * times.size
    tally = {
        "conversions": int(n_readings),
        "easy": sum(c.kind == "easy" for c in flat),
        "hard": sum(c.kind in ("hard", "surface") for c in flat),
        "start": sum(c.kind == "start" for c in flat),
        "below": sum(c.kind == "below" for c in flat),
        "bottom_interp": sum(b["kind"] == "interp" for b in bottom.values()),
        "bottom_extrap": sum(b["kind"] == "extrap" for b in bottom.values()),
        "depth_reads": sum(len(o["points"]) - 2 for o in overall.values()),
        "trapezoids": sum(len(o["strips"]) for o in overall.values()),
        "overflow": len(design),
    }
    tally["interpolations"] = tally["hard"] + tally["bottom_interp"] + tally["depth_reads"]
    tally["arithmetic"] = (tally["conversions"] * 3 + tally["hard"] * 6 + tally["bottom_interp"] * 6
                           + tally["bottom_extrap"] * 6 + tally["depth_reads"] * 6 + tally["trapezoids"] * 3
                           + len(overall) * 2 + tally["overflow"] * 3)
    return {"depths": depths, "times": times, "rows": rows, "conc": np.asarray(conc, float), "c0": float(c0), "H": H,
            "levels": levels, "R": R, "conv_work": conv_work, "table": table, "bottom": bottom, "overall": overall,
            "design": design, "tally": tally}
