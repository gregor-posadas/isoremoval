"""Isoremoval-curve calculations for settling-column data.

Method (same idea as the CE 323 lecture, done exactly instead of by hand):

1. Percent removal at each port and time:  R = (C0 - C) / C0 x 100.
2. The removal profile at any time t is a set of points down the column:
   100 % at the water surface (z = 0) and the port removals below it, joined by
   straight lines. Between readings, each port's removal changes linearly with
   time, starting from 0 % at t = 0 (the column starts well mixed).
3. An isoremoval curve for p % is the depth where that profile equals p, traced
   over time. The time it reaches the column bottom H is solved exactly.
4. Past the last reading there is no data, so a curve is extended in a straight
   line along its last measured segment, as in the lecture. Those results are
   flagged as extrapolated.
5. Overall removal when a curve reaches the bottom at time t:
   R_T = (1/H) x integral of the removal profile from 0 to H (trapezoids).
   Detention time = t;  overflow rate = H / t (converted to m/d).

Changes from the October 2025 version (kept in tests/legacy_calc.py), and why:
- Removal is anchored at 0 % at t = 0. The old code extrapolated backwards from
  the first two readings, so it reported 14-20 % removal at t = 2-5 min.
- The time a curve reaches the bottom is solved exactly. The old code read it
  off a 5-minute grid (for example 46.0 min instead of 44.4 min for the 40 %
  curve in the sample data), which shifted overflow rate and detention time.
- Removals outside 0-100 % are clipped, not dropped. The old code discarded a
  port whose extrapolated removal passed 100 %, so late results could rest on
  two of five ports.
- Curves are extended past the last reading along their own last segment
  (the lecture's method) instead of extrapolating every port's removal in time,
  which could run past 100 %.
- Above the first port, a curve is drawn as a straight line from the origin
  (steady settling) until it reaches that port, instead of jumping down at t = 0.
- Noisy readings can be made non-increasing over time at each port
  ("monotone clean-up", isotonic regression). Gaussian smoothing remains as an
  option, but it treats unevenly spaced readings as evenly spaced.
"""
import re

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

DEPTH_TO_METERS = {"m": 1.0, "ft": 0.3048, "cm": 0.01, "in": 0.0254}
GRID_STEP_MIN = 0.25    # time resolution used for drawing curves
P_STEP = 0.5            # percent spacing of the curves used to build late profiles


# ---------------------------------------------------------------------------
# Sample data (the numbers used in the CE 323 lecture and the original app)
# ---------------------------------------------------------------------------
def sample_frame():
    data = {
        "Depth (m)": [0.5, 1.0, 1.5, 2.0, 2.5],
        10: [14, 15, 15.4, 16, 17],
        20: [10, 13, 14.2, 14.6, 15],
        35: [7, 10.6, 12, 12.6, 13],
        50: [6.2, 8.2, 10, 11, 11.4],
        70: [5, 7, 7.8, 9, 10],
        85: [4, 6, 7, 8, 8.8],
    }
    return pd.DataFrame(data)


SAMPLE_INITIAL = 20.0


# ---------------------------------------------------------------------------
# 1. Read the table
# ---------------------------------------------------------------------------
class DataProblem(Exception):
    """A problem with the uploaded table, phrased so a student can fix it."""


def _header_to_minutes(h):
    if isinstance(h, (int, float, np.integer, np.floating)) and not pd.isna(h):
        return float(h)
    m = re.search(r"-?\d+(\.\d+)?", str(h))
    if not m:
        raise DataProblem(
            f'The column heading "{h}" isn\'t a time. Every heading after the first '
            "column should be a time in minutes, like 10, 20 or 35."
        )
    return float(m.group(0))


def parse_frame(df):
    """Return depths (1-D), times (1-D, minutes) and readings (depth x time)."""
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.shape[1] < 3 or df.shape[0] < 2:
        raise DataProblem(
            "The table needs at least two depths (rows) and two sampling times "
            "(columns) after the depth column."
        )
    times = np.array([_header_to_minutes(c) for c in df.columns[1:]], dtype=float)
    depths = pd.to_numeric(df.iloc[:, 0], errors="coerce").values.astype(float)
    if np.isnan(depths).any():
        raise DataProblem(
            "Some depths in the first column aren't numbers. The first column should "
            "hold only the sampling-port depths, like 0.5, 1.0, 1.5."
        )
    if (depths <= 0).any():
        raise DataProblem("Depths should be measured down from the water surface, so each one is above 0.")
    if (times <= 0).any():
        raise DataProblem("Sampling times should be minutes after the test started, so each one is above 0.")
    conc = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").values.astype(float)
    if np.isnan(conc).any():
        rows, cols = np.where(np.isnan(conc))
        where = ", ".join(f"depth {depths[r]:g} at {times[c]:g} min" for r, c in zip(rows[:3], cols[:3]))
        raise DataProblem(
            f"Some readings are blank or not numbers ({where}). Fill them in, or remove "
            "that row or column, then upload again."
        )
    if len(set(times)) != len(times):
        raise DataProblem("Two columns have the same time. Each sampling time should appear once.")
    if len(set(depths)) != len(depths):
        raise DataProblem("Two rows have the same depth. Each sampling port should appear once.")
    order_d, order_t = np.argsort(depths), np.argsort(times)
    return depths[order_d], times[order_t], conc[np.ix_(order_d, order_t)]


# ---------------------------------------------------------------------------
# 2. Data check
# ---------------------------------------------------------------------------
def assess_data_stratification(conc_matrix, depths_arr, init_c):
    final_concs = conc_matrix[:, -1]
    conc_range = np.max(final_concs) - np.min(final_concs)
    conc_variation = conc_range / np.max(final_concs) * 100
    final_removals = (init_c - final_concs) / init_c * 100
    removal_range = np.max(final_removals) - np.min(final_removals)
    rises = []
    for i in range(len(depths_arr)):
        for j in range(1, conc_matrix.shape[1]):
            if conc_matrix[i, j] > conc_matrix[i, j - 1]:
                rises.append((i, j))
    return {
        "conc_variation": conc_variation,
        "removal_range": removal_range,
        "rises": rises,
        "above_initial": int((conc_matrix > init_c).sum()),
        "stratification_quality": "High" if removal_range > 25 else "Medium" if removal_range > 10 else "Low",
    }


def _non_increasing(y):
    """Closest non-increasing sequence to y (pool-adjacent-violators)."""
    vals, wts = [], []
    for v in -np.asarray(y, dtype=float):
        vals.append(v)
        wts.append(1.0)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            v2 = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            vals[-2:], wts[-2:] = [v2], [w]
    out = []
    for v, w in zip(vals, wts):
        out.extend([v] * int(w))
    return -np.array(out)


def clean_monotone(conc):
    """Make each port's readings non-increasing over time, changing as little as possible."""
    return np.vstack([_non_increasing(row) for row in conc])


def clean_gaussian(conc, sigma):
    """The original app's smoothing (assumes evenly spaced readings)."""
    out = np.copy(conc)
    for i in range(conc.shape[0]):
        out[i, :] = gaussian_filter1d(conc[i, :], sigma=sigma)
    return out


def removal_table(conc, init_c, depths, times):
    return pd.DataFrame((init_c - conc) / init_c * 100, index=depths, columns=times)


# ---------------------------------------------------------------------------
# 3. Curves
# ---------------------------------------------------------------------------
class Curves:
    """Isoremoval curves for one dataset and one column depth H."""

    def __init__(self, removal_df, H, percent_list, enforce_monotonicity=True):
        self.depths = removal_df.index.values.astype(float)
        self.times = removal_df.columns.values.astype(float)
        self.t_last = float(self.times.max())
        self.t_prev = float(np.sort(self.times)[-2])
        self.H = float(H)
        self.monotone = enforce_monotonicity
        self.percent_list = sorted(percent_list)
        R = np.clip(removal_df.values.astype(float), 0, 100)
        self.t_aug = np.concatenate([[0.0], self.times])
        self.R_aug = np.hstack([np.zeros((len(self.depths), 1)), R])
        self.grid = np.unique(np.concatenate([np.arange(0, self.t_last + 1e-9, GRID_STEP_MIN), self.times]))
        self._cache = {}
        # port removals on the drawing grid, computed once
        self.R_grid = np.vstack([np.interp(self.grid, self.t_aug, row) for row in self.R_aug])

    # -- profile at one time (t <= last reading) -------------------------
    def port_removals(self, t):
        return np.array([np.interp(t, self.t_aug, row) for row in self.R_aug])

    def profile(self, t, r=None):
        """Depths and removals from the surface to H at time t (within the data)."""
        if r is None:
            r = self.port_removals(t)
        z = np.concatenate([[0.0], self.depths])
        rr = np.concatenate([[100.0], r])
        if self.H > z[-1]:   # column deeper than the deepest port: extend the last segment
            slope = (rr[-1] - rr[-2]) / (z[-1] - z[-2])
            rr = np.append(rr, np.clip(rr[-1] + slope * (self.H - z[-1]), 0, 100))
            z = np.append(z, self.H)
        elif self.H < z[-1]:  # only the top H of the column is used
            rH = np.interp(self.H, z, rr)
            keep = z < self.H
            z, rr = np.append(z[keep], self.H), np.append(rr[keep], rH)
        return z, rr

    @staticmethod
    def _first_crossing(z, rr, p):
        """Shallowest depth where the profile falls to p; inf if it never does above the bottom."""
        if p >= 100:
            return 0.0
        for i in range(1, len(z)):
            if rr[i] <= p <= rr[i - 1] or rr[i - 1] <= p <= rr[i]:
                if rr[i] == rr[i - 1]:
                    return z[i - 1]
                return z[i - 1] + (z[i] - z[i - 1]) * (rr[i - 1] - p) / (rr[i - 1] - rr[i])
        return np.inf if rr[-1] > p else np.nan

    def _measured_depths(self, p):
        """Depth of curve p at every grid time up to the last reading (inf = past the bottom)."""
        key = round(float(p), 6)
        if key not in self._cache:
            zs = np.array([0.0 if t == 0 else self._first_crossing(*self.profile(t, self.R_grid[:, i]), p)
                           for i, t in enumerate(self.grid)])
            # Above the first port there are no readings. A removal level leaves the surface at a steady
            # speed (discrete settling) until it reaches the first port, so draw that part as a straight
            # line from the origin instead of jumping to the 100 %-to-first-port line at t = 0.
            z1 = self.depths[0]
            r1 = np.maximum.accumulate(self.R_grid[0])
            hit = np.where(r1 >= p)[0]
            if len(hit):
                k = hit[0]
                if k > 0:
                    t0, t1_, a, b = self.grid[k - 1], self.grid[k], r1[k - 1], r1[k]
                    tau = t0 + (t1_ - t0) * (p - a) / (b - a) if b > a else t1_
                else:
                    tau = self.grid[0]
                early = self.grid < tau
                zs[early] = z1 * self.grid[early] / tau
            elif p < 100:
                z_end = zs[-1] if np.isfinite(zs[-1]) else z1
                zs = z_end * self.grid / self.t_last
            if self.monotone:   # a curve never moves back up
                filled = np.where(np.isnan(zs), -np.inf, zs)
                zs = np.where(np.isnan(zs), np.nan, np.maximum.accumulate(filled))
            self._cache[key] = zs
        return self._cache[key]

    def depth_at(self, p, t):
        """Depth of curve p at time t, extended in a straight line past the last reading."""
        zs = self._measured_depths(p)
        if t <= self.t_last:
            return float(np.interp(t, self.grid, np.where(np.isinf(zs), 1e9, zs)))
        z_last = zs[-1]
        z_prev = zs[np.searchsorted(self.grid, self.t_prev)]
        if np.isinf(z_last) or z_last >= self.H:
            return np.inf
        if np.isnan(z_last) or np.isnan(z_prev):
            return np.nan
        slope = (z_last - z_prev) / (self.t_last - self.t_prev)
        return z_last + slope * (t - self.t_last) if slope > 0 else z_last

    # -- when a curve reaches the bottom ---------------------------------
    def bottom_removal_series(self):
        rH = np.array([self.profile(t)[1][-1] for t in self.t_aug])
        return np.maximum.accumulate(rH) if self.monotone else rH

    def time_to_bottom(self, p):
        """Exact time curve p reaches depth H, and whether it needed extrapolation."""
        rH = self.bottom_removal_series()
        hit = np.where(rH >= p)[0]
        if len(hit):
            k = hit[0]
            if k == 0:
                return 0.0, False
            t0, t1, r0, r1 = self.t_aug[k - 1], self.t_aug[k], rH[k - 1], rH[k]
            return float(t0 + (t1 - t0) * (p - r0) / (r1 - r0)), False
        z_last = self.depth_at(p, self.t_last)
        z_prev = self.depth_at(p, self.t_prev)
        if not np.isfinite(z_last) or not np.isfinite(z_prev):
            return None, True
        slope = (z_last - z_prev) / (self.t_last - self.t_prev)
        if slope <= 0:
            return None, True
        return float(self.t_last + (self.H - z_last) / slope), True

    # -- overall removal -------------------------------------------------
    def removal_profile_at(self, t):
        """(depths, removals) used for the overall-removal integral at time t."""
        if t <= self.t_last + 1e-9:
            return self.profile(t)
        ps = np.arange(100.0, 0.0, -P_STEP)
        zs = np.array([self.depth_at(p, t) for p in ps])
        ok = np.isfinite(zs)
        ps, zs = ps[ok], zs[ok]
        order = np.argsort(zs, kind="stable")
        zs, ps = zs[order], ps[order]
        ps = np.minimum.accumulate(ps)          # removal can only fall with depth
        rH = float(np.interp(self.H, zs, ps))
        keep = zs < self.H
        z = np.concatenate([[0.0], zs[keep], [self.H]])
        r = np.concatenate([[100.0], ps[keep], [rH]])
        z, idx = np.unique(z, return_index=True)
        return z, r[idx]

    def overall_removal(self, t):
        z, r = self.removal_profile_at(t)
        return float(np.clip(np.trapezoid(r, z) / self.H, 0, 100))

    # -- outputs ---------------------------------------------------------
    def draw_points(self, p):
        """Measured part (times, depths) and the dotted extension, or None."""
        t_b, extrap = self.time_to_bottom(p)
        zs = self._measured_depths(p)
        ok = np.isfinite(zs) & (zs < self.H)
        if t_b is not None and not extrap:
            ok &= self.grid < t_b
        t_m, z_m = self.grid[ok], zs[ok]
        if len(t_m) == 0:
            return None
        if t_b is not None and not extrap:
            t_m, z_m = np.append(t_m, t_b), np.append(z_m, self.H)
            return t_m, z_m, None
        ext = None
        if t_b is not None:
            ext = (np.array([t_m[-1], t_b]), np.array([z_m[-1], self.H]))
        return t_m, z_m, ext

    def depth_table(self, step=5):
        times = np.unique(np.concatenate([np.arange(0, self.t_last + 1e-9, step), self.times]))
        rows = {}
        for p in self.percent_list:
            vals = []
            for t in times:
                z = self.depth_at(p, t)
                vals.append(np.nan if not np.isfinite(z) or z > self.H + 1e-9 else round(z, 3))
            rows[p] = vals
        return pd.DataFrame(rows, index=times)

    def design_table(self, depth_unit):
        rows = []
        for p in self.percent_list:
            t_b, extrap = self.time_to_bottom(p)
            if t_b is None or t_b <= 0:
                continue
            rows.append({
                "Curve (%)": p,
                "Reaches bottom at (min)": round(t_b, 2),
                "Detention time (h)": round(t_b / 60.0, 3),
                "Overflow rate (m/d)": round(self.H * DEPTH_TO_METERS[depth_unit] / t_b * 1440.0, 2),
                "Overall removal (%)": round(self.overall_removal(t_b), 2),
                "Extrapolated": "Yes" if extrap else "No",
            })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("Detention time (h)").reset_index(drop=True)


def read_design(design_df, target):
    """Overflow rate and detention time for a target overall removal (linear between points)."""
    if design_df.empty or len(design_df) < 2:
        return None
    df = design_df.sort_values("Overall removal (%)")
    r = df["Overall removal (%)"].values
    if target < r.min() or target > r.max() or np.any(np.diff(r) <= 0):
        return None
    lo = np.searchsorted(r, target)
    lo = max(lo - 1, 0) if r[min(lo, len(r) - 1)] != target else lo
    extrap = bool((df["Extrapolated"].iloc[[lo, min(lo + 1, len(r) - 1)]] == "Yes").any())
    return (float(np.interp(target, r, df["Overflow rate (m/d)"].values)),
            float(np.interp(target, r, df["Detention time (h)"].values)), extrap)
