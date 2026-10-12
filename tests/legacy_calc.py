"""Original (October 2025) isoremoval math, kept unchanged as a reference.

This is the first app's calculation code moved into functions. It was checked
against the old app's output on the sample data (identical results). The tests
use it to show where the corrected method in iso_calc.py differs, and why.
"""
import re

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

DEPTH_TO_METERS = {
    "m": 1.0,
    "ft": 0.3048,
    "cm": 0.01,
    "in": 0.0254,
}

TIME_STEP_MIN = 5  # spacing of the time grid the curves are drawn on


# ---------------------------------------------------------------------------
# Sample data (same numbers as the lab handout and the original app)
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
# 1. Parse
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
    """Return depths (1-D), times (1-D, minutes) and concentrations (depth x time)."""
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
    # Sort rows shallow -> deep and columns early -> late (calculations sort anyway).
    order_d = np.argsort(depths)
    order_t = np.argsort(times)
    return depths[order_d], times[order_t], conc[np.ix_(order_d, order_t)]


# ---------------------------------------------------------------------------
# 2. Data check (unchanged from original)
# ---------------------------------------------------------------------------
def assess_data_stratification(conc_matrix, depths_arr, init_c):
    final_concs = conc_matrix[:, -1]
    conc_range = np.max(final_concs) - np.min(final_concs)
    conc_variation = conc_range / np.max(final_concs) * 100
    final_removals = (init_c - final_concs) / init_c * 100
    removal_range = np.max(final_removals) - np.min(final_removals)
    issues = []
    for i in range(len(depths_arr)):
        for j in range(1, conc_matrix.shape[1]):
            if conc_matrix[i, j] > conc_matrix[i, j - 1]:
                issues.append((depths_arr[i], j))
    return {
        "conc_range": conc_range,
        "conc_variation": conc_variation,
        "removal_range": removal_range,
        "non_monotonic_count": len(issues),
        "non_monotonic": issues,
        "stratification_quality": "High" if removal_range > 25 else "Medium" if removal_range > 10 else "Low",
    }


def default_smoothing(quality):
    sigma = 1.5 if quality["stratification_quality"] == "Low" else 0.8 if quality["stratification_quality"] == "Medium" else 0.5
    return quality["non_monotonic_count"] > 0, sigma


def smooth(conc, sigma):
    out = np.copy(conc)
    for i in range(conc.shape[0]):
        out[i, :] = gaussian_filter1d(conc[i, :], sigma=sigma)
    return out


# ---------------------------------------------------------------------------
# 3. Percent removal and isoremoval curves (unchanged from original)
# ---------------------------------------------------------------------------
def removal_table(conc, init_c, depths, times):
    removals = (init_c - conc) / init_c * 100
    return pd.DataFrame(removals, index=depths, columns=times)


class Curves:
    """Isoremoval curves for one dataset and one design depth H (plot_max)."""

    def __init__(self, removal_df, plot_max, percent_list, enforce_monotonicity=True):
        self.removal_df = removal_df
        self.plot_max = float(plot_max)
        self.percent_list = list(percent_list)
        times = removal_df.columns.values.astype(float)
        self.times_ref = np.arange(0, max(times) + 10, step=TIME_STEP_MIN)
        self.interp_time_removal = {}
        for d_ in removal_df.index:
            t_ = removal_df.columns.values.astype(float)
            r_ = removal_df.loc[d_].values.astype(float)
            self.interp_time_removal[d_] = interp1d(t_, r_, kind="linear", fill_value="extrapolate")

        self.iso_depths = pd.DataFrame(index=self.times_ref, columns=self.percent_list, dtype=float)
        for p_ in self.percent_list:
            depth_list = []
            for t_ in self.times_ref:
                if t_ == 0.0:
                    depth_list.append(0.0)
                    continue
                depth_list.append(self.invert_removal_depth(p_, t_))
            arr_ = np.clip(np.array(depth_list, dtype=float), 0, self.plot_max)
            if enforce_monotonicity:
                for i in range(1, len(arr_)):
                    if not np.isnan(arr_[i]) and not np.isnan(arr_[i - 1]):
                        if arr_[i] < arr_[i - 1]:
                            arr_[i] = arr_[i - 1]
            self.iso_depths[p_] = arr_

    # -- removal profile at one time ------------------------------------
    def removals_at(self, tval):
        all_r = np.array([float(self.interp_time_removal[d_](tval)) for d_ in self.removal_df.index])
        all_d = np.array(list(self.removal_df.index), dtype=float)
        return all_d, all_r

    def bracket(self, perc, tval):
        """The two (depth, removal) points either side of perc at time tval, or None."""
        all_d, all_r = self.removals_at(tval)
        mask_ = (all_r >= 0) & (all_r <= 100)
        r_ok, d_ok = all_r[mask_], all_d[mask_]
        if len(r_ok) < 2:
            return None
        idx_s = np.argsort(r_ok)
        r_s, d_s = r_ok[idx_s], d_ok[idx_s]
        if perc < r_s[0] or perc > r_s[-1]:
            return None
        ixx = np.searchsorted(r_s, perc)
        if ixx == 0 or ixx >= len(r_s):
            return None
        return (d_s[ixx - 1], r_s[ixx - 1]), (d_s[ixx], r_s[ixx])

    def invert_removal_depth(self, perc, tval):
        all_d, all_r = self.removals_at(tval)
        mask_ = (all_r >= 0) & (all_r <= 100)
        r_ok = all_r[mask_]
        d_ok = all_d[mask_]
        if len(r_ok) < 2:
            return np.nan
        idx_s = np.argsort(r_ok)
        r_s = r_ok[idx_s]
        d_s = d_ok[idx_s]
        if perc < r_s[0] or perc > r_s[-1]:
            return np.nan
        ixx = np.searchsorted(r_s, perc)
        if ixx == 0:
            return d_s[0]
        elif ixx >= len(r_s):
            return d_s[-1]
        r_lo, r_hi = r_s[ixx - 1], r_s[ixx]
        dd_lo, dd_hi = d_s[ixx - 1], d_s[ixx]
        if abs(r_hi - r_lo) < 1e-12:
            return dd_lo
        slope = (dd_hi - dd_lo) / (r_hi - r_lo)
        cand = dd_lo + slope * (perc - r_lo)
        return np.clip(cand, 0, self.plot_max)

    # -- drawing helpers ------------------------------------------------
    def curve_points(self, p_):
        """(times, depths) for plotting, extended to the bottom as the original did."""
        d_ser = self.iso_depths[p_].values.astype(float)
        t_ser = self.iso_depths.index.values.astype(float)
        mask__ = ~np.isnan(d_ser)
        t_ok, d_ok = t_ser[mask__], d_ser[mask__]
        if len(t_ok) < 2:
            return None
        sidx = np.argsort(t_ok)
        return extend_final_segment(t_ok[sidx], d_ok[sidx], self.plot_max)

    # -- design results -------------------------------------------------
    def find_time_for_bottom(self, p_):
        srs = self.iso_depths[p_].dropna()
        if srs.empty:
            return None
        plot_max = self.plot_max
        dvals = srs.values
        tvals = srs.index.values.astype(float)
        idx_ = np.argsort(dvals)
        d_s = dvals[idx_]
        t_s = tvals[idx_]
        if plot_max < d_s[0]:
            if len(d_s) < 2:
                return None
            d1, d2 = d_s[0], d_s[1]
            t1, t2 = t_s[0], t_s[1]
            if abs(d2 - d1) < 1e-12:
                return None
            slope = (t2 - t1) / (d2 - d1)
            c_ = t1 + slope * (plot_max - d1)
            return c_ if c_ >= 0 else None
        if plot_max > d_s[-1]:
            if len(d_s) < 2:
                return None
            d1, d2 = d_s[-2], d_s[-1]
            t1, t2 = t_s[-2], t_s[-1]
            if abs(d2 - d1) < 1e-12:
                return None
            slope = (t2 - t1) / (d2 - d1)
            c_ = t2 + slope * (plot_max - d2)
            return c_ if c_ >= 0 else None
        ixx = np.searchsorted(d_s, plot_max)
        if ixx == 0:
            return float(t_s[0])
        if ixx >= len(d_s):
            return float(t_s[-1])
        dl, dh = d_s[ixx - 1], d_s[ixx]
        tl, th = t_s[ixx - 1], t_s[ixx]
        if abs(dh - dl) < 1e-12:
            return float(tl)
        slope = (th - tl) / (dh - dl)
        return tl + slope * (plot_max - dl)

    def overall_removal_profile(self, tval):
        """The (depth, removal) pairs the overall-removal integral uses at time tval."""
        pairs = []
        for d_ in self.removal_df.index:
            r_ = float(self.interp_time_removal[d_](tval))
            if 0 <= r_ <= 100:
                pairs.append((float(d_), r_))
        if len(pairs) < 2:
            return None
        pairs = sorted(pairs, key=lambda x: x[0])
        pairs.insert(0, (0.0, 100.0))
        if pairs[-1][0] < self.plot_max:
            d1, r1 = pairs[-2]
            d2, r2 = pairs[-1]
            if d2 > d1:
                slope = (r2 - r1) / (d2 - d1)
                r_bottom = float(np.clip(r2 + slope * (self.plot_max - d2), 0, 100))
                pairs.append((self.plot_max, r_bottom))
        return pairs

    def compute_overall_removal(self, tval):
        pairs = self.overall_removal_profile(tval)
        if pairs is None:
            return np.nan
        integral = 0.0
        for i in range(1, len(pairs)):
            d1, r1 = pairs[i - 1]
            d2, r2 = pairs[i]
            integral += (r1 + r2) / 2.0 * (d2 - d1)
        return float(np.clip(integral / self.plot_max, 0, 100))

    def design_table(self, depth_unit):
        last_reading = float(max(self.removal_df.columns.values.astype(float)))
        rows = []
        for p_ in self.percent_list:
            t_bot = self.find_time_for_bottom(p_)
            if t_bot is None or t_bot < 1e-12:
                continue
            vo = (self.plot_max * DEPTH_TO_METERS[depth_unit] / t_bot) * 1440.0
            R_tot = self.compute_overall_removal(t_bot)
            if np.isnan(R_tot):
                continue
            rows.append({
                "Curve (%)": p_,
                "Reaches bottom at (min)": round(t_bot, 2),
                "Detention time (h)": round(t_bot / 60.0, 2),
                "Overflow rate (m/d)": round(vo, 2),
                "Overall removal (%)": round(R_tot, 2),
                "Past last reading?": "Yes, extrapolated" if t_bot > last_reading + 1e-9 else "No",
            })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("Detention time (h)").reset_index(drop=True)


def extend_final_segment(tvals, dvals, bottom):
    if len(tvals) < 2:
        return tvals, dvals
    if dvals[-1] >= bottom:
        return tvals, dvals
    d2, d1 = dvals[-1], dvals[-2]
    t2, t1 = tvals[-1], tvals[-2]
    if abs(t2 - t1) < 1e-12:
        return tvals, dvals
    slope = (d2 - d1) / (t2 - t1)
    if abs(slope) < 1e-12:
        return tvals, dvals
    t_ext = t2 + (bottom - d2) / slope
    if t_ext > t2:
        return np.append(tvals, t_ext), np.append(dvals, bottom)
    return tvals, dvals


def read_design(design_df, target):
    """Interpolate overflow rate and detention time for a target overall removal.

    Returns (overflow_m_d, detention_h) or None when the target is outside the
    range the curves cover. Plain linear interpolation between the plotted points.
    """
    if design_df.empty or len(design_df) < 2:
        return None
    df = design_df.sort_values("Overall removal (%)")
    r = df["Overall removal (%)"].values
    if target < r.min() or target > r.max() or np.any(np.diff(r) <= 0):
        return None
    vo = float(np.interp(target, r, df["Overflow rate (m/d)"].values))
    th = float(np.interp(target, r, df["Detention time (h)"].values))
    return vo, th
