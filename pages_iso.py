"""Pages for the isoremoval app: the tool itself and the accessibility notes."""
from io import BytesIO

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import streamlit as st

import iso_calc as ic
import ui_theme as ui

# Links shown to students. Leave a value empty to hide that link.
BASIN_TOOL_URL = "https://github.com/gregor-posadas/Sedimentation-Basin-Generator"
# The deployed basin app (https://....streamlit.app). When set, Step 4 shows a button that opens it with the
# student's overflow rate, detention time and target already filled in.
BASIN_APP_URL = ""
WALKTHROUGH_URL = ""   # paste the walkthrough video link here once it's posted

UNIT_NAMES = {"m": "meters (m)", "ft": "feet (ft)", "cm": "centimeters (cm)", "in": "inches (in)"}
MEASURES = {
    "Turbidity (NTU)": ("turbidity", "NTU"),
    "Suspended solids (mg/L)": ("suspended solids", "mg/L"),
}
CLEANUPS = {
    "none": "Use readings as measured",
    "monotone": "Even out readings that go up over time (recommended for noisy data)",
    "gaussian": "Smooth each port's readings (the original app's method)",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _template_xlsx():
    buf = BytesIO()
    ic.sample_frame().to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


def _template_csv():
    return ic.sample_frame().to_csv(index=False).encode()


def _read_upload(file):
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    return pd.read_csv(file)


def _png(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    return buf.getvalue()


def _fmt(x, nd=2):
    return f"{x:,.{nd}f}"


@st.cache_data(show_spinner=False)
def _compute(depths, times, conc, c0, H, cleanup, sigma, monotone, percents, unit):
    if cleanup == "monotone":
        used = ic.clean_monotone(conc)
    elif cleanup == "gaussian":
        used = ic.clean_gaussian(conc, sigma)
    else:
        used = conc
    rd = ic.removal_table(used, c0, depths, times)
    cv = ic.Curves(rd, H, list(percents), enforce_monotonicity=monotone)
    return used, rd, cv, cv.design_table(unit), cv.depth_table()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def curves_figure(cv, percents, unit, measure_unit):
    cols = ui.curve_colors(len(percents))
    with ui.figure_style() as c:
        fig, ax = plt.subplots(figsize=(10.5, 6.2))
        ends = []
        far = []
        xmax_view = cv.t_last * 1.15
        for p, col, mk in zip(percents, cols, ui.MARKERS * 3):
            pts = cv.draw_points(p)
            if pts is None:
                continue
            t_m, z_m, ext = pts
            mark = [i for i, t in enumerate(t_m) if np.isclose(t, cv.times).any()]
            ax.plot(t_m, z_m, color=col, marker=mk, markevery=mark, markersize=7, markeredgecolor=c["paper"],
                    markeredgewidth=0.8, solid_capstyle="round", zorder=3)
            if ext is not None:
                ax.plot(ext[0], ext[1], color=col, linestyle=(0, (1.2, 2.2)), linewidth=2.2, zorder=2)
                end_t, end_z = ext[0][-1], ext[1][-1]
            else:
                end_t, end_z = t_m[-1], z_m[-1]
            if end_t > 3 * cv.t_last:
                far.append((p, end_t, ext))
            else:
                xmax_view = max(xmax_view, end_t * 1.06)
                ends.append((end_t, end_z, p, col))
        for p, end_t, ext in far:   # label curves that leave the right edge where they leave it
            col = cols[percents.index(p)]
            x_edge = xmax_view * 0.985
            z_edge = float(np.interp(x_edge, ext[0], ext[1]))
            ends.append((x_edge, z_edge, p, col))
        far = [(p, t) for p, t, _ in far]
        # direct labels, nudged apart where they would overlap
        ends.sort()
        last_x = {}
        for t_e, z_e, p, col in ends:
            row = 0
            while row in last_x and t_e - last_x[row] < 0.045 * xmax_view:
                row += 1
            last_x[row] = t_e
            dy = -(0.045 + 0.055 * row) * cv.H if np.isclose(z_e, cv.H) else 0
            ax.annotate(f"{p:g}%", (t_e, z_e), xytext=(t_e, z_e + dy), ha="center",
                        va="bottom" if dy else "center", fontsize=12, fontweight="bold", color=col,
                        bbox=dict(boxstyle="round,pad=0.15", fc=c["paper"], ec="none", alpha=0.85),
                        arrowprops=None, zorder=5)
        ax.axvline(cv.t_last, color=c["ink2"], linewidth=1.2, linestyle="--", zorder=1)
        ax.text(cv.t_last, -0.02 * cv.H, f" last reading, {cv.t_last:g} min", color=c["ink2"], fontsize=11,
                va="bottom", ha="left")
        ax.set_xlim(0, xmax_view)
        ax.set_ylim(cv.H, -0.06 * cv.H)
        ax.set_xlabel("Time since the test started (min)")
        ax.set_ylabel(f"Depth below the surface ({unit})")
        ax.grid(True)
        handles = [Line2D([0], [0], color=c["ink"], lw=2.4, marker="o", markevery=[1], label="From your readings"),
                   Line2D([0], [0], color=c["ink"], lw=2.2, linestyle=(0, (1.2, 2.2)),
                          label="Extended past the last reading")]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=12)
        fig.tight_layout()
    return fig, far


def design_figure(df, x_col, x_label, color_key, marker, target=None, read=None):
    with ui.figure_style() as c:
        fig, ax = plt.subplots(figsize=(5.4, 4.3))
        x, y = df[x_col].values, df["Overall removal (%)"].values
        ext = (df["Extrapolated"] == "Yes").values
        ax.plot(x, y, color=c[color_key], linewidth=2.4, zorder=2)
        ax.plot(x[~ext], y[~ext], linestyle="none", marker=marker, markersize=9, color=c[color_key],
                markeredgecolor=c["paper"], zorder=3, label="From your readings")
        if ext.any():
            ax.plot(x[ext], y[ext], linestyle="none", marker=marker, markersize=9, markerfacecolor=c["paper"],
                    markeredgecolor=c[color_key], markeredgewidth=2, zorder=3, label="Extrapolated")
        for xi, yi, p in zip(x, y, df["Curve (%)"].values):
            ax.annotate(f"{p:g}%", (xi, yi), xytext=(6, -12), textcoords="offset points", fontsize=10,
                        color=c["ink2"])
        if target is not None and read is not None:
            xr = read
            ax.plot([ax.get_xlim()[0], xr], [target, target], color=c["ink"], lw=1.4, ls="--", zorder=4)
            ax.plot([xr, xr], [ax.get_ylim()[0], target], color=c["ink"], lw=1.4, ls="--", zorder=4)
            ax.annotate(f"{target:g}% → {xr:,.3g}", (xr, target), xytext=(10, -26), textcoords="offset points",
                        fontsize=12, fontweight="bold", color=c["ink"],
                        bbox=dict(boxstyle="round,pad=0.25", fc=c["paper"], ec=c["ink"], lw=1))
        ax.set_xlabel(x_label)
        ax.set_ylabel("Overall removal (%)")
        ax.grid(True)
        ax.legend(loc="best", fontsize=11)
        fig.tight_layout()
    return fig


def profile_figure(z, r, unit):
    with ui.figure_style() as c:
        fig, ax = plt.subplots(figsize=(5.0, 4.3))
        ax.fill_betweenx(z, 0, r, color=c["fill"], alpha=0.55, linewidth=0)
        ax.plot(r, z, color=c["blue"], marker="o" if len(z) <= 12 else None, markersize=7,
                markeredgecolor=c["paper"], linewidth=2.4)
        ax.set_ylim(z.max(), 0)
        ax.set_xlim(0, 105)
        ax.set_xlabel("Removal (%)")
        ax.set_ylabel(f"Depth ({unit})")
        area = np.trapezoid(r, z)
        ax.text(r.min() * 0.45, z.max() * 0.5, f"shaded area\n= {area:.1f} %·{unit}", ha="center", va="center",
                fontsize=12, fontweight="bold", color=c["ink"])
        ax.grid(True)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# The tool
# ---------------------------------------------------------------------------
def _sidebar():
    with st.sidebar:
        st.markdown('<a class="skip-link" href="#step-1">Skip to the tool</a>', unsafe_allow_html=True)
        st.markdown("### On this page")
        st.markdown(
            "1. [Load your data](#step-1)\n"
            "2. [Describe your test](#step-2)\n"
            "3. [Isoremoval curves](#step-3)\n"
            "4. [Design values](#step-4)"
        )
        if WALKTHROUGH_URL:
            st.markdown(f"[Watch the walkthrough video]({WALKTHROUGH_URL})")
        if BASIN_APP_URL or BASIN_TOOL_URL:
            st.markdown("### Next")
            st.markdown(f"Take your design values to the [Sedimentation Basin Design tool]"
                        f"({BASIN_APP_URL or BASIN_TOOL_URL}).")
        st.markdown("### Display")
        st.caption("Light or dark follows your device. To change it, open the menu at the top right, then Settings.")
        st.caption("Built for CE 323 Environmental and Fluids Laboratory, Boise State University. "
                   "For learning; not certified for engineering design.")


def tool_page():
    ui.inject_css()
    _sidebar()

    st.title("Isoremoval curves")
    st.markdown(
        "Turn your settling-column readings into isoremoval curves, then into the two plots you need to design "
        "a sedimentation basin: **overall removal vs. overflow rate** and **overall removal vs. detention time**."
    )
    st.markdown(
        "**How it works:** 1 load your readings → 2 describe the test → 3 check the curves → "
        "4 read your design values."
    )

    # ---------------- Step 1 ----------------
    ui.step_header(1, "Load your data", "step-1")
    left, right = st.columns([3, 2], gap="large")
    with left:
        upload = st.file_uploader(
            "Settling-column readings (Excel or CSV)", type=["xlsx", "xls", "csv"],
            help="First column: port depths. Column headings: sampling times in minutes. Cells: your readings.",
        )
    with right:
        st.markdown("**No file yet?**")
        if st.button("Use the lecture's sample data", icon=":material/table_view:", use_container_width=True):
            st.session_state["use_sample"] = True
        st.download_button("Download the Excel template", _template_xlsx(), "isoremoval_template.xlsx",
                           icon=":material/download:", use_container_width=True)
        st.download_button("Download the CSV template", _template_csv(), "isoremoval_template.csv",
                           icon=":material/download:", use_container_width=True)

    with st.expander("How to lay out your file"):
        st.markdown(
            "- **First column:** the depth of each sampling port below the water surface (one row per port).\n"
            "- **Other column headings:** the sampling times, in minutes after the test started "
            "(10, 20, 35 … ; \"10 min\" also works).\n"
            "- **Cells:** what you measured at that port and time, in one unit (NTU or mg/L).\n"
            "- Don't include the starting reading (C₀) as a column; you'll enter it in Step 2."
        )
        st.dataframe(ic.sample_frame(), hide_index=True)

    if upload is not None:
        st.session_state["use_sample"] = False
        try:
            raw = _read_upload(upload)
        except Exception:
            st.error("This file couldn't be opened. Save it as .xlsx or .csv and upload it again.",
                     icon=":material/error:")
            st.stop()
        source = upload.name
    elif st.session_state.get("use_sample"):
        raw = ic.sample_frame()
        source = "sample"
    else:
        st.info("Upload your readings, or use the sample data to see how the tool works.", icon=":material/info:")
        st.stop()

    try:
        depths, times, conc = ic.parse_frame(raw)
    except ic.DataProblem as e:
        st.error(f"{e}", icon=":material/error:")
        st.caption("The template above shows the layout the tool expects.")
        st.stop()

    if source == "sample":
        st.success("Using the sample data from the lecture (C₀ = 20 mg/L, five ports from 0.5 to 2.5 m).",
                   icon=":material/check_circle:")
        if st.button("Use my own file instead", icon=":material/upload_file:"):
            st.session_state["use_sample"] = False
            st.rerun()

    # ---------------- Step 2 ----------------
    ui.step_header(2, "Describe your test", "step-2")
    key = f"{source}-{len(depths)}x{len(times)}"
    a, b = st.columns(2, gap="large")
    c_, d_ = st.columns(2, gap="large")
    measure = a.selectbox("What you measured", list(MEASURES),
                          index=1 if source == "sample" else 0, key=f"measure-{key}")
    mname, munit = MEASURES[measure]
    c0 = b.number_input(f"Starting value, C₀ ({munit})", min_value=0.01,
                        value=float(ic.SAMPLE_INITIAL), step=0.1, format="%.2f", key=f"c0-{key}",
                        help="The reading of the well-mixed water before settling started. "
                             "Every result depends on it, so double-check it.")
    unit = c_.selectbox("Depth units", list(UNIT_NAMES), format_func=lambda u: UNIT_NAMES[u],
                        key=f"unit-{key}")
    H = d_.number_input(f"Column depth used, H ({unit})", min_value=float(depths.min()) * 1.0001,
                        value=float(depths.max()), step=0.1, key=f"H-{key}",
                        help="Usually the deepest sampling port. Detention time and overflow rate are "
                             "measured to this depth.")

    st.dataframe(
        pd.DataFrame(conc, index=pd.Index(depths, name=f"Depth ({unit})"),
                     columns=pd.Index([f"{t:g} min" for t in times], name=f"{mname} ({munit})")),
        use_container_width=True,
    )
    st.caption(f"Your readings: rows are port depths, columns are minutes after the start, cells in {munit}.")

    q = ic.assess_data_stratification(conc, depths, c0)
    verdict = {
        "High": ("Clear difference between the top and bottom of the column.", ":material/check_circle:"),
        "Medium": ("Some difference between the top and bottom of the column; curves will work.",
                   ":material/check_circle:"),
        "Low": ("Little difference between the top and bottom. The test may have needed more time, "
                "so the curves will bunch together.", ":material/warning:"),
    }[q["stratification_quality"]]
    st.subheader("Data check")
    m1, m2, m3 = st.columns(3)
    m1.metric("Spread between ports", q["stratification_quality"],
              help="High if removal at the last reading differs by more than 25 points from the top port "
                   "to the bottom port, Medium above 10, Low otherwise.")
    m2.metric("Removal range at the last reading", f"{q['removal_range']:.1f} points",
              help="Highest minus lowest percent removal across the ports at the final time.")
    m3.metric("Change from top to bottom", f"{q['conc_variation']:.1f} %",
              help=f"(highest − lowest) ÷ highest {mname} reading at the final time.")
    st.markdown(f"{verdict[0]}")
    if q["above_initial"]:
        st.warning(f"{q['above_initial']} reading(s) are higher than C₀ ({c0:g} {munit}), which gives negative "
                   "removal. Check C₀ and those cells; the tool treats negative removal as 0 %.",
                   icon=":material/warning:")
    if q["rises"]:
        cells = ", ".join(f"{depths[i]:g} {unit} at {times[j]:g} min" for i, j in q["rises"][:4])
        more = " …" if len(q["rises"]) > 4 else ""
        st.warning(f"{len(q['rises'])} reading(s) go *up* over time ({cells}{more}). Settling should only lower "
                   "a reading, so these are probably sampling noise.", icon=":material/warning:")

    default_clean = "monotone" if q["rises"] else "none"
    with st.expander("Clean-up and options", expanded=bool(q["rises"])):
        cleanup = st.radio("Readings that go up over time", list(CLEANUPS), format_func=CLEANUPS.get,
                           index=list(CLEANUPS).index(default_clean), key=f"clean-{key}")
        sigma = 0.8
        if cleanup == "gaussian":
            sigma = st.slider("Smoothing strength", 0.1, 3.0, 0.8, 0.1, key=f"sigma-{key}",
                              help="Higher is smoother. This method treats readings as evenly spaced in time.")
        st.caption("Evening out changes only the readings that go up, by the least amount needed "
                   "(isotonic regression); readings that already fall over time are left exactly as measured.")
        monotone = st.checkbox("Keep each curve from moving back up the column", value=True, key=f"mono-{key}",
                               help="A removal level can only move deeper as time passes.")

    # ---------------- Step 3 ----------------
    ui.step_header(3, "Isoremoval curves", "step-3")
    st.markdown(
        "Each curve follows one percent removal down the column over time. These are also called "
        "**isoconcentration curves**. Where a curve meets the bottom of the column tells you the "
        "detention time for that percent removal."
    )
    used0 = ic.clean_monotone(conc) if cleanup == "monotone" else (
        ic.clean_gaussian(conc, sigma) if cleanup == "gaussian" else conc)
    rd0 = ic.removal_table(used0, c0, depths, times)
    lo, hi = float(np.clip(rd0.values, 0, 100).min()), float(np.clip(rd0.values, 0, 100).max())
    options = list(range(5, 100, 5))
    default_p = [p for p in range(10, 100, 10) if lo <= p <= hi] or [p for p in options if lo <= p <= hi][:6]
    percents = st.pills("Percent removal curves to draw", options, selection_mode="multi", default=default_p,
                        format_func=lambda p: f"{p}%", key=f"pills-{key}",
                        help=f"Your readings range from {lo:.0f} % to {hi:.0f} % removal. Curves outside that "
                             "range rest on the 100 %-at-the-surface assumption or on extrapolation.")
    percents = sorted(percents or [])
    if not percents:
        st.info("Pick at least one percentage above to draw its curve.", icon=":material/info:")
        st.stop()

    used, rd, cv, design, depth_tab = _compute(depths, times, conc, float(c0), float(H), cleanup, float(sigma),
                                                monotone, tuple(percents), unit)

    with st.expander(f"Percent removal at each port and time"):
        st.markdown(f"Removal = (C₀ − reading) ÷ C₀ × 100. For example, at {depths[0]:g} {unit} after "
                    f"{times[0]:g} min: ({c0:g} − {used[0, 0]:.4g}) ÷ {c0:g} × 100 = **{rd.iloc[0, 0]:.1f} %**.")
        st.dataframe(rd.round(1).rename(columns=lambda t: f"{t:g} min").rename_axis(f"Depth ({unit})"),
                     use_container_width=True)

    reach = design[design["Extrapolated"] == "No"]["Curve (%)"].tolist() if not design.empty else []
    desc = (f"Depth (downward) against time for {', '.join(f'{p}%' for p in percents)} removal. "
            f"Solid lines come from your readings; dotted lines extend a curve in a straight line past your "
            f"last reading at {cv.t_last:g} min. ")
    desc += (f"Within your data, the {', '.join(f'{p}%' for p in reach)} curve(s) reach the bottom at {H:g} {unit}."
             if reach else "No curve reaches the bottom within your data.")
    ui.figure_description(desc)
    fig, far = curves_figure(cv, percents, unit, munit)
    st.pyplot(fig, use_container_width=True)
    if far:
        st.caption("Off the right edge: " + "; ".join(f"{p}% reaches the bottom at {t:,.0f} min" for p, t in far)
                   + ". That is more than three times your last reading, so treat it with caution.")

    with st.expander("The curves as a table"):
        st.markdown(f"Depth ({unit}) of each curve at each time. Blank means the curve is below the "
                    "bottom of the column (or doesn't exist yet).")
        st.dataframe(depth_tab.rename(columns=lambda p: f"{p}%").rename_axis("Time (min)")
                     .map(lambda v: "" if pd.isna(v) else f"{v:.3f}"),
                     use_container_width=True)

    with st.expander("How each point on a curve is found"):
        _interp_example(cv, percents, unit)

    fig_png = _png(fig)
    plt.close(fig)

    # ---------------- Step 4 ----------------
    ui.step_header(4, "Design values", "step-4")
    st.markdown(
        f"For each curve, the time it reaches the bottom (H = {H:g} {unit}) is its **detention time**, "
        f"the **overflow rate** is H ÷ that time (in m/d), and the **overall removal** is the shaded area under the "
        f"removal profile at that time, divided by H."
    )
    if design.empty:
        st.warning("None of the chosen curves reach the bottom of the column, so there are no design values yet. "
                   "Try lower percentages in Step 3.", icon=":material/warning:")
        st.stop()

    show = design.copy()
    show["× last reading"] = (show["Reaches bottom at (min)"] / cv.t_last).round(2)
    show = show.rename(columns={"Curve (%)": "Curve", "Reaches bottom at (min)": "Bottom at (min)",
                                "Detention time (h)": "Detention (h)", "Overflow rate (m/d)": "Overflow (m/d)",
                                "Overall removal (%)": "Overall removal (%)"})
    show["Curve"] = show["Curve"].map(lambda p: f"{p}%")
    st.dataframe(show, hide_index=True, use_container_width=True,
                 column_config={"Extrapolated": st.column_config.TextColumn(
                     help="Yes when the curve reaches the bottom after your last reading, so the value rests on "
                          "extending the curve.")})
    if (design["Extrapolated"] == "Yes").any():
        st.caption("Rows marked Extrapolated reach the bottom after your last reading. Their values come from "
                   "extending each curve in a straight line, so they're less certain the further past the data "
                   "they go (see the × last reading column).")

    st.subheader("Read your design values")
    rmin, rmax = float(design["Overall removal (%)"].min()), float(design["Overall removal (%)"].max())
    tcol, _ = st.columns([1, 2])
    target = tcol.number_input("Target overall removal (%)", min_value=0.0, max_value=100.0,
                             value=float(np.clip(70.0, rmin, rmax)), step=1.0, key=f"target-{key}",
                             help=f"Your curves cover {rmin:.1f} % to {rmax:.1f} % overall removal.")
    read = ic.read_design(design, target)
    if read is None:
        st.warning(f"{target:g} % is outside what your curves cover ({rmin:.1f}–{rmax:.1f} %). "
                   "Pick a target in that range, or add curves in Step 3.", icon=":material/warning:")
        vo = th = None
    else:
        vo, th, ex = read
        st.markdown(f"For **{target:g} % overall removal**, read from your curves: overflow rate ≈ "
                    f"**{vo:,.1f} m/d** and detention time ≈ **{th:.2f} h** ({th * 60:.0f} min)."
                    + (" Part of this comes from extrapolated points." if ex else ""))
        if BASIN_APP_URL:
            st.link_button("Open the Basin Design tool with these values",
                           f"{BASIN_APP_URL}?vo={vo:.2f}&t={th:.3f}&R={target:g}", icon=":material/open_in_new:")
        elif BASIN_TOOL_URL:
            st.markdown(f"Next, take these to the [Sedimentation Basin Design tool]({BASIN_TOOL_URL}).")

    g1, g2 = st.columns(2, gap="large")
    with g1:
        ui.figure_description("Overall removal falls as overflow rate rises: a faster surface loading lets more "
                              "particles leave with the effluent. Filled points come from your readings; "
                              "hollow points are extrapolated.")
        f_vo = design_figure(design, "Overflow rate (m/d)", "Overflow rate (m/d)", "orange", "s",
                             target if vo else None, vo)
        st.pyplot(f_vo, use_container_width=True)
    with g2:
        ui.figure_description("Overall removal rises with detention time: the longer water stays, the more "
                              "particles settle out.")
        f_th = design_figure(design, "Detention time (h)", "Detention time (h)", "blue", "o",
                             target if th else None, th)
        st.pyplot(f_th, use_container_width=True)

    with st.expander("Worked example: overall removal for one curve", expanded=False):
        _removal_example(cv, design, unit)

    # ---------------- Downloads ----------------
    st.subheader("Download your results")
    xbuf = BytesIO()
    with pd.ExcelWriter(xbuf, engine="openpyxl") as xw:
        pd.DataFrame(conc, index=depths, columns=times).rename_axis(f"Depth ({unit}) / Time (min)").to_excel(
            xw, sheet_name="Readings")
        if cleanup != "none":
            pd.DataFrame(used, index=depths, columns=times).rename_axis(f"Depth ({unit}) / Time (min)").to_excel(
                xw, sheet_name="Readings used")
        rd.rename_axis(f"Depth ({unit}) / Time (min)").to_excel(xw, sheet_name="Percent removal")
        depth_tab.rename(columns=lambda p: f"{p}%").rename_axis("Time (min)").to_excel(xw, sheet_name="Curve depths")
        show.to_excel(xw, sheet_name="Design values", index=False)
        pd.DataFrame({"Setting": ["Measured", "C0", "Column depth H", "Depth units", "Clean-up",
                                  "Curves kept from moving up", "Source"],
                      "Value": [measure, f"{c0:g} {munit}", f"{H:g}", unit, CLEANUPS[cleanup], monotone, source]}
                     ).to_excel(xw, sheet_name="Settings", index=False)
    d1, d2 = st.columns(2)
    d3, d4 = st.columns(2)
    d1.download_button("All tables (.xlsx)", xbuf.getvalue(), "isoremoval_results.xlsx",
                       icon=":material/download:", use_container_width=True)
    d2.download_button("Curves plot (.png)", fig_png, "isoremoval_curves.png", icon=":material/download:",
                       use_container_width=True)
    d3.download_button("Overflow-rate plot (.png)", _png(f_vo), "removal_vs_overflow_rate.png",
                       icon=":material/download:", use_container_width=True)
    d4.download_button("Detention-time plot (.png)", _png(f_th), "removal_vs_detention_time.png",
                       icon=":material/download:", use_container_width=True)
    st.caption("Plots download in the colours you're viewing. For a printed report, switch to light mode first.")
    plt.close("all")


def _interp_example(cv, percents, unit):
    p = percents[len(percents) // 2]
    for t in cv.times:
        z, r = cv.profile(t)
        for i in range(1, len(z)):
            if r[i] + 0.5 < p < r[i - 1] - 0.5 and i - 1 > 0:
                z1, r1, z2, r2 = z[i - 1], r[i - 1], z[i], r[i]
                depth = z1 + (z2 - z1) * (r1 - p) / (r1 - r2)
                st.markdown(
                    f"Take the **{p}% curve at {t:g} min**. Removal is {r1:.1f} % at {z1:g} {unit} and {r2:.1f} % "
                    f"at {z2:g} {unit}, so {p} % sits between them. Interpolating in a straight line:\n\n"
                    f"depth = {z1:g} + ({z2:g} − {z1:g}) × ({r1:.1f} − {p}) ÷ ({r1:.1f} − {r2:.1f}) "
                    f"= **{depth:.3f} {unit}**\n\n"
                    "The tool repeats this every quarter minute for every curve, using a straight line in time "
                    "between your readings, and 0 % removal at the start. Above the top port it assumes 100 % "
                    "removal at the water surface, the same assumption the overall-removal area uses."
                )
                return
    st.markdown("Pick a curve between your readings' removal values to see a worked example.")


def _removal_example(cv, design, unit):
    rows = design["Curve (%)"].tolist()
    default = rows.index(70) if 70 in rows else len(rows) // 2
    p = st.selectbox("Curve", rows, index=default, format_func=lambda v: f"{v}% curve")
    row = design[design["Curve (%)"] == p].iloc[0]
    t_b = row["Reaches bottom at (min)"]
    z, r = cv.removal_profile_at(t_b)
    left, right = st.columns([1, 1], gap="large")
    with left:
        ui.figure_description(f"The removal profile when the {p}% curve reaches the bottom at {t_b:g} min: "
                              f"removal against depth, with the area under it shaded.")
        fig = profile_figure(z, r, unit)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        area = float(np.trapezoid(r, z))
        st.markdown(
            f"1. At t = {t_b:g} min, find the removal at each depth (100 % at the surface).\n"
            f"2. Add up trapezoids: each is (average removal of two neighbouring points) × (depth between them).\n"
            f"3. Shaded area = {area:.1f} %·{unit}.\n"
            f"4. Overall removal = area ÷ H = {area:.1f} ÷ {cv.H:g} = **{area / cv.H:.1f} %**."
        )
        if row["Extrapolated"] == "Yes":
            st.caption("This time is past your last reading, so the profile comes from extending the curves, "
                       "as in the lecture's hand method.")
        pts = pd.DataFrame({f"Depth ({unit})": np.round(z, 3), "Removal (%)": np.round(r, 1)})
        if len(pts) > 12:
            pts = pts.iloc[np.unique(np.linspace(0, len(pts) - 1, 12).astype(int))]
            st.caption("Every few points shown; the calculation uses all of them.")
        st.dataframe(pts, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Accessibility page
# ---------------------------------------------------------------------------
def accessibility_page():
    ui.inject_css()
    _sidebar()
    st.title("Accessibility")
    st.markdown("These are the choices behind this tool and where it still falls short. "
                "The target is WCAG 2.1 level AA, the level most universities ask for.")
    st.subheader("Reading")
    st.markdown(
        "- Text is set in **Atkinson Hyperlegible Next**, designed by the Braille Institute for readers with low "
        "vision: letters that are easy to confuse (I, l, 1; O, 0) are drawn to differ.\n"
        "- Reading text is 18 px, lines stop at about 70 characters, and there are only two weights.\n"
        "- New terms (isoremoval curve, detention time, overflow rate) are explained where they first appear, "
        "and every field has a visible label and a short note.")
    st.subheader("Colour and figures")
    st.markdown(
        "- Light mode is near-black on warm paper and dark mode is white on black, both above 4.5:1 contrast. "
        "The theme follows your device; switch it in the menu at the top right, under Settings.\n"
        "- **Colour is never the only cue.** Each curve has its own marker shape and is labelled with its "
        "percentage at its end; extended parts are dotted; extrapolated design points are hollow.\n"
        "- Curve colours come from Fabio Crameri's **Davos** colour map, which is even to the eye and readable "
        "in greyscale. Design-plot colours follow Jack Baker's guidance on choosing colours for figures. "
        "They were checked under simulated colour blindness (Machado, Oliveira and Fernandes, 2009).\n"
        "- Every figure has a sentence above it saying what it shows, and the same numbers are available as a "
        "table and in the Excel download.")
    st.subheader("Keyboard, touch and motion")
    st.markdown(
        "- Everything works with a keyboard; focus shows as a thick outline. Tab past the menu with "
        "\"Skip to the tool\".\n"
        "- Buttons and fields are at least 48 px tall.\n"
        "- If your device asks for reduced motion, the app's animations are turned off.")
    st.subheader("What still falls short")
    st.markdown(
        "- Plots are images. Screen readers get the sentence above each one and the tables, not the picture.\n"
        "- On a phone the curve plot's labels are small; use the expand button on the plot, or the table.\n"
        "- Neighbouring curves are close in colour on purpose (darker means more removal); tell them apart by "
        "their marker shape and the label at the end of each line.\n"
        "- Some widgets (the file uploader, menus) are Streamlit's own, so their behaviour is Streamlit's.\n"
        "- If something here doesn't work for you, tell your instructor or open an issue on the "
        "[GitHub repository](https://github.com/gregor-posadas/isoremoval/issues), and it will be fixed.")
