"""Pages for the isoremoval app: the tool (with its teaching text) and the accessibility notes."""
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import iso_calc as ic
import ui_theme as ui
from explainers import LINKS, PICS
import column_anim
import report_figs
import streamlit.components.v1 as components
from ui_theme import term, md

BASIN_APP_URL = "https://sedimentationbasindesign.streamlit.app/"
WALKTHROUGH_URL = ""   # paste the walkthrough video link here once it's posted

UNIT_NAMES = {"m": "meters (m)", "ft": "feet (ft)", "cm": "centimeters (cm)", "in": "inches (in)"}
MEASURES = {"Turbidity (NTU)": ("turbidity", "NTU"), "Suspended solids (mg/L)": ("suspended solids", "mg/L")}
CLEANUPS = {
    "none": "Use the readings exactly as measured",
    "monotone": "Even out readings that go up over time (recommended for noisy data)",
    "gaussian": "Smooth each port's readings (the original app's method)",
}

# ---------------------------------------------------------------------------
# Explainer terms (definitions written for someone meeting these ideas for the first time)
# ---------------------------------------------------------------------------
D = {
    "column": "A tall, clear tube (in CE 323, 2.5 m) filled with the water you want to treat. It is mixed, then "
              "left perfectly still so the particles can settle, and sampled at several depths over time.",
    "port": "A small tap on the side of the column at a known depth. Opening it draws a sample from that depth.",
    "turbidity": "How cloudy the water is, measured by how much light the particles scatter. Reported in NTU "
                 "(nephelometric turbidity units). More particles, higher NTU.",
    "tss": "Total suspended solids: the mass of particles caught on a filter from a known volume of water, "
           "in mg/L.",
    "c0": "The starting concentration: the reading of the well-mixed water at time zero, before any settling. "
          "Every percent removal is measured against it.",
    "removal": "How much of the starting concentration is gone from that spot: (C₀ − reading) ÷ C₀ × 100. "
               "30 % removal at 0.5 m means 30 % of the particles that started there have sunk below it.",
    "iso": "A line joining every time and depth that share one percent removal. The 50 % curve shows where, at "
           "each moment, exactly half of the particles have settled out. Also called an isoconcentration curve "
           "(\"iso\" means equal).",
    "interp": "Estimating a value between two known ones by drawing a straight line between them. If removal "
              "is 50 % at 0.5 m and 35 % at 1.0 m, then 40 % is about two-thirds of the way down, near 0.83 m.",
    "extrap": "Estimating beyond the data by continuing a line past the last known point. Useful, but less "
              "certain the farther you go.",
    "discrete": "Settling where each particle keeps its size and speed, so it falls in a straight line. Sand "
                "grains behave this way.",
    "floc": "Settling where particles bump into each other and stick, so they grow and speed up as they fall. "
            "Alum flocs behave this way, which is why a column test is needed.",
    "detention": "How long water stays in the basin: basin volume ÷ flow. In the column, it is the time a curve "
                 "takes to reach the bottom.",
    "overflow": "The flow divided by the basin's surface area (Q ÷ A), in m/d. It is also the settling speed a "
                "particle needs to reach the bottom before the water leaves. Also called the surface loading rate.",
    "overall": "The share of all the particles in the whole column that have settled out by a given time: the "
               "average of the removal over the full depth.",
    "trapezoid": "A way to find the area under a line made of straight pieces: split it into strips, and for each "
                 "strip multiply its average height by its width, then add them up.",
    "strat": "How different the top and bottom of the column are. A good test ends with clearly cleaner water "
             "near the top than near the bottom.",
    "h": "The depth you analyse to, usually the deepest port. Detention time and overflow rate both depend on it.",
}


def T(text, key):
    return term(text, D[key], more=LINKS.get(key), pic=PICS.get(key))


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
    return pd.read_excel(file) if file.name.lower().endswith((".xlsx", ".xls")) else pd.read_csv(file)


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


@st.cache_data(show_spinner=False)
def _replay_html(depths, times, used, c0, H, unit, munit, levels, monotone):
    rd = ic.removal_table(used, c0, depths, times)
    cv = ic.Curves(rd, H, list(levels), enforce_monotonicity=monotone)
    return column_anim.build(cv, depths, times, used, c0, unit, munit, levels)


def _figures_zip(cv, percents, unit, design, target, read, depths):
    figs = []
    f, _ = report_figs.curves_fig(cv, percents, unit, curve_points)
    figs.append(("isoremoval_curves", f))
    figs.append(("design_plots", report_figs.design_fig(design, percents, target, (read[0], read[1]) if read else None)))
    rows = design["Curve (%)"].tolist()
    p = st.session_state.get("_worked_p") if st.session_state.get("_worked_p") in rows else (50 if 50 in rows else rows[len(rows) // 2])
    t_b = design[design["Curve (%)"] == p].iloc[0]["Reaches bottom at (min)"]
    z, r = cv.removal_profile_at(t_b)
    z, r = np.asarray(z, float), np.asarray(r, float)
    figs.append((f"overall_removal_{p}pct", report_figs.profile_fig(z, r, unit, cv.H, _bands(z, cv.H, depths), p, t_b)))
    return report_figs.bundle(figs)


def curve_points(cv, p):
    """Points to draw for curve p: measured part (every 5 min and at each reading) and the dotted extension."""
    times = np.unique(np.concatenate([np.arange(0, cv.t_last + 1e-9, 5.0), cv.times]))
    t_b, extrap = cv.time_to_bottom(p)
    ts, zs = [], []
    for t in times:
        if t_b is not None and not extrap and t >= t_b:
            break
        z = cv.depth_at(p, t)
        if not np.isfinite(z) or z >= cv.H:
            break
        ts.append(float(t)); zs.append(float(z))
    ext = None
    if t_b is not None and not extrap:
        ts.append(t_b); zs.append(cv.H)
    elif t_b is not None and ts:
        ext = ([ts[-1], t_b], [zs[-1], cv.H])
    return ts, zs, ext, t_b, extrap


def _sel_points(key):
    try:
        return st.session_state[key]["selection"]["points"]
    except Exception:
        return []


def _new_selection(key):
    """The clicked point if the selection changed since we last acted on it, else None."""
    pts = _sel_points(key)
    sig = repr([(p.get("curve_number"), p.get("point_index")) for p in pts])
    if st.session_state.get(f"_{key}_sig") == sig:
        return None
    st.session_state[f"_{key}_sig"] = sig
    return pts[0] if pts else None


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def curves_figure(cv, percents, unit, follow=None):
    cols = ui.curve_colors(len(percents))
    fig = go.Figure()
    trace_map = []
    xmax = cv.t_last * 1.12
    far = []
    for p, col, sym in zip(percents, cols, ui.SYMBOLS):
        ts, zs, ext, t_b, extrap = curve_points(cv, p)
        if not ts:
            continue
        dim = follow is not None and p != follow
        op = 0.22 if dim else 1.0
        width = 4.5 if p == follow else 2.6
        fig.add_trace(go.Scatter(
            x=ts, y=zs, mode="lines+markers", name=f"{p} %", legendgroup=str(p),
            line=dict(color=col, width=width), opacity=op,
            marker=dict(symbol=sym, size=9 if p == follow else 7, color=col, line=dict(color="#ffffff", width=1)),
            customdata=[p] * len(ts),
            hovertemplate=f"<b>{p} % removal curve</b><br>Time: %{{x:.1f}} min<br>Depth: %{{y:.2f}} {unit}<extra></extra>",
            selected=dict(marker=dict(size=14)), unselected=dict(marker=dict(opacity=op)),
        ))
        trace_map.append(p)
        if ext is not None:
            fig.add_trace(go.Scatter(
                x=ext[0], y=ext[1], mode="lines+markers", legendgroup=str(p), showlegend=False,
                line=dict(color=col, width=width, dash="dot"), opacity=op,
                marker=dict(symbol=[sym, sym + "-open"], size=[0.1, 10], color=col, line=dict(color=col, width=2)),
                customdata=[p, p],
                hovertemplate=(f"<b>{p} % curve, extended past your data</b><br>Reaches the bottom at about "
                               f"%{{x:.0f}} min<extra></extra>"),
                selected=dict(marker=dict(size=14)), unselected=dict(marker=dict(opacity=op)),
            ))
            trace_map.append(p)
        if t_b is not None:
            if t_b > 3 * cv.t_last:
                far.append((p, t_b))
            else:
                xmax = max(xmax, t_b * 1.05)
    if follow is not None:
        ts, zs, ext, t_b, extrap = curve_points(cv, follow)
        if t_b is not None:
            fig.add_trace(go.Scatter(
                x=[0, t_b], y=[0, cv.H], mode="lines", showlegend=False,
                line=dict(color=ui.INK2, width=2, dash="dash"), hoverinfo="skip"))
            trace_map.append(follow)
    fig.add_vline(x=cv.t_last, line=dict(color=ui.INK2, width=1.4, dash="dash"))
    fig.add_annotation(x=cv.t_last, y=1.0, yref="paper", yanchor="bottom", text=f"last reading ({cv.t_last:g} min)",
                       showarrow=False, font=dict(size=13, color=ui.INK2), xanchor="center")
    ui.base_layout(fig, height=440, legend_title="Isoremoval curve (percent removal)", n_items=len(percents), per_row=4,
                   top=36)
    fig.update_xaxes(**ui.style_axes(fig, automargin=False, title_standoff=14, title_text="Time since the test started (min)", range=[0, xmax]))
    fig.update_yaxes(**ui.style_axes(fig, title_text=f"Depth below the water surface ({unit})",
                                     range=[cv.H * 1.02, -0.04 * cv.H]))
    return fig, trace_map, far


def design_figure(design, percents, x_col, title, target=None, xread=None, show_ytitle=True):
    """One design plot (overall removal against x_col). Two of these sit side by side in columns, with
    identical heights, margins and y range, so they line up on desktop and stack on a phone."""
    cols = dict(zip(percents, ui.curve_colors(len(percents))))
    syms = dict(zip(percents, ui.SYMBOLS))
    fig = go.Figure()
    pt_map = []
    if target is not None and xread is not None:
        fig.add_shape(type="line", x0=0, x1=xread, y0=target, y1=target, line=dict(color=ui.INK, width=1.6, dash="dash"))
        fig.add_shape(type="line", x0=xread, x1=xread, y0=0, y1=target, line=dict(color=ui.INK, width=1.6, dash="dash"))
        fig.add_trace(go.Scatter(x=[xread], y=[target], mode="markers", showlegend=False,
                                 marker=dict(symbol="star", size=15, color=ui.GOLD, line=dict(color=ui.INK, width=1.5)),
                                 hovertemplate=f"<b>Target {target:g} %</b><br>{x_col.split(' (')[0]} ≈ {xread:.3g}<extra></extra>"))
        pt_map.append(None)
    d = design.sort_values(x_col)
    fig.add_trace(go.Scatter(x=d[x_col], y=d["Overall removal (%)"], mode="lines", showlegend=False,
                             line=dict(color=ui.AXIS, width=2), hoverinfo="skip"))
    pt_map.append(None)
    for _, r in design.iterrows():
        p = int(r["Curve (%)"])
        ex = r["Extrapolated"] == "Yes"
        fig.add_trace(go.Scatter(
            x=[r[x_col]], y=[r["Overall removal (%)"]], mode="markers", showlegend=False,
            marker=dict(symbol=syms[p] + ("-open" if ex else ""), size=13, color=cols[p],
                        line=dict(color=cols[p] if ex else "#ffffff", width=2.5 if ex else 1)),
            hovertemplate=(f"<b>{p} % curve</b>{' (extrapolated)' if ex else ''}<br>Overflow rate: "
                           f"{r['Overflow rate (m/d)']:.1f} m/d<br>Detention time: {r['Detention time (h)']:.2f} h"
                           f"<br>Overall removal: {r['Overall removal (%)']:.1f} %<extra></extra>"),
            selected=dict(marker=dict(size=18)), unselected=dict(marker=dict(opacity=1))))
        pt_map.append(p)
    ui.base_layout(fig, height=360, legend_title=None, top=52)
    fig.update_layout(showlegend=False, margin=dict(b=ui.XAXIS_BAND + 6),
                      title=dict(text=title, x=0.5, xanchor="center", y=1, yref="container", yanchor="top", pad=dict(t=14),
                                 font=dict(size=16, color=ui.INK, family=ui.FONT)))
    ylo = max(0, design["Overall removal (%)"].min() - 8)
    fig.update_yaxes(**ui.style_axes(fig, range=[ylo, min(100, design["Overall removal (%)"].max() + 6)],
                                     title_text="Overall removal (%)"))
    fig.update_xaxes(**ui.style_axes(fig, automargin=False, title_standoff=14, rangemode="tozero", title_text=x_col))
    return fig, pt_map


def design_legend(design, percents, target):
    """One shared legend for the two design plots, as HTML so it wraps neatly at any width."""
    cols = dict(zip(percents, ui.curve_colors(len(percents))))
    syms = dict(zip(percents, ui.SYMBOLS))
    def svg(sym, col, hollow):
        fill = "#ffffff" if hollow else col
        shapes = {
            "circle": '<circle cx="9" cy="9" r="6"/>', "square": '<rect x="3.5" y="3.5" width="11" height="11"/>',
            "diamond": '<polygon points="9,1.5 16.5,9 9,16.5 1.5,9"/>', "triangle-up": '<polygon points="9,2 16,15 2,15"/>',
            "triangle-down": '<polygon points="2,3 16,3 9,16"/>', "star": '<polygon points="9,1 11.3,6.6 17.3,6.9 12.6,10.7 14.2,16.5 9,13.2 3.8,16.5 5.4,10.7 0.7,6.9 6.7,6.6"/>',
        }
        g = shapes.get(sym, '<path d="M6,2h6v4h4v6h-4v4h-6v-4h-4v-6h4z"/>' if sym == "cross" else
                       '<path d="M3,5l2-2 4,4 4-4 2,2-4,4 4,4-2,2-4-4-4,4-2-2 4-4z"/>' if sym == "x" else '<circle cx="9" cy="9" r="6"/>')
        stroke = col if hollow or sym != "star" else "#1b1a17"
        return (f'<svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true"><g fill="{fill}" stroke="{stroke}" '
                f'stroke-width="{2 if hollow else 1}">{g}</g></svg>')
    items = []
    for _, r in design.iterrows():
        p = int(r["Curve (%)"]); ex = r["Extrapolated"] == "Yes"
        items.append(f'<span class="lg__i">{svg(syms[p], cols[p], ex)}{p} %{" (extrap.)" if ex else ""}</span>')
    if target is not None:
        items.append(f'<span class="lg__i">{svg("star", ui.GOLD, False)}Target {target:g} %</span>')
    md('<div class="lg" role="group" aria-label="Legend for the two design plots"><div class="lg__t">Isoremoval curve '
       '(hollow = extrapolated)</div><div class="lg__r">' + "".join(items) + '</div></div>')


def profile_figure(z, r, unit, H, bands):
    fig = go.Figure()
    fills = ["#c9d8ec", "#a9c1e0"]
    for i, (z0, z1) in enumerate(bands):
        m = (z >= z0 - 1e-9) & (z <= z1 + 1e-9)
        zz = np.concatenate([[z0], z[m], [z1]])
        rr = np.concatenate([[np.interp(z0, z, r)], r[m], [np.interp(z1, z, r)]])
        area = float(np.trapezoid(rr, zz))
        fig.add_trace(go.Scatter(
            x=np.concatenate([rr, [0, 0]]), y=np.concatenate([zz, [z1, z0]]), fill="toself", mode="lines",
            line=dict(width=1.5, color="#ffffff"), fillcolor=fills[i % 2], showlegend=(i == 0),
            name="Strips of the shaded area",
            hovertemplate=f"<b>Strip {z0:g}–{z1:g} {unit}</b><br>Area: {area:.1f} %·{unit}<extra></extra>"))
    fig.add_trace(go.Scatter(x=r, y=z, mode="lines+markers" if len(z) <= 14 else "lines", name="Removal at each depth", cliponaxis=False,
                             line=dict(color=ui.BLUE, width=3), marker=dict(size=9, color=ui.BLUE, line=dict(color="#fff", width=1)),
                             hovertemplate=f"Depth %{{y:.2f}} {unit}<br>Removal %{{x:.1f}} %<extra></extra>"))
    ui.base_layout(fig, height=360, n_items=2, per_row=1)
    fig.update_xaxes(**ui.style_axes(fig, automargin=False, title_standoff=14, title_text="Removal at that moment (%)", range=[0, 104]))
    fig.update_yaxes(**ui.style_axes(fig, title_text=f"Depth ({unit})", range=[H * 1.03, -0.03 * H]))
    return fig


def _bands(z, H, depths):
    edges = np.unique(np.concatenate([[0.0], depths[depths < H], [H]]))
    return list(zip(edges[:-1], edges[1:]))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _sidebar():
    with st.sidebar:
        st.markdown('<a class="skip-link" href="#step-1">Skip to the tool</a>', unsafe_allow_html=True)
        st.markdown("### On this page")
        st.markdown(
            "0. [New here? The idea](#the-idea)\n"
            "1. [Load your readings](#step-1)\n"
            "2. [Describe your test](#step-2)\n"
            "3. [Percent removal](#step-3)\n"
            "4. [Isoremoval curves](#step-4)\n"
            "5. [Curves to design values](#step-5)\n"
            "6. [Read your design values](#step-6)"
        )
        if WALKTHROUGH_URL:
            st.markdown(f"[Watch the walkthrough video]({WALKTHROUGH_URL})")
        st.markdown("### Next")
        st.markdown(f"Take your design values to the [Sedimentation Basin Design tool]({BASIN_APP_URL}).")
        st.markdown("### Display")
        st.caption("Opens in light mode. To switch to dark, open the menu at the top right, then Settings. "
                   "Dotted-underlined words explain themselves: hover, tap, or press Enter on them.")
        st.caption("Built for CE 323 Environmental and Fluids Laboratory, Boise State University. "
                   "For learning; not certified for engineering design.")


# ---------------------------------------------------------------------------
# The tool
# ---------------------------------------------------------------------------
def tool_page():
    ui.setup_page()
    _sidebar()

    st.title("Isoremoval curves")
    md(f'<p class="lede">This tool turns the readings from your {T("settling column", "column")} test into '
       f'{T("isoremoval curves", "iso")}, and then into the two numbers you need to design a sedimentation basin: '
       f'an {T("overflow rate", "overflow")} and a {T("detention time", "detention")}. Each step explains what it is '
       f'doing and why, so you can learn the method as you use it.</p>')

    # ---------------- The idea ----------------
    st.header("New here? The idea in five minutes", anchor="the-idea")
    with st.expander("Read this first if you haven't met isoremoval curves before", expanded=True):
        md(f"""
<p><b>The problem.</b> A sedimentation basin is a big, slow tank. Water flows in at one end and out at the other,
and while it crosses, particles sink to the bottom. To size the tank you need to know how fast your particles
sink. You can't calculate that for real water: alum flocs come in every size, and they
{T("stick together and speed up as they fall", "floc")}. So you measure it.</p>
<p><b>The test.</b> Fill a {T("settling column", "column")} with the water, mix it, and let it stand.
At set times (here 10, 20, 35, 50, 70 and 85 minutes) take a small sample from each {T("port", "port")} and
measure its {T("turbidity", "turbidity")} or {T("suspended solids", "tss")}. As particles sink, readings fall,
first near the top and later deeper down.</p>
<p><b>Percent removal.</b> Compare each reading with the starting value {T("C₀", "c0")}:
removal = (C₀ − reading) ÷ C₀ × 100. If C₀ = 20 mg/L and a port reads 14 mg/L, 30 % of the particles that were at
that depth have sunk past it.</p>
<p><b>Isoremoval curves.</b> Mark on a depth-versus-time chart every place where removal is, say, exactly 50 %,
and join the marks. That line is the 50 % {T("isoremoval curve", "iso")}. It starts at the surface and moves down
as time passes, because more and more particles get out of the way.</p>
<p><b>From curves to a basin.</b> When a curve reaches the bottom of the column, its time is a
{T("detention time", "detention")}, and column depth ÷ that time is an {T("overflow rate", "overflow")}. Averaging
removal over the whole depth at that moment gives the {T("overall removal", "overall")}. Do this for each curve and
you get two design plots: overall removal against overflow rate (which sets the basin's surface area) and against
detention time (which sets its volume).</p>
<p>The steps below do each of these for your data. Words with a dotted underline explain themselves.</p>
""")

    # ---------------- Step 1 ----------------
    ui.step_header(1, "Load your readings", "step-1")
    md("Upload the table of readings from your settling-column test, or start with the sample data from the lecture "
       "to see how everything works.")
    left, right = st.columns([3, 2], gap="large")
    with left:
        upload = st.file_uploader(
            "Settling-column readings (Excel or CSV)", type=["xlsx", "xls", "csv"],
            help="First column: port depths. Column headings: sampling times in minutes. Cells: your readings.")
    with right:
        st.markdown("**No file yet?**")
        if st.button("Use the lecture's sample data", icon=":material/table_view:", use_container_width=True):
            st.session_state["use_sample"] = True
        st.download_button("Download the Excel template", _template_xlsx(), "isoremoval_template.xlsx",
                           icon=":material/download:", use_container_width=True)
        st.download_button("Download the CSV template", _template_csv(), "isoremoval_template.csv",
                           icon=":material/download:", use_container_width=True)
    with st.expander("How to lay out your file"):
        md(f"- **First column:** the depth of each {T('port', 'port')} below the water surface, one row per port.\n"
           "- **The other column headings:** the sampling times, in minutes after the test started "
           "(10, 20, 35 …; \"10 min\" also works).\n"
           "- **Each cell:** what you measured at that port and time, all in one unit (NTU or mg/L).\n"
           f"- Leave out the starting reading {T('C₀', 'c0')}; you'll type it in Step 2.")
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
        raw, source = ic.sample_frame(), "sample"
    else:
        st.info("Upload your readings, or use the sample data, to continue.", icon=":material/info:")
        st.stop()

    try:
        depths, times, conc = ic.parse_frame(raw)
    except ic.DataProblem as e:
        st.error(f"{e}", icon=":material/error:")
        st.caption("Compare your file with the template above.")
        st.stop()

    if source == "sample":
        st.success("Using the lecture's sample data: suspended solids, C₀ = 20 mg/L, five ports from 0.5 to 2.5 m, "
                   "read at six times up to 85 minutes.", icon=":material/check_circle:")
        if st.button("Use my own file instead", icon=":material/upload_file:"):
            st.session_state["use_sample"] = False
            st.rerun()

    # ---------------- Step 2 ----------------
    ui.step_header(2, "Describe your test", "step-2")
    md(f"Tell the tool what the numbers mean. The most important value is {T('C₀', 'c0')}: every result is "
       "measured against it, so a wrong C₀ shifts everything.")
    key = f"{source}-{len(depths)}x{len(times)}"
    a, b = st.columns(2, gap="large")
    c_, d_ = st.columns(2, gap="large")
    measure = a.selectbox("What you measured", list(MEASURES), index=1 if source == "sample" else 0,
                          key=f"measure-{key}")
    mname, munit = MEASURES[measure]
    c0 = b.number_input(f"Starting value, C₀ ({munit})", min_value=0.01, value=float(ic.SAMPLE_INITIAL), step=0.1,
                        format="%.2f", key=f"c0-{key}",
                        help="The reading of the well-mixed water before settling started.")
    unit = c_.selectbox("Depth units", list(UNIT_NAMES), format_func=lambda u: UNIT_NAMES[u], key=f"unit-{key}")
    H = d_.number_input(f"Column depth to analyse, H ({unit})", min_value=float(depths.min()) * 1.0001,
                        value=float(depths.max()), step=0.1, key=f"H-{key}",
                        help="Usually the deepest port. Detention time and overflow rate are measured to this depth.")
    st.dataframe(pd.DataFrame(conc, index=pd.Index(depths, name=f"Depth ({unit})"),
                              columns=pd.Index([f"{t:g} min" for t in times])), use_container_width=True)
    st.caption(f"Your readings in {munit}: one row per port (shallow at the top), one column per sampling time.")

    q = ic.assess_data_stratification(conc, depths, c0)
    st.subheader("Check your readings")
    md(f"Before drawing anything, it's worth checking the test worked. By the last reading, the top of the column "
       f"should be clearly cleaner than the bottom: that difference is called {T('stratification', 'strat')}.")
    m1, m2 = st.columns(2)
    m1.metric("Difference between top and bottom ports", f"{q['removal_range']:.1f} points",
              help="Highest minus lowest percent removal across the ports at the last reading.")
    m2.metric("How clearly the column separated", q["stratification_quality"],
              help="High above 25 points, Medium above 10, Low below that.")
    st.markdown({
        "High": "The column separated clearly, so the curves will be well spread out.",
        "Medium": "The column separated enough for useful curves.",
        "Low": "The top and bottom are still similar, so the test may have needed more time. Curves will bunch "
               "together and design values will rest more on extrapolation.",
    }[q["stratification_quality"]])
    if q["above_initial"]:
        st.warning(f"{q['above_initial']} reading(s) are higher than C₀ ({c0:g} {munit}). That would mean particles "
                   "appeared, which can't happen, so check C₀ and those cells. The tool counts them as 0 % removal.",
                   icon=":material/warning:")
    if q["rises"]:
        cells = ", ".join(f"{depths[i]:g} {unit} at {times[j]:g} min" for i, j in q["rises"][:4])
        st.warning(f"{len(q['rises'])} reading(s) go up over time ({cells}{' …' if len(q['rises']) > 4 else ''}). "
                   "Once particles sink past a port they don't come back, so a reading that rises is almost always "
                   "sampling noise, like a stirred-up sample or an air bubble.", icon=":material/warning:")
    default_clean = "monotone" if q["rises"] else "none"
    with st.expander("Clean-up and options", expanded=bool(q["rises"])):
        cleanup = st.radio("What to do with readings that go up over time", list(CLEANUPS), format_func=CLEANUPS.get,
                           index=list(CLEANUPS).index(default_clean), key=f"clean-{key}")
        sigma = 0.8
        if cleanup == "gaussian":
            sigma = st.slider("Smoothing strength", 0.1, 3.0, 0.8, 0.1, key=f"sigma-{key}",
                              help="Higher is smoother. This method treats readings as evenly spaced in time.")
        st.caption("\"Even out\" lowers only the readings that rise, by the least amount needed to make each port's "
                   "readings fall or stay level over time (isotonic regression). Readings that already behave are "
                   "left exactly as measured.")
        monotone = st.checkbox("Never let a curve move back up the column", value=True, key=f"mono-{key}",
                               help="Once a removal level has reached a depth, it can't rise again.")

    used0 = ic.clean_monotone(conc) if cleanup == "monotone" else (
        ic.clean_gaussian(conc, sigma) if cleanup == "gaussian" else conc)
    rd0 = ic.removal_table(used0, c0, depths, times)

    st.subheader("Replay your test", anchor="replay")
    md(f"This is your {T('settling column', 'column')}, rebuilt from your readings. Press play to watch it settle. Each dot "
       f"is a particle, and each one sinks along its own isoremoval curve, so the dots go exactly where your readings "
       f"say: at every port and sampling time, the share of dots still above the port matches your measured removal. "
       f"Because the particles stick together and grow ({T('flocculent settling', 'floc')}), every dot speeds up as "
       f"it falls and the faster, bigger ones pull ahead. Dots that reach the floor build up the sludge, and the share "
       f"on the floor follows your overall removal. The ports flash at each sampling time and your readings fill in the "
       f"table. The dashed lines mark where removal is 20 %, 40 %, 60 % … right now; the orange dot starts at the "
       f"surface and rides one of them, which is the isoremoval curve you'll draw in Step 4.")
    lo0, hi0 = float(np.clip(rd0.values, 0, 100).min()), float(np.clip(rd0.values, 0, 100).max())
    lv = [p for p in range(20, 100, 20) if lo0 - 10 <= p <= hi0 + 10] or [50]
    components.html(_replay_html(depths, times, used0, float(c0), float(H), unit, munit, tuple(lv), monotone),
                    height=700, scrolling=False)
    st.caption("Use Play, Start over and the time slider (arrow keys work on it). The text under the animation reads out "
               "the removal at each port. Particles are drawn only where your data say they are still suspended.")

    # ---------------- Step 3 ----------------
    ui.step_header(3, "Turn readings into percent removal", "step-3")
    md(f"A reading on its own doesn't say much; what matters is how much of the starting concentration is gone. "
       f"For every cell the tool works out the {T('percent removal', 'removal')}:")
    ex_r = (c0 - used0[0, 0]) / c0 * 100
    ui.idea("The formula", f"<p><b>Removal = (C₀ − reading) ÷ C₀ × 100</b></p><p>For the shallowest port "
            f"({depths[0]:g} {unit}) at the first reading ({times[0]:g} min): ({c0:g} − {used0[0, 0]:.4g}) ÷ {c0:g} "
            f"× 100 = <b>{ex_r:.1f} %</b>.</p>")
    try:
        import cmcrameri.cm as cmc
        cmap = cmc.davos_r
    except Exception:
        cmap = "Blues"
    import matplotlib
    cm = matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap

    def _cell(v):
        rgb = cm(float(np.clip(v, 0, 100)) / 100)[:3]
        lum = sum(w * (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
                  for w, c in zip((0.2126, 0.7152, 0.0722), rgb))
        ink = "#000000" if (lum + 0.05) / 0.05 >= 1.05 / (lum + 0.05) else "#ffffff"
        return f"background-color: {matplotlib.colors.to_hex(rgb)}; color: {ink}"

    sty = (rd0.clip(0, 100).round(1).rename(columns=lambda t: f"{t:g} min").rename_axis(f"Depth ({unit})")
           .style.map(_cell).format("{:.1f} %"))
    st.dataframe(sty, use_container_width=True)
    st.caption("Darker cells = more removed. Read across a row to follow one port through time, or down a column to "
               "compare depths at one moment.")
    ui.idea("What to notice", f"""
<p><b>Removal grows to the right</b> (with time): the longer the column stands, the more particles have had time
to sink past each port.</p>
<p><b>Removal shrinks downward</b> (with depth): to clear a deep port, a particle has to fall farther, so only the
faster ones have made it in the same time. Meanwhile particles from above are still arriving.</p>
<p>That pattern is what makes the curves in the next step slope downward over time.</p>""")

    # ---------------- Step 4 ----------------
    ui.step_header(4, "Draw the isoremoval curves", "step-4")
    md(f"Now pick a percentage, say 50 %, and find every place in the table where removal is exactly 50 %. Usually "
       f"that falls between two ports, so the tool {T('interpolates', 'interp')} to find the depth. Joining those "
       f"points over time draws the 50 % {T('isoremoval curve', 'iso')}.")
    lo, hi = float(np.clip(rd0.values, 0, 100).min()), float(np.clip(rd0.values, 0, 100).max())
    options = list(range(5, 100, 5))
    default_p = [p for p in range(10, 100, 10) if lo <= p <= hi] or [p for p in options if lo <= p <= hi][:6]
    percents = st.pills("Curves to draw", options, selection_mode="multi", default=default_p,
                        format_func=lambda p: f"{p} %", key=f"pills-{key}",
                        help=f"Your table runs from {lo:.0f} % to {hi:.0f} % removal. Curves outside that range rest on "
                             "assumptions or extrapolation.")
    percents = sorted(percents or [])
    if not percents:
        st.info("Pick at least one percentage to draw its curve.", icon=":material/info:")
        st.stop()
    used, rd, cv, design, depth_tab = _compute(depths, times, conc, float(c0), float(H), cleanup, float(sigma),
                                                monotone, tuple(percents), unit)

    # a click on the chart picks the curve to follow
    fkey = f"follow-{key}"
    clicked = _new_selection(f"curves-{key}")
    tmap = st.session_state.get(f"tmap-{key}", [])
    if clicked is not None and clicked.get("curve_number") is not None and clicked["curve_number"] < len(tmap):
        p_clicked = tmap[clicked["curve_number"]]
        if p_clicked in percents:
            st.session_state[fkey] = p_clicked
    if st.session_state.get(fkey) not in percents:
        st.session_state[fkey] = None
    follow = st.segmented_control("Follow one curve (or click any point on the chart)", percents, key=fkey,
                                  format_func=lambda p: f"{p} %",
                                  help="Highlights one curve and explains it below the chart.")

    reach = design[design["Extrapolated"] == "No"]["Curve (%)"].tolist() if not design.empty else []
    desc = (f"Depth against time for the {', '.join(f'{p} %' for p in percents)} curves. Depth increases downward, "
            f"like the column itself. Solid lines come from your readings; dotted lines continue a curve past your last "
            f"reading at {cv.t_last:g} min. " + ("A dashed grey straight line runs from the start to where the followed "
                                          "curve meets the bottom. " if follow is not None else ""))
    desc += (f"Within your data the {', '.join(f'{p} %' for p in reach)} curve(s) reach the bottom ({H:g} {unit})."
             if reach else "No curve reaches the bottom within your data.")
    ui.figure_description(desc)
    fig, tmap, far = curves_figure(cv, percents, unit, follow)
    st.session_state[f"tmap-{key}"] = tmap
    st.plotly_chart(fig, theme=None, key=f"curves-{key}", on_select="rerun", selection_mode="points",
                    config=ui.plot_config("isoremoval_curves"), alt=desc)
    st.caption("Solid line: from your readings. Dotted line: continued past your last reading, ending in a hollow "
               "marker where it reaches the bottom. Hover over a point to read it, click it to follow that curve, drag "
               "to zoom (double-click to reset), and use the camera icon to save the chart.")
    if far:
        st.caption("Beyond the right edge: " + "; ".join(f"the {p} % curve reaches the bottom at about {t:,.0f} min"
                                                         for p, t in far)
                   + ". That is more than three times your last reading, so treat it with real caution.")

    if follow is not None:
        _explain_curve(cv, follow, unit)
    else:
        ui.idea("How to read the chart", f"""
<p>Each line follows one percent removal down the column. Pick a line and read along it: at a given time, the line's
depth is where that much of the water has cleared. Above the line, more than that percentage is gone; below it, less.</p>
<p>Lines for low percentages drop quickly, because the fastest few particles sink fast. Lines for high percentages
creep down slowly, because removing most of the particles means waiting for slow ones too.</p>
<p>Click any point, or pick a curve above, to follow one curve and see what its shape tells you.</p>""")

    with st.expander("The curves as a table"):
        md(f"The depth ({unit}) of each curve every 5 minutes and at each reading. A blank cell means that curve hasn't "
           "started yet or is already below the column's bottom.")
        st.dataframe(depth_tab.rename(columns=lambda p: f"{p} %").rename_axis("Time (min)")
                     .map(lambda v: "" if pd.isna(v) else f"{v:.3f}"), use_container_width=True)
    with st.expander("How one point on a curve is found (worked interpolation)"):
        _interp_example(cv, percents, unit)

    # ---------------- Step 5 ----------------
    ui.step_header(5, "From curves to design values", "step-5")
    md(f"Each curve gives three numbers. Think of the column as a model of the basin: water enters at the top, and "
       f"a particle has to sink the full depth H = {H:g} {unit} before the water leaves.")
    ui.idea("Three numbers per curve", f"""
<p><b>1. {T('Detention time', 'detention')}.</b> The time the curve takes to reach the bottom. Hold water that long
and that percentage is removed even at the very bottom.</p>
<p><b>2. {T('Overflow rate', 'overflow')}</b> = H ÷ detention time. This is the speed of a particle that starts at
the surface and just reaches the bottom in that time. Every particle faster than this is removed completely. The
tool converts it to m/d, the unit used for basin design.</p>
<p><b>3. {T('Overall removal', 'overall')}.</b> Slower particles aren't all lost: the ones that started part-way down
reach the bottom too. So at the moment the curve hits the bottom, removal is 100 % at the surface and falls to the
curve's own percentage at the bottom. The average over the whole depth is the overall removal. It is always higher
than the curve's percentage.</p>""")
    if design.empty:
        st.warning("None of the chosen curves reach the bottom of the column yet, so there are no design values. "
                   "Pick lower percentages in Step 4.", icon=":material/warning:")
        st.stop()
    show = design.copy()
    show["× last reading"] = (show["Reaches bottom at (min)"] / cv.t_last).round(2)
    show = show.rename(columns={"Curve (%)": "Curve", "Reaches bottom at (min)": "At bottom (min)",
                                "Detention time (h)": "Detention (h)", "Overflow rate (m/d)": "Overflow (m/d)",
                                "Overall removal (%)": "Overall removal (%)", "× last reading": "× data"})
    show["Curve"] = show["Curve"].map(lambda p: f"{p} %")
    st.dataframe(show, hide_index=True, use_container_width=True, column_config={
        "Extrapolated": st.column_config.TextColumn(help="Yes when the curve reaches the bottom after your last reading."),
        "At bottom (min)": st.column_config.NumberColumn(help="When the curve reaches the bottom of the column."),
        "Detention (h)": st.column_config.NumberColumn(help="Detention time = the time above, in hours."),
        "Overflow (m/d)": st.column_config.NumberColumn(help="Overflow rate = H ÷ detention time, in metres per day."),
        "× data": st.column_config.NumberColumn(help="How far past your last reading: 2 means twice as long.")})
    if (design["Extrapolated"] == "Yes").any():
        md(f"Rows marked **Extrapolated** reach the bottom after your last reading, so their values come from "
           f"{T('extrapolating', 'extrap')} each curve in a straight line along its last stretch, the same way you would "
           "by hand. The × data column shows how far beyond your last reading each one goes (2 = twice as long).")

    # target readout (before the plot so the crosshair can be drawn)
    st.subheader("The two design plots", anchor="step-6-plots")
    md("Plot each curve's overall removal against its overflow rate, and against its detention time. Each point is one "
       "curve, drawn in the same colour and shape as on the curves chart. The two plots share the vertical axis, so a "
       "point sits at the same height in both.")
    wkey = f"worked-{key}"
    pmaps = st.session_state.get(f"pmap-{key}", {})
    rows_p = design["Curve (%)"].tolist()
    pc = None
    for k_ in ("a", "b"):
        cd = _new_selection(f"design-{key}-{k_}")
        pm = pmaps.get(k_, []) if isinstance(pmaps, dict) else []
        if cd is not None and cd.get("curve_number") is not None and cd["curve_number"] < len(pm):
            pc = pm[cd["curve_number"]]
    if pc is not None:
        if pc in rows_p:
            st.session_state[wkey] = pc
            st.session_state[f"wxn-{key}"] = st.session_state.get(f"wxn-{key}", 0) + 1
    if st.session_state.get(wkey) not in rows_p:
        st.session_state[wkey] = 50 if 50 in rows_p else rows_p[len(rows_p) // 2]

    rmin, rmax = float(design["Overall removal (%)"].min()), float(design["Overall removal (%)"].max())
    tkey = f"target-{key}"
    if tkey not in st.session_state:
        st.session_state[tkey] = float(np.clip(70.0, rmin, rmax))
    target = st.session_state[tkey]
    read = ic.read_design(design, target)
    vo_r, th_r = (read[0], read[1]) if read else (None, None)
    ddesc = ("Two plots side by side with the same overall-removal axis. Left: overall removal falls as the overflow "
             "rate rises. Right: overall removal rises with detention time. Hollow points are extrapolated; the star "
             "marks your target.")
    ui.figure_description(ddesc)
    g1, g2 = st.columns(2, gap="small")
    pmaps = {}
    for col, k_, x_col, ttl, xr, fname in [(g1, "a", "Overflow rate (m/d)", "Overall removal vs. overflow rate", vo_r, "removal_vs_overflow_rate"),
                                            (g2, "b", "Detention time (h)", "Overall removal vs. detention time", th_r, "removal_vs_detention_time")]:
        figd, pm = design_figure(design, percents, x_col, ttl, target if read else None, xr)
        pmaps[k_] = pm
        with col:
            st.plotly_chart(figd, theme=None, key=f"design-{key}-{k_}", on_select="rerun", selection_mode="points",
                            config=ui.plot_config(fname), alt=ddesc)
    st.session_state[f"pmap-{key}"] = pmaps
    design_legend(design, percents, target if read else None)
    ui.idea("Why two plots", f"""
<p><b>Overflow rate sets the basin's surface area</b> (area = flow ÷ overflow rate). Lower overflow rate, bigger
footprint, more removal. That's the left plot: removal falls as overflow rate rises.</p>
<p><b>Detention time sets the basin's volume</b> (volume = flow × time). Longer time, bigger tank, more removal.
That's the right plot.</p>
<p>A real design has to meet both, which is what the basin tool checks.</p>""")

    wxn = st.session_state.get(f"wxn-{key}", 0)   # bumped by a click on the design plots, which opens this box
    with st.expander("Worked example: how overall removal is calculated", expanded=wxn > 0, key=f"wx-{key}-{wxn}"):
        _removal_example(cv, design, unit, wkey, depths)

    # ---------------- Step 6 ----------------
    ui.step_header(6, "Read your design values", "step-6")
    md("Choose the overall removal your basin must achieve. The tool reads across to each plot (the dashed lines and "
       "the star above) and down to the overflow rate and detention time.")
    tcol, _ = st.columns([1, 2])
    tcol.number_input("Target overall removal (%)", min_value=0.0, max_value=100.0, step=1.0,
                      key=tkey, help=f"Your curves cover {rmin:.1f} % to {rmax:.1f} % overall removal.")
    if read is None:
        st.warning(f"{target:g} % is outside what your curves cover ({rmin:.1f}–{rmax:.1f} %). Pick a target in that "
                   "range, or draw more curves in Step 4.", icon=":material/warning:")
    else:
        vo, th, ex = read
        m1, m2, m3 = st.columns(3)
        m1.metric("Target overall removal", f"{target:g} %")
        m2.metric("Overflow rate", f"{vo:,.1f} m/d")
        m3.metric("Detention time", f"{th:.2f} h", help=f"{th * 60:.0f} minutes")
        md(f"To remove {target:g} % of the particles, a basin needs an overflow rate of at most **{vo:,.1f} m/d** and a "
           f"detention time of at least **{th:.2f} h**." + (" Part of this rests on extrapolated points." if ex else ""))
        vs = ",".join(f"{int(r['Curve (%)'])}:{r['Overflow rate (m/d)']:.2f}" for _, r in design.iterrows())
        st.link_button("Design the basin with these values",
                       f"{BASIN_APP_URL}?vo={vo:.2f}&t={th:.3f}&R={target:g}&vs={vs}",
                       icon=":material/open_in_new:", type="primary")
        st.caption("Opens the Sedimentation Basin Design tool in a new tab with these numbers filled in, and with your "
                   "curves' settling speeds so its animated basin uses your own particles.")

    # ---------------- Downloads ----------------
    st.subheader("Save your results")
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
        pd.DataFrame({"Setting": ["Measured", "C0", "Column depth H", "Depth units", "Clean-up", "Curves kept from "
                                  "moving up", "Source"],
                      "Value": [measure, f"{c0:g} {munit}", f"{H:g}", unit, CLEANUPS[cleanup], monotone, source]}
                     ).to_excel(xw, sheet_name="Settings", index=False)
    d1, d2 = st.columns(2)
    d1.download_button("Download all tables (.xlsx)", xbuf.getvalue(), "isoremoval_results.xlsx",
                       icon=":material/download:", use_container_width=True)
    d2.download_button("Download figures (.zip)", _figures_zip(cv, percents, unit, design, target, read, depths),
                       "isoremoval_figures.zip", icon=":material/download:", use_container_width=True)
    st.caption("The figures download has report-ready versions of the curves chart, the two design plots and the "
               "worked overall-removal profile, each as a 300 dpi PNG and a vector SVG, sized for a letter page. "
               "You can also save any chart as shown with the camera icon at its top right.")


def _explain_curve(cv, p, unit):
    ts, zs, ext, t_b, extrap = curve_points(cv, p)
    pts = [(t, z) for t, z in zip(ts, zs) if np.isclose(t, cv.times).any()]
    sample = "; ".join(f"{z:.2f} {unit} at {t:g} min" for t, z in pts[:4])
    body = f"<p><b>Reading the {p} % curve.</b> It leaves the surface at time zero and sinks: {sample}.</p>"
    if t_b is not None and not extrap:
        body += (f"<p>It reaches the bottom ({cv.H:g} {unit}) at <b>{t_b:.1f} min</b>. From that moment, at least "
                 f"{p} % of the particles are gone at every depth in the column.</p>")
    elif t_b is not None:
        body += (f"<p>It hasn't reached the bottom by your last reading, so it is continued in a straight line "
                 f"(dotted) and reaches the bottom at about <b>{t_b:.0f} min</b>, an estimate.</p>")
    body += (f"<p><b>Why it bends.</b> The dashed grey line goes straight from the start to where this curve meets the "
             f"bottom. If particles kept their size, which is called {T('discrete settling', 'discrete')}, the curve would "
             f"follow it. Real curves bend downward, getting steeper with time, because flocs stick together and fall "
             f"faster as they grow, which is called {T('flocculent settling', 'floc')}. That's why the test is needed.</p>")
    ui.idea("This curve", body)


def _interp_example(cv, percents, unit):
    p = percents[len(percents) // 2]
    for t in cv.times:
        z, r = cv.profile(t)
        for i in range(1, len(z)):
            if r[i] + 0.5 < p < r[i - 1] - 0.5 and i - 1 > 0:
                z1, r1, z2, r2 = z[i - 1], r[i - 1], z[i], r[i]
                depth = z1 + (z2 - z1) * (r1 - p) / (r1 - r2)
                md(f"Take the **{p} % curve at {t:g} min**. At that time removal is {r1:.1f} % at {z1:g} {unit} and "
                   f"{r2:.1f} % at {z2:g} {unit}, so {p} % lies between those two ports. Drawing a straight line "
                   f"between them:\n\n"
                   f"depth = {z1:g} + ({z2:g} − {z1:g}) × ({r1:.1f} − {p}) ÷ ({r1:.1f} − {r2:.1f}) = **{depth:.3f} {unit}**\n\n"
                   f"By hand you'd repeat this dozens of times. The tool does it every quarter-minute for every curve. "
                   f"Between readings it assumes removal changes steadily with time, starting from 0 % at time zero, and "
                   f"above the top port it uses 100 % removal at the water surface, where the water has cleared from "
                   f"the very first moment.")
                return
    md("Pick a curve that falls between your ports to see a worked example.")


def _removal_example(cv, design, unit, wkey, depths):
    rows = design["Curve (%)"].tolist()
    p = st.selectbox("Curve (or click a point on the design plots)", rows, key=wkey,
                     format_func=lambda v: f"{v} % curve")
    st.session_state["_worked_p"] = p
    row = design[design["Curve (%)"] == p].iloc[0]
    t_b = row["Reaches bottom at (min)"]
    z, r = cv.removal_profile_at(t_b)
    z, r = np.asarray(z, float), np.asarray(r, float)
    bands = _bands(z, cv.H, depths)
    md(f"Freeze the column at **{t_b:g} min**, the moment the {p} % curve reaches the bottom, and look at removal "
       f"down the depth. It is 100 % at the surface and falls to {p} % at the bottom.")
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        pdesc = (f"Removal against depth at {t_b:g} min, with the area under the line shaded in strips, one strip "
                 "between each pair of ports.")
        ui.figure_description(pdesc)
        st.plotly_chart(profile_figure(z, r, unit, cv.H, bands), theme=None, key=f"profile-{p}",
                        config=ui.plot_config(f"removal_profile_{p}pct"), alt=pdesc)
    with right:
        area_rows = []
        for z0, z1 in bands:
            m = (z >= z0 - 1e-9) & (z <= z1 + 1e-9)
            zz = np.concatenate([[z0], z[m], [z1]])
            rr = np.concatenate([[np.interp(z0, z, r)], r[m], [np.interp(z1, z, r)]])
            area_rows.append((z0, z1, float(np.interp(z0, z, r)), float(np.interp(z1, z, r)), float(np.trapezoid(rr, zz))))
        total = sum(a[-1] for a in area_rows)
        md(f"<p>Each strip is roughly a {T('trapezoid', 'trapezoid')}: its area is the average removal at its top and "
           f"bottom times its height. Hover over a strip to see its area.</p>")
        tab = pd.DataFrame([{"Strip": f"{a:g}–{b:g} {unit}", "Top (%)": round(ra, 1), "Bottom (%)": round(rb, 1),
                             f"Area (%·{unit})": round(ar, 1)} for a, b, ra, rb, ar in area_rows])
        st.dataframe(tab, hide_index=True, use_container_width=True)
        md(f"Total shaded area = **{total:.1f} %·{unit}**. Dividing by the depth gives the average removal over the column:"
           f"\n\n**Overall removal = {total:.1f} ÷ {cv.H:g} = {total / cv.H:.1f} %**")
        if row["Extrapolated"] == "Yes":
            st.caption("This moment is after your last reading, so the profile comes from extending each curve, as "
                       "you would by hand. Strips are not exact trapezoids here; the tool adds up the exact area.")


# ---------------------------------------------------------------------------
# Accessibility page
# ---------------------------------------------------------------------------
def accessibility_page():
    ui.setup_page()
    _sidebar()
    st.title("Accessibility")
    md("These are the choices behind this tool and where it still falls short. The target is WCAG 2.1 level AA, the "
       "level most universities ask for.")
    st.subheader("Reading")
    md("- Text is set in **Open Sans**, a humanist sans-serif with open letter shapes, large x-height and clear "
       "figures, served with the app so it looks the same on every device.\n"
       "- Reading text is 18 px and lines stop at about 72 characters.\n"
       "- Words with a dotted underline explain themselves. Hover to preview; click, tap, Enter or Space to pin; "
       "Escape or a click elsewhere closes the note.\n"
       "- Every field has a visible label and a short note; every step says what it does and why.")
    st.subheader("Colour and figures")
    md("- The tool opens in light mode: near-black on warm paper. Dark mode (white on black) is in the menu at the top "
       "right, under Settings, and your choice is remembered. Both are above 4.5:1 contrast.\n"
       "- Charts sit on white, in both modes, so they print and paste into reports cleanly.\n"
       "- **Colour is never the only cue.** Each curve has its own marker shape, shown in a boxed legend outside the "
       "plotting area; extended parts are dotted; extrapolated design points are hollow. Nothing is labelled on top of "
       "the data. Hover over any point to read its values.\n"
       "- Curve colours come from Fabio Crameri's **Davos** colour map (even to the eye, readable in greyscale); other "
       "colours follow Jack Baker's guidance on figure colours. They were checked under simulated colour blindness "
       "(Machado, Oliveira and Fernandes, 2009).\n"
       "- Every chart has a sentence above it saying what it shows, the same text as its screen-reader description, "
       "and its numbers in a table and in the Excel download.")
    st.subheader("Keyboard, touch and motion")
    md("- Everything except the charts works with a keyboard, with a thick focus outline. \"Skip to the tool\" jumps "
       "past the menu. For the charts, the \"Follow one curve\" control and the worked-example menu do what clicking "
       "a point does.\n"
       "- Buttons and fields are at least 48 px tall.\n"
       "- Sections fade up gently as you scroll. If your device asks for reduced motion, there is no fading or "
       "animation at all.")
    st.subheader("What still falls short")
    md("- The charts (Plotly) can't be explored point by point with a keyboard or screen reader; use the tables "
       "and the Excel download instead.\n"
       "- Some widgets (the file uploader, menus) are Streamlit's own, so their behaviour is Streamlit's.\n"
       "- If something doesn't work for you, tell your instructor or open an issue on "
       "[GitHub](https://github.com/gregor-posadas/isoremoval/issues), and it will be fixed.")
