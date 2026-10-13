"""'By hand' page: the settling-column analysis done the traditional way, step by step and animated."""
import json

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import byhand_anim
import hand_method as hm
import iso_calc as ic
import ui_theme as ui
from pages_iso import T, BASIN_APP_URL
from ui_theme import md

ANIM_HEIGHT = 660
TOOL_PAGE = None      # set by the app entry point (st.Page of the tool)
STEPS = [("bh-1", "The raw readings"), ("bh-2", "Percent removal"), ("bh-3", "Flip the table"),
         ("bh-4", "Interpolate the hard cells"), ("bh-5", "Plot the curves"), ("bh-6", "Detention times"),
         ("bh-7", "Overall removal"), ("bh-8", "Design values"), ("bh-compare", "By hand vs. the tool")]


@st.cache_data(show_spinner=False)
def _analyse(depths, times, conc, c0, levels, unit, munit, version):
    A = hm.analyse(np.array(depths), np.array(times), np.array(conc), c0, list(levels))
    return A, json.dumps(byhand_anim.payload(A, unit, munit))


@st.cache_data(show_spinner=False)
def _scene(scene, data_json, version):
    return byhand_anim.build(scene, data_json)


def _anim(scene, data_json):
    components.html(_scene(scene, data_json, byhand_anim.VERSION), height=ANIM_HEIGHT, scrolling=False)


def _sidebar():
    with st.sidebar:
        st.markdown("### On this page")
        st.markdown("\n".join(f"{i + 1}. [{name}](#{a})" for i, (a, name) in enumerate(STEPS)))
        st.markdown("### Back to the tool")
        if TOOL_PAGE is not None:
            st.page_link(TOOL_PAGE, label="Isoremoval curves", icon=":material/show_chart:")
        st.caption("It does all of this for you.")
        st.markdown("### Display")
        st.caption("Each animation plays when you scroll to it. Pause it, drag the slider to go back, or change the "
                   "speed. With reduced motion turned on, each one shows its finished picture instead.")


def byhand_page():
    ui.setup_page()
    _sidebar()
    st.title("The method by hand")
    md('<p class="lede">Before this tool, CE 323 students turned their settling-column readings into design values '
       'with a calculator, a ruler and graph paper. This page walks through that method one step at a time, with '
       'animations that show where every number comes from and where it goes. It\'s here so you can see what the '
       'tool is doing for you, and check you understand each step.</p>')

    data = st.session_state.get("byhand_data")
    use_sample = data is None
    if data is not None:
        use_sample = st.toggle("Use the lecture's sample data instead of mine", value=False)
    if use_sample:
        d_, t_, c_ = ic.parse_frame(ic.sample_frame())
        data = {"depths": d_.tolist(), "times": t_.tolist(), "conc": c_.tolist(), "c0": float(ic.SAMPLE_INITIAL),
                "unit": "m", "munit": "mg/L", "levels": [20, 30, 40, 50, 60, 70, 80], "source": "sample", "H": float(d_.max())}
        st.info("Using the lecture's sample data. To see your own numbers here, load them in the Isoremoval curves "
                "tool first, then come back.", icon=":material/info:")
    else:
        src = "your file" if data["source"] != "sample" else "the sample data"
        st.success(f"Using the readings you loaded in the Isoremoval curves tool ({src}), C₀ = {data['c0']:g} "
                   f"{data['munit']}, with the curves you picked there.", icon=":material/check_circle:")
        if abs(data.get("H", max(data["depths"])) - max(data["depths"])) > 1e-9:
            st.caption(f"By hand, the column depth is taken as the deepest port ({max(data['depths']):g} {data['unit']}).")
    unit, munit = data["unit"], data["munit"]
    A, js = _analyse(tuple(data["depths"]), tuple(data["times"]), tuple(map(tuple, data["conc"])), float(data["c0"]),
                     tuple(float(p) for p in data["levels"]), unit, munit, byhand_anim.VERSION)
    Tl = A["tally"]

    md("<p><b>What it takes by hand, for this one dataset:</b></p>")
    m = st.columns(4)
    m[0].metric("Conversions", Tl["conversions"], help="Readings turned into percent removal")
    m[1].metric("Interpolations", Tl["interpolations"])
    m[2].metric("Extrapolations", Tl["bottom_extrap"])
    m[3].metric("Arithmetic steps", f"≈ {Tl['arithmetic']}", help="Roughly: every subtraction, division and multiplication")
    st.caption("Counted from your numbers: every subtraction, division and multiplication you'd do on a calculator. "
               "The tool does all of it, at a much finer step, in well under a second.")

    # 1 ------------------------------------------------------------------
    ui.step_header(1, "The raw readings", "bh-1")
    md(f"You start with one reading for every {T('port', 'port')} at every sampling time, plus {T('C₀', 'c0')}, "
       f"the reading of the well-mixed water before anything settled.")
    _anim("raw", js)

    # 2 ------------------------------------------------------------------
    ui.step_header(2, "Turn every reading into percent removal", "bh-2")
    md(f"Each reading becomes a {T('percent removal', 'removal')}: how much of the starting concentration is gone "
       f"from that spot. One formula, applied {Tl['conversions']} times.")
    _anim("convert", js)
    with st.expander(f"All {Tl['conversions']} conversions"):
        st.dataframe(pd.DataFrame(A["conv_work"], index=pd.Index([f"{z:g} {unit}" for z in A["depths"]], name="Depth"),
                                  columns=[f"{t:g} min" for t in A["times"]]), use_container_width=True)

    # 3 ------------------------------------------------------------------
    ui.step_header(3, "Flip the table around", "bh-3")
    md("This is the step that trips people up. The percent table answers <i>\"how much is removed at this depth?\"</i>, "
       f"but an {T('isoremoval curve', 'iso')} needs the opposite: <i>\"how deep is this removal level at this "
       "time?\"</i> So you make a new table with the sampling times down the side and the removal levels you want to "
       "plot across the top. Some cells can be copied straight across; the rest have to be worked out.")
    _anim("flip", js)

    # 4 ------------------------------------------------------------------
    ui.step_header(4, "Interpolate the hard cells", "bh-4")
    md(f"Most cells fall between two ports, so you {T('interpolate', 'interp')}: find the two depths whose removals "
       f"bracket the level, and go down the same fraction of the gap. This table needs {Tl['hard']} of them.")
    _anim("interp", js)
    with st.expander(f"All {Tl['hard']} interpolations"):
        rows = []
        for i, row in enumerate(A["table"]):
            for c in row:
                if c.kind in ("hard", "surface"):
                    rows.append({"Time (min)": f"{c.t:g}", "Level": f"{c.p:g} %", "Working": c.work["eq"],
                                 f"Depth ({unit})": round(c.depth, 3)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Above the top port there are no readings, so the surface counts as 100 % removal.")

    # 5 ------------------------------------------------------------------
    ui.step_header(5, "Plot the curves", "bh-5")
    md("Each column of the finished table is one curve: plot its depths against time and join the points.")
    _anim("plot", js)

    # 6 ------------------------------------------------------------------
    ui.step_header(6, "When does each curve reach the bottom?", "bh-6")
    md(f"That time is the curve's {T('detention time', 'detention')}. If the bottom port crosses the level between "
       f"two samples, interpolate in time; if the curve never gets there during the test, {T('extrapolate', 'extrap')} "
       f"along its last straight piece.")
    _anim("bottom", js)
    with st.expander("The working for every curve"):
        for p in A["levels"]:
            b = A["bottom"][p]
            md(f"- **{p:g} %**: {b['why']}" + (f" {b['eq']}" if "eq" in b else ""))

    # 7 ------------------------------------------------------------------
    ui.step_header(7, "Overall removal for each curve", "bh-7")
    md(f"At a curve's detention time, read how deep each higher curve is, then find the {T('overall removal', 'overall')} "
       f"from the area under that profile with the {T('trapezoid rule', 'trapezoid')}. Every curve needs its own set "
       f"of readings and strips: {Tl['trapezoids']} strips in all here.")
    _anim("overall", js)
    with st.expander("The working for every curve"):
        for p in A["levels"]:
            if p in A["overall"]:
                md(f"- **{p:g} %** at {A['overall'][p]['t']:.1f} min: {A['overall'][p]['eq']}")

    # 8 ------------------------------------------------------------------
    ui.step_header(8, "Design values", "bh-8")
    md(f"Last, turn each curve's time into an {T('overflow rate', 'overflow')} (column depth ÷ time × 1440) and a "
       "detention time in hours, and plot overall removal against both. These are the two design plots.")
    _anim("design", js)

    # compare ------------------------------------------------------------
    st.header("By hand vs. the tool", anchor="bh-compare")
    H = max(data["depths"])
    R = ic.removal_table(np.array(data["conc"], float), float(data["c0"]), np.array(data["depths"], float),
                         np.array(data["times"], float))
    cv = ic.Curves(R, H, [int(p) for p in data["levels"]])
    tool = cv.design_table(unit).set_index("Curve (%)")
    k = ic.DEPTH_TO_METERS[unit]
    rows = []
    for r in A["design"]:
        p = int(r["p"])
        tr = tool.loc[p] if p in tool.index else None
        rows.append({"Curve": f"{p} %",
                     "t, by hand (min)": round(r["t"], 1),
                     "t, tool (min)": None if tr is None else round(float(tr["Reaches bottom at (min)"]), 1),
                     "R, by hand (%)": round(r["R"], 1),
                     "R, tool (%)": None if tr is None else round(float(tr["Overall removal (%)"]), 1),
                     "v₀, by hand (m/d)": round(r["v"] * k, 1),
                     "v₀, tool (m/d)": None if tr is None else round(float(tr["Overflow rate (m/d)"]), 1)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("t: detention time (when the curve reaches the bottom). R: overall removal. v₀: overflow rate.")
    md("<p>The numbers land close, which is the point: the tool follows the same method. The differences come from "
       "how coarse the hand version has to be. By hand you only work at the sampling times and read depths off the "
       "plot, so overall removal is built from a few strips. The tool works every half minute, uses the exact "
       "crossing of each curve with the bottom, and builds the profile from fine steps, so its numbers are both "
       "faster and steadier. Values marked as extrapolated in the tool rest on extending the data, by hand or not, "
       "so treat them with the same care.</p>")
    if TOOL_PAGE is not None:
        st.page_link(TOOL_PAGE, label="Back to the Isoremoval curves tool", icon=":material/arrow_back:")
    md(f'<p>Ready to design? Take your values to the <a href="{BASIN_APP_URL}">Sedimentation Basin Design tool</a>.</p>')
