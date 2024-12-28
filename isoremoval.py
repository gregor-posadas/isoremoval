import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.patches import FancyBboxPatch
import base64
from io import BytesIO

# -------------------------------------------------------------------------
# Set Streamlit page config
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Isoremoval Curves (On-the-fly Vertical Integration)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------------
# Title and Intro
# -------------------------------------------------------------------------
st.title("Isoremoval Curves Generator with Independent Overall Removal Calculation")

st.markdown("""
This version of the code demonstrates:
1. Generating isoremoval curves in a typical table (using a `times_reference` array).
2. **On-the-fly** computation of the overall removal at the exact time each curve intersects 
   the bottom, by **interpolating every other curve** at that same time.
   
This way, you get a valid Overall_Removal_% for **all** curves that reach the bottom, 
even if their intersection time isn't one of the discrete points in `times_reference`.
""")

# -------------------------------------------------------------------------
# Sidebar: Provide sample file and example table
# -------------------------------------------------------------------------
st.sidebar.header("Upload Data File")

def generate_sample_excel():
    data = {
        'Depth (m)': [0.5, 1.0, 1.5, 2.0, 2.5],
        # Suppose these are times in minutes
        10: [14, 15, 15.4, 16, 17],
        20: [10, 13, 14.2, 14.6, 15],
        35: [7, 10.6, 12, 12.6, 13],
        50: [6.2, 8.2, 10, 11, 11.4],
        70: [5, 7, 7.8, 9, 10],
        85: [4, 6, 7, 8, 8.8]
    }
    df = pd.DataFrame(data)
    columns = ['Depth (m)', 10, 20, 35, 50, 70, 85]
    df = df[columns]
    towrite = BytesIO()
    df.to_excel(towrite, index=False, engine='openpyxl')
    towrite.seek(0)
    return towrite

def get_table_download_link():
    sample_excel = generate_sample_excel()
    b64 = base64.b64encode(sample_excel.read()).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="sample_isoremoval.xlsx">Download Sample Excel File</a>'

st.sidebar.markdown(get_table_download_link(), unsafe_allow_html=True)

st.sidebar.markdown("""
**Expected Format (example):**

| Depth (m) |  10 |  20 |  35 |  50 |  70 |  85 |
|-----------|-----|-----|-----|-----|-----|-----|
|  0.5      |14   |10   |7    |6.2  |5    |4    |
|  1.0      |15   |13   |10.6 |8.2  |7    |6    |
|  1.5      |15.4 |14.2 |12   |10   |7.8  |7    |
|  2.0      |16   |14.6 |12.6 |11   |9    |8    |
|  2.5      |17   |15   |13   |11.4 |10   |8.8  |

*First column:* Depth in meters  
*Column headers (beyond the first):* times in minutes  
*Cells:* measured concentration (mg/L)
""")

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx", "xls"]
)

# -------------------------------------------------------------------------
# Sidebar: Input Parameters
# -------------------------------------------------------------------------
st.sidebar.header("Input Parameters")

initial_concentration = st.sidebar.number_input(
    "Initial Concentration (mg/L)",
    min_value=0.0,
    value=20.0,
    step=0.1
)

depth_units = st.sidebar.selectbox(
    "Select Units for Depth",
    options=["Meters (m)", "Feet (ft)", "Centimeters (cm)", "Inches (in)"],
    index=0
)

max_depth = st.sidebar.number_input(
    "Maximum Depth",
    min_value=0.0,
    value=0.0,
    step=0.1,
    help="Specify the maximum depth for the plots. If 0, uses the maximum depth from data."
)

# -------------------------------------------------------------------------
# 1. Load and parse data
# -------------------------------------------------------------------------
def load_data(uploaded_file):
    if uploaded_file is None:
        return None

    file_ext = uploaded_file.name.split(".")[-1].lower()
    if file_ext in ["xlsx", "xls"]:
        df = pd.read_excel(uploaded_file)
    elif file_ext == "csv":
        df = pd.read_csv(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload .xlsx, .xls, or .csv")
        return None
    return df

def parse_input_data(df):
    try:
        # Convert columns (except the first) to numeric
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        depths = df.iloc[:, 0].values.astype(float)
        try:
            times_float = df.columns[1:].astype(float)
        except ValueError:
            st.error("Could not parse column headers as float. Ensure they're numeric times.")
            return None, None, None

        times = times_float.values
        concentrations = df.iloc[:, 1:].values.astype(float)
        return depths, times, concentrations

    except Exception as e:
        st.error(f"Error parsing data: {e}")
        return None, None, None

if uploaded_file is None:
    st.info("Please upload a CSV or Excel file to continue.")
    st.stop()

df = load_data(uploaded_file)
if df is None:
    st.stop()

depths, times, concentrations = parse_input_data(df)
if depths is None or times is None or concentrations is None:
    st.stop()

# Validate shapes
if concentrations.shape[0] != len(depths) or concentrations.shape[1] != len(times):
    st.error("The shape of the concentrations matrix does not match the number of depths and times.")
    st.stop()

# Determine max depth
if max_depth > 0:
    plot_max_depth = max_depth
    if plot_max_depth < np.max(depths):
        st.warning(
            f"Specified maximum depth ({plot_max_depth} {depth_units}) is less "
            f"than the max depth in data ({np.max(depths)})."
        )
else:
    plot_max_depth = np.max(depths)

# -------------------------------------------------------------------------
# 2. Show the user's data
# -------------------------------------------------------------------------
st.subheader("Your Uploaded Data (Concentrations)")
st.write("Below is the raw data you uploaded:")
st.dataframe(df.style.format(precision=2))

# -------------------------------------------------------------------------
# 3. Compute Percent Removals
# -------------------------------------------------------------------------
percent_removals = (initial_concentration - concentrations) / initial_concentration * 100
removal_df = pd.DataFrame(percent_removals, index=depths, columns=times)

st.subheader("Percent Removal (Table) vs. Time and Depth")
removal_df_display = removal_df.round(2)
removal_df_display.index.name = f"Depth ({depth_units})"
removal_df_display.columns.name = "Time (min)"
st.dataframe(removal_df_display)

# -------------------------------------------------------------------------
# 4. User picks which % removal curves to plot
# -------------------------------------------------------------------------
st.sidebar.subheader("Select Which % Removal Curves to Plot")
default_curves = "10,20,30,40,50,60,70,80"
user_curves_input = st.sidebar.text_input(
    "Enter comma-separated % values (1 to 99):",
    value=default_curves
)
try:
    user_curves_list = [int(x.strip()) for x in user_curves_input.split(",")]
    user_curves_list = [val for val in user_curves_list if 1 <= val < 100]
except:
    user_curves_list = [10, 20, 30, 40, 50, 60, 70, 80]

if len(user_curves_list) == 0:
    user_curves_list = [10, 20, 30, 40, 50, 60, 70, 80]

percent_removal_reference = sorted(list(set(user_curves_list)))

# -------------------------------------------------------------------------
# 5. Build Isoremoval Curves (times_reference-based)
# -------------------------------------------------------------------------
times_reference = np.arange(0, max(times) + 10, step=5)
interpolated_depths = pd.DataFrame(index=times_reference, columns=percent_removal_reference)

# For each depth, define removal% as function of time
interp_time_to_removal = {}
for d in removal_df.index:
    tvals = removal_df.columns.values.astype(float)
    rvals = removal_df.loc[d].values.astype(float)
    # allow extrapolate in time if you want
    interp_time_to_removal[d] = interp1d(
        tvals, rvals,
        kind='linear',
        fill_value="extrapolate"
    )

# Now invert removal->depth
for perc in percent_removal_reference:
    depth_list = []
    for t_val in times_reference:
        if t_val == 0:
            # define 0% removal => depth=0 for continuity
            depth_list.append(0.0)
            continue

        # Evaluate removal at each depth
        local_r = []
        local_d = []
        for d_val in removal_df.index:
            rr = interp_time_to_removal[d_val](t_val)
            local_r.append(rr)
            local_d.append(d_val)
        local_r = np.array(local_r)
        local_d = np.array(local_d)

        # Filter 0..100
        mask = (local_r >= 0) & (local_r <= 100)
        vr = local_r[mask]
        vd = local_d[mask]
        if len(vr) < 2:
            depth_list.append(np.nan)
            continue

        # Sort
        idx_s = np.argsort(vr)
        vr_s = vr[idx_s]
        vd_s = vd[idx_s]

        # If perc < vr_s.min() or > vr_s.max(), we skip or do partial?
        if perc < vr_s[0] or perc > vr_s[-1]:
            depth_list.append(np.nan)
            continue

        # Otherwise, do a linear interpolation
        ixx = np.searchsorted(vr_s, perc)
        if ixx == 0:
            depth_list.append(vd_s[0])
        elif ixx >= len(vr_s):
            depth_list.append(vd_s[-1])
        else:
            r_lo, r_hi = vr_s[ixx-1], vr_s[ixx]
            d_lo, d_hi = vd_s[ixx-1], vd_s[ixx]
            if (r_hi == r_lo):
                depth_list.append(d_lo)
            else:
                slope = (d_hi - d_lo)/(r_hi - r_lo)
                cand_depth = d_lo + slope*(perc - r_lo)
                depth_list.append(cand_depth)

    depth_array = np.array(depth_list)
    depth_array = np.clip(depth_array, 0, plot_max_depth)
    interpolated_depths[perc] = depth_array

interpolated_depths = interpolated_depths.apply(pd.to_numeric, errors='coerce')

# -------------------------------------------------------------------------
# 6. Plot Isoremoval Curves
# -------------------------------------------------------------------------
st.subheader("Generated Isoremoval Curves (Reference)")

fig, ax = plt.subplots(figsize=(14,10))
cmap = plt.get_cmap('tab20')
colors = cmap(np.linspace(0, 1, len(percent_removal_reference)))

for perc, c in zip(percent_removal_reference, colors):
    d_ser = interpolated_depths[perc].values.astype(float)
    t_ser = interpolated_depths.index.values.astype(float)
    mask = (~np.isnan(d_ser))
    ax.plot(t_ser[mask], d_ser[mask], label=f'{perc}%', color=c, marker='o', markersize=3)

ax.set_xlabel("Time (min)", fontsize=14, weight='bold')
ax.set_ylabel(f"Depth ({depth_units})", fontsize=14, weight='bold')
ax.set_title("Isoremoval Curves (via times_reference)", fontsize=16, weight='bold')
ax.set_ylim(plot_max_depth, 0)
ax.grid(True, linestyle='--', linewidth=0.5)

plt.subplots_adjust(bottom=0.25)
handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(
    handles, labels,
    title='Percent Removal',
    loc='upper center',
    bbox_to_anchor=(0.5, -0.25),
    ncol=6,
    fontsize=8,
    title_fontsize=10,
    frameon=True
)
legend.get_title().set_weight('bold')
legend.get_frame().set_facecolor('#f9f9f9')
legend.get_frame().set_edgecolor('gray')
legend.get_frame().set_linewidth(1.5)
legend.get_frame().set_boxstyle('round,pad=0.3,rounding_size=0.2')
legend.get_frame().set_alpha(0.9)

# optional shadow
legend_box = legend.get_frame()
shadow_box = FancyBboxPatch(
    (legend_box.get_x() - 0.02, legend_box.get_y() - 0.02),
    legend_box.get_width() + 0.04,
    legend_box.get_height() + 0.04,
    boxstyle='round,pad=0.3,rounding_size=0.2',
    linewidth=0,
    color='gray',
    alpha=0.2,
    zorder=0
)
ax.add_patch(shadow_box)

st.pyplot(fig)

# Show the Interpolated Depths Table
st.subheader("Interpolated Depths Table (times_reference-based)")
interp_disp = interpolated_depths.round(3)
interp_disp.index.name = "Time (min)"
st.dataframe(interp_disp)

# -------------------------------------------------------------------------
# 7. On-the-fly Overall Removal
# -------------------------------------------------------------------------
st.header("Suspended Solids Removal vs. Detention Time & Overflow Rate (On-the-Fly)")

st.markdown("""
Below, we find the exact time each curve intersects the bottom by direct interpolation 
(again if needed). Then we **independently** interpolate every other curve at that same time 
to do the vertical piecewise integration (the 'vertical line + midpoint' approach). 
Thus we do **not** rely on the time being in `times_reference`.
""")

# 7.1 Build a time->depth function for each curve in percent_removal_reference
# Because we only have "removal->depth" for discrete times, let's do a second function:
# time->(depth) for a given removal%. We'll do a direct interpolation from the "time->removal" approach again
# or we can build "time->depth" for each curve by evaluating the code we used for 'interpolated_depths' but finer.

# But we want a simpler approach: We'll do a small function that, given time t, returns depth for curve R
# by the original approach: we do "for each depth, what's removal? find the depth that matches R"
# That might be slow, but let's do a caching approach or a direct function.

# Instead, let's do a simpler direct method:
# We already have removal_df (depth->(time->removal)).
# We'll define "time->(removal at depth d)" as we have. But we want "time->(depth that has removal=R)."
# We'll define a function depth_of_curve_at_time(R, t) that does the same logic we used above on-the-fly.

def removal_at_time_and_depth(d_val, t_val, time_to_removal):
    """Return removal% at depth d_val for time t_val, using interpolation function."""
    return time_to_removal[d_val](t_val)

def depth_of_curve_at_time(R, t_val, removal_df, time_to_removal, d_min=0.0, d_max=None):
    """
    On-the-fly: we evaluate removal at each known depth in removal_df.index 
    then do a short interpolation. 
    We skip points outside 0..100 range or no data.
    If R isn't between min..max of removal, we skip.
    """
    # evaluate removal at all depths
    if d_max is None:
        d_max = np.max(removal_df.index)

    local_r = []
    local_d = []
    for d_val in removal_df.index:
        r_ = removal_at_time_and_depth(d_val, t_val, time_to_removal)
        local_r.append(r_)
        local_d.append(d_val)

    local_r = np.array(local_r)
    local_d = np.array(local_d)

    mask = (local_r >= 0) & (local_r <= 100)
    vr = local_r[mask]
    vd = local_d[mask]
    if len(vr) < 2:
        return np.nan

    idx_s = np.argsort(vr)
    vr_s = vr[idx_s]
    vd_s = vd[idx_s]

    # if R < vr_s[0] or R > vr_s[-1], return NaN
    if R < vr_s[0] or R > vr_s[-1]:
        return np.nan

    ixx = np.searchsorted(vr_s, R)
    if ixx == 0:
        return vd_s[0]
    if ixx >= len(vr_s):
        return vd_s[-1]
    r_lo, r_hi = vr_s[ixx-1], vr_s[ixx]
    d_lo, d_hi = vd_s[ixx-1], vd_s[ixx]
    if abs(r_hi - r_lo) < 1e-15:
        return d_lo
    slope = (d_hi - d_lo)/(r_hi - r_lo)
    cand_d = d_lo + slope*(R - r_lo)
    # clip to [0, d_max] if you want
    cand_d = np.clip(cand_d, 0, d_max)
    return cand_d

def find_time_for_bottom(R, removal_df, time_to_removal):
    """
    We'll find the time at which curve R hits the bottom (plot_max_depth).
    We do a direct approach: search in time domain, or do a solver approach?

    But let's do a simple "time->(depth_of_curve_at_time(R, t))" interpolation,
    then solve for depth=plot_max_depth. We'll do a 1D bracket approach or 
    a small bisection. 
    We'll bracket time from 0..some max, say 2x max of 'times'.

    If it can't find it, return None.
    """
    # We'll define a function f(t) = depth_of_curve_at_time(R, t) - plot_max_depth
    # We want f(t)=0
    t_min = 0.0
    t_max = max(times)*5  # or something bigger

    def f(t_):
        d_ = depth_of_curve_at_time(R, t_, removal_df, time_to_removal, d_max=plot_max_depth)
        if np.isnan(d_):
            # if we get nan, let's define +ve or -ve in a guessy way
            # We'll define f(t)= -some big if removal isn't valid
            return -9999
        return d_ - plot_max_depth

    # We'll do a simple bracket search in steps
    steps = 200
    t_array = np.linspace(t_min, t_max, steps)
    f_vals = [f(tt) for tt in t_array]

    # find sign changes
    sign_changes = []
    for i in range(len(f_vals)-1):
        if np.isnan(f_vals[i]) or np.isnan(f_vals[i+1]):
            continue
        if f_vals[i]*f_vals[i+1] < 0:
            sign_changes.append(i)

    if len(sign_changes) == 0:
        return None

    # We'll just pick the first sign change
    i_sc = sign_changes[0]
    t_left, t_right = t_array[i_sc], t_array[i_sc+1]

    # bisection
    for _ in range(30):
        t_mid = 0.5*(t_left + t_right)
        fm = f(t_mid)
        fl = f(t_left)
        if fl*fm < 0:
            t_right = t_mid
        else:
            t_left = t_mid
    return 0.5*(t_left+t_right)

def compute_vertical_removal_fraction_on_the_fly(t_intersect, removal_df, time_to_removal):
    """
    We'll do the "vertical line + midpoint" approach at time = t_intersect.
    1) For each curve R in percent_removal_reference, find depth_of_curve_at_time(R, t_intersect).
    2) That gives us a set of (R, depth).
    3) Sort by depth ascending.
    4) Integrate up the column from 0% to 100% or from min to max. Possibly add 0% or 100% if you want the top boundary.

    We'll do a simpler piecewise approach:
    R_total = R_a + (H1/H)*(R_b - R_a) + ...
    (the same formula you described).
    """
    pairs = []
    for R in percent_removal_reference:
        d_ = depth_of_curve_at_time(R, t_intersect, removal_df, time_to_removal, d_max=plot_max_depth)
        if not np.isnan(d_) and 0 <= d_ <= plot_max_depth:
            pairs.append((R, d_))

    # Optionally, add top as 100% at depth=0
    # and bottom as 0% at depth=plot_max_depth
    # if you want a closed bracket. We'll do that if you'd like.
    # For demonstration, let's do it:
    pairs.append((100.0, 0.0))  # top
    pairs.append((0.0, plot_max_depth))  # bottom

    # Now sort by depth ascending
    # Then implement the formula:
    # R_total = R_a + sum( (H_i / totalH) * (R_b - R_a) ), 
    # where H_i is the band thickness, R_a, R_b are the adjacent curve values.
    pairs_sorted = sorted(pairs, key=lambda x: x[1])
    total_depth = plot_max_depth
    if len(pairs_sorted) < 2:
        return np.nan

    # We'll walk pairs_sorted from shallow to deep
    # e.g. (100%, 0.0) -> (some R1, d1) -> (some R2, d2) -> (0%, plot_max_depth)
    R_total = 0.0
    prev_R, prev_d = pairs_sorted[0]
    for i in range(1, len(pairs_sorted)):
        curr_R, curr_d = pairs_sorted[i]
        band_thick = curr_d - prev_d
        if band_thick < 0:
            # skip or reverse
            continue
        # We do midpoint logic like:
        # partial = (band_thick/total_depth)*(curr_R - prev_R)
        # plus the base offset. 
        # But in the formula you gave:
        # R(T_o) = R_a + (H_1/H)*(R_b - R_a) + ...
        # Let's see how that works in practice:
        # R_total = prev_R + sum( (band_thick/H)*(curr_R - prev_R ) ) ??? 
        # Actually, to replicate your formula, let's do it step by step:

        # The original formula is:
        # R_total = R_a + (H_1/H)*(R_b - R_a) + (H_2/H)*(R_c - R_b) + ...
        # So we can store partial sum:
        R_total += (band_thick / total_depth)*(curr_R - prev_R)
        prev_R, prev_d = curr_R, curr_d

    # Then we add the "starting" removal, which is the first curve's R?
    # In the formula R_a + sum(...) we might do R_total += pairs_sorted[0][0].
    # Let's add that at the end:
    R_total += pairs_sorted[0][0]

    # If that yields something outside 0..100, clamp
    R_total = max(0.0, min(100.0, R_total))
    return R_total

results_list = []
for R in percent_removal_reference:
    t_int = find_time_for_bottom(R, removal_df, interp_time_to_removal)
    if t_int is None or t_int <= 1e-9:
        continue

    # Overflow rate
    v_o = (plot_max_depth / t_int)*1440.0
    # vertical line overall removal
    R_tot = compute_vertical_removal_fraction_on_the_fly(t_int, removal_df, interp_time_to_removal)
    t_h = t_int/60.0

    results_list.append({
        'Isoremoval_Curve_%': R,
        'Time_Intersect_Bottom_min': t_int,
        'Detention_Time_h': t_h,
        'Overflow_Rate_m_d': v_o,
        'Overall_Removal_%': R_tot
    })

if len(results_list) == 0:
    st.warning("""
No isoremoval curves intersect the specified maximum depth (or code couldn't find a valid intersection).
If your data or times_reference is too sparse, or if the removal saturates quickly, 
you might see no intersections. 
Alternatively, the bracket search might fail if data is inconsistent.
""")
else:
    results_df = pd.DataFrame(results_list).sort_values('Detention_Time_h')
    st.subheader("Summary of Intersection Times & Computed Removals (On-the-Fly)")

    st.dataframe(results_df.round(2))

    # Plots
    fig_rt, ax_rt = plt.subplots(figsize=(7,5))
    ax_rt.plot(
        results_df['Detention_Time_h'],
        results_df['Overall_Removal_%'],
        marker='o', linestyle='-',
        color='blue'
    )
    ax_rt.set_xlabel("Detention Time (hours)", fontsize=12)
    ax_rt.set_ylabel("Overall Removal (%)", fontsize=12)
    ax_rt.set_title("Suspended Solids Removal vs. Detention Time", fontsize=14, weight='bold')
    ax_rt.grid(True)
    st.pyplot(fig_rt)

    fig_vo, ax_vo = plt.subplots(figsize=(7,5))
    ax_vo.plot(
        results_df['Overflow_Rate_m_d'],
        results_df['Overall_Removal_%'],
        marker='s', linestyle='--',
        color='red'
    )
    ax_vo.set_xlabel("Overflow Rate (m/d)", fontsize=12)
    ax_vo.set_ylabel("Overall Removal (%)", fontsize=12)
    ax_vo.set_title("Suspended Solids Removal vs. Overflow Rate", fontsize=14, weight='bold')
    ax_vo.grid(True)
    st.pyplot(fig_vo)

    st.markdown("""
    **Notes**:
    1. We used a bracket/bisection method (`find_time_for_bottom`) to locate the time 
       each curve intersects the bottom, without relying on `times_reference`.  
    2. We then computed a **vertical line** "piecewise" integration 
       (`compute_vertical_removal_fraction_on_the_fly`) at that exact time, by re-interpolating 
       all other curves.  
    3. The default piecewise formula used was 
       \\( R_{\\mathrm{total}} = R_{a} + \\sum ( (H_{i}/H) (R_{b}-R_{a}) ) \\).  
       You can tweak it if you prefer a midpoint approach or some alternative.
    """)

st.success("Done! You can adjust inputs or upload different data as needed.")
