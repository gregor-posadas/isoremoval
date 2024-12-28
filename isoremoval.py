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
    page_title="Isoremoval Curves",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------------
# Title and Intro
# -------------------------------------------------------------------------
st.title("Isoremoval Curves Generator (CSV or Excel)")

st.markdown("""
This application allows you to:

1. Upload your data (depths, times, concentrations) from **CSV or Excel**.
2. Generate **isoremoval curves** for a selected range of removal percentages.
3. Compute **suspended solids removal** vs. **detention time** and vs. **overflow rate**, 
   based on batch-settling column data and a specified maximum depth.
""")

# -------------------------------------------------------------------------
# Sidebar: Provide sample file and example table
# -------------------------------------------------------------------------
st.sidebar.header("Upload Data File")

def generate_sample_excel():
    data = {
        'Depth (m)': [0.5, 1.0, 1.5, 2.0, 2.5],
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

# Example table shown in the sidebar:
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

# File uploader
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
# Helper function: load data
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

# -------------------------------------------------------------------------
# Helper function: parse data (depths, times, concentrations)
# -------------------------------------------------------------------------
def parse_input_data(df):
    try:
        # Convert columns (except the first) to numeric
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        depths = df.iloc[:, 0].values.astype(float)

        try:
            times_float = df.columns[1:].astype(float)
        except ValueError:
            st.error("Could not parse column headers as float. Please ensure they're numeric times.")
            return None, None, None

        times = times_float.values
        concentrations = df.iloc[:, 1:].values.astype(float)

        return depths, times, concentrations
    except Exception as e:
        st.error(f"Error parsing data: {e}")
        return None, None, None

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
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
# Show the user's uploaded data
# -------------------------------------------------------------------------
st.subheader("Your Uploaded Data (Concentrations)")
st.write("Below is the raw data you uploaded:")
st.dataframe(df.style.format(precision=2))

# -------------------------------------------------------------------------
# Compute Percent Removals
# -------------------------------------------------------------------------
percent_removals = (initial_concentration - concentrations) / initial_concentration * 100
removal_df = pd.DataFrame(percent_removals, index=depths, columns=times)

# -------------------------------------------------------------------------
# Show the Removal table
# -------------------------------------------------------------------------
st.subheader("Percent Removal (Table) vs. Time and Depth")

removal_df_display = removal_df.round(2)
removal_df_display.index.name = f"Depth ({depth_units})"
removal_df_display.columns.name = "Time (min)"
st.dataframe(removal_df_display)

# -------------------------------------------------------------------------
# Let user pick which % removal curves to plot
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
# Interpolate Depth vs. Time for each % removal
# -------------------------------------------------------------------------
# Let's define a times_reference
times_reference = np.arange(0, max(times) + 10, step=5)

# Build a DataFrame: rows = times_reference, cols = % removal
interpolated_depths = pd.DataFrame(index=times_reference, columns=percent_removal_reference)

# 1) Build a dictionary: for each depth, removal% as a function of time
interp_funcs_over_time = {}
for d in removal_df.index:
    time_vals = removal_df.columns.values.astype(float)
    removal_vals = removal_df.loc[d].values.astype(float)

    # We'll allow extrapolation in time
    interp_funcs_over_time[d] = interp1d(
        time_vals,
        removal_vals,
        kind='linear',
        fill_value="extrapolate"
    )

# 2) For each time in times_reference, we find removal at each depth.
#    Then we invert (removal -> depth) for the desired % removal.
for perc in percent_removal_reference:
    depth_list = []
    for t_val in times_reference:
        if t_val == 0.0:
            # define removal at time=0 as 0 => depth=0 or something?
            # We'll set depth=0 to indicate top of column
            depth_list.append(0.0)
            continue

        # Evaluate removal at all depths
        local_r = []
        local_d = []
        for d_val in removal_df.index:
            r_val = interp_funcs_over_time[d_val](t_val)
            local_r.append(r_val)
            local_d.append(d_val)

        local_r = np.array(local_r, dtype=float)
        local_d = np.array(local_d, dtype=float)

        # We'll do 0..100 clamp
        valid_mask = (local_r >= 0) & (local_r <= 100)
        vr = local_r[valid_mask]
        vd = local_d[valid_mask]

        if len(vr) < 1:
            depth_list.append(np.nan)
            continue

        # Sort by removal ascending
        sort_idx = np.argsort(vr)
        vr_sorted = vr[sort_idx]
        vd_sorted = vd[sort_idx]

        # If perc < vr_sorted.min(), or perc > vr_sorted.max(), we'll do an extrapolation:
        if perc < vr_sorted.min():
            # Extrapolate below min removal
            # We'll define slope from the first two points
            if len(vr_sorted) >= 2:
                r1, r2 = vr_sorted[0], vr_sorted[1]
                d1, d2 = vd_sorted[0], vd_sorted[1]
                slope = (d2 - d1)/(r2 - r1) if (r2 != r1) else 0
                cand_d = d1 + slope*(perc - r1)
                # We'll clamp or leave as is
                if cand_d < 0 or cand_d > plot_max_depth:
                    depth_list.append(np.nan)
                else:
                    depth_list.append(cand_d)
            else:
                # Only one point => can't interpolate
                depth_list.append(np.nan)

        elif perc > vr_sorted.max():
            # Extrapolate above max removal
            if len(vr_sorted) >= 2:
                r1, r2 = vr_sorted[-2], vr_sorted[-1]
                d1, d2 = vd_sorted[-2], vd_sorted[-1]
                slope = (d2 - d1)/(r2 - r1) if (r2 != r1) else 0
                cand_d = d2 + slope*(perc - r2)
                # clamp or store
                if cand_d < 0 or cand_d > plot_max_depth:
                    # We'll store anyway but if you want to clamp to max_depth, do so
                    depth_list.append(cand_d)
                else:
                    depth_list.append(cand_d)
            else:
                depth_list.append(np.nan)

        else:
            # Normal interpolation inside the domain
            # Find the bounding indices i..i+1 for perc
            idx = np.searchsorted(vr_sorted, perc)
            if idx == 0:
                # should be covered by "perc < vr_sorted.min()", but just in case
                depth_list.append(vd_sorted[0])
            elif idx >= len(vr_sorted):
                # covered by "perc > vr_sorted.max()" but just in case
                depth_list.append(vd_sorted[-1])
            else:
                r_lo, r_hi = vr_sorted[idx-1], vr_sorted[idx]
                d_lo, d_hi = vd_sorted[idx-1], vd_sorted[idx]
                if (r_hi == r_lo):
                    depth_list.append(d_lo)
                else:
                    slope = (d_hi - d_lo)/(r_hi - r_lo)
                    cand_d = d_lo + slope*(perc - r_lo)
                    depth_list.append(cand_d)

    # Now we have an array of depths vs. time for this perc
    # We'll clamp to [0, plot_max_depth] if desired
    depth_array = np.array(depth_list)
    depth_array = np.clip(depth_array, 0, plot_max_depth)
    interpolated_depths[perc] = depth_array

# Clean up
interpolated_depths = interpolated_depths.apply(pd.to_numeric, errors='coerce')

# -------------------------------------------------------------------------
# "Generated Isoremoval Curves" and Main Plot
# -------------------------------------------------------------------------
st.subheader("Generated Isoremoval Curves")

fig, ax = plt.subplots(figsize=(14, 10))
plt.rcParams['text.usetex'] = False

cmap = plt.get_cmap('tab20')
colors = cmap(np.linspace(0, 1, len(percent_removal_reference)))

# We will store separate arrays for the "solid" portion and "dashed" portion
# so we can plot them with the same color, different linestyles
for perc, color in zip(percent_removal_reference, colors):
    depth_series = interpolated_depths[perc].values.astype(float)
    time_series = interpolated_depths.index.values.astype(float)

    # Build a mask for valid data
    valid_mask = ~np.isnan(depth_series)

    # We want to create a "continuous" array from t=0 up to t=max(times_reference),
    # but also identify if there's any region we considered "extrapolated" deeper
    # Actually, we did the extrapolation directly in the depth calculation.
    # Let's define that the "measured domain" is anywhere the original table had data
    # near or up to the actual max depth from the data. 
    # Instead, we'll do a simpler approach: 
    # We'll consider the last measured depth from the actual data for this curve at each time
    # But that requires a time-based approach. 
    # Simpler: We'll just say the entire curve is valid, 
    # and if the depth is exactly at plot_max_depth at some portion, that portion is "dashed."

    # We'll break the array at the point where depth_array hits plot_max_depth
    # Actually, the code might do so multiple times. We'll do a simple approach:
    # we find the largest depth in the original measured data for that curve. 
    # But we don't actually have a direct "largest measured" array for each perc.

    # => We'll do a simpler approach: 
    # Plot the entire line in solid. Then re-plot the portion that is "clamped" at max depth" as dashed.
    # But that might be complicated if partial points are at max_depth.

    # Let's do a more direct approach:
    # We'll just find segments in time_series that are valid and continuous,
    # within each segment, check if we are at plot_max_depth or not.
    # We'll define a small function to chunk out continuous valid segments:
    def chunkify_time_depth(t, d, mask):
        # returns list of segments (t_seg, d_seg)
        segments = []
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            return segments
        start = idxs[0]
        for i in range(1, len(idxs)):
            if idxs[i] != idxs[i-1] + 1:
                # break
                segments.append((t[start:idxs[i-1]+1], d[start:idxs[i-1]+1]))
                start = idxs[i]
        # final
        segments.append((t[start:idxs[-1]+1], d[start:idxs[-1]+1]))
        return segments

    segments = chunkify_time_depth(time_series, depth_series, valid_mask)

    for (t_seg, d_seg) in segments:
        # We'll find any portion that is exactly plot_max_depth (extrap)
        # Let's define a threshold: if the depth is within 0.001 of plot_max_depth, we consider it "extrap."
        is_extrap = np.isclose(d_seg, plot_max_depth, atol=1e-6)
        # We'll now split the segment into sub-segments: "not extrap" (solid) vs "extrap" (dashed).
        if len(t_seg) <= 1:
            continue

        # We'll walk through the array and break whenever is_extrap changes
        substart = 0
        for i in range(1, len(t_seg)):
            if is_extrap[i] != is_extrap[i-1]:
                # plot from substart..i
                style = '--' if is_extrap[substart] else '-'
                plt_segment_t = t_seg[substart:i]
                plt_segment_d = d_seg[substart:i]
                ax.plot(plt_segment_t, plt_segment_d, color=color, linestyle=style)
                substart = i

        # final sub
        style = '--' if is_extrap[substart] else '-'
        plt_segment_t = t_seg[substart:]
        plt_segment_d = d_seg[substart:]
        ax.plot(plt_segment_t, plt_segment_d, color=color, linestyle=style, label=f'{perc}%')

# Format the plot
ax.set_xlabel('Time (min)', fontsize=14, weight='bold')
ax.set_ylabel(f'Depth ({depth_units})', fontsize=14, weight='bold')
ax.set_title('Isoremoval Curves', fontsize=16, weight='bold')
ax.set_ylim(plot_max_depth, 0)
ax.grid(True, linestyle='--', linewidth=0.5)

# Because we label each sub-segment, the legend might have duplicates.
# We'll fix that by a quick legend handle fix.
handles, labels = ax.get_legend_handles_labels()
temp = {}
unique_handles = []
unique_labels = []
for h, l in zip(handles, labels):
    if l not in temp:
        unique_handles.append(h)
        unique_labels.append(l)
        temp[l] = 1

legend = ax.legend(
    unique_handles,
    unique_labels,
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

# Optional shadow patch behind legend
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

plt.subplots_adjust(bottom=0.3)
st.pyplot(fig)

# -------------------------------------------------------------------------
# Show the Interpolated Depths Table
# -------------------------------------------------------------------------
st.subheader("Interpolated Depths Table")
st.write("Depth (m) at which each % removal occurs, for each time in `times_reference` (including extrapolation).")
interp_disp = interpolated_depths.round(3)
interp_disp.index.name = "Time (min)"
st.dataframe(interp_disp)

# -------------------------------------------------------------------------
# Subplots for each % removal
# -------------------------------------------------------------------------
st.subheader("Subplots of Each % Removal")

n_sub = len(percent_removal_reference)
n_cols = 4
n_rows = (n_sub + n_cols - 1) // n_cols

fig_sub, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                             figsize=(5*n_cols, 4*n_rows),
                             constrained_layout=True)

axes = axes.flatten()
cmap_sub = plt.get_cmap('tab20')

for i, perc in enumerate(percent_removal_reference):
    axx = axes[i]
    d_series = interpolated_depths[perc].values.astype(float)
    t_series = interpolated_depths.index.values.astype(float)
    mask = ~np.isnan(d_series)
    
    # We'll replicate the "split into segments" approach quickly for each subplot
    def chunkify_time_depth(t, d, m):
        segs = []
        idxs = np.where(m)[0]
        if len(idxs) == 0:
            return segs
        start = idxs[0]
        for j in range(1, len(idxs)):
            if idxs[j] != idxs[j-1] + 1:
                segs.append((t[start:idxs[j-1]+1], d[start:idxs[j-1]+1]))
                start = idxs[j]
        segs.append((t[start:idxs[-1]+1], d[start:idxs[-1]+1]))
        return segs

    segments_sub = chunkify_time_depth(t_series, d_series, mask)
    color_sub = cmap_sub(i/len(percent_removal_reference))

    for (tt, dd) in segments_sub:
        # dash or solid if dd == plot_max_depth
        is_ext = np.isclose(dd, plot_max_depth, atol=1e-6)
        substart = 0
        for k in range(1, len(tt)):
            if is_ext[k] != is_ext[k-1]:
                style_ = '--' if is_ext[substart] else '-'
                axx.plot(tt[substart:k], dd[substart:k], color=color_sub, linestyle=style_, marker='o')
                substart = k
        # final
        style_ = '--' if len(is_ext) > 0 and is_ext[substart] else '-'
        axx.plot(tt[substart:], dd[substart:], color=color_sub, linestyle=style_, marker='o', label=f'{perc}%')

    axx.invert_yaxis()
    axx.set_title(f'{perc}% Removal', fontsize=12, weight='bold')
    axx.set_xlabel("Time (min)", fontsize=10, weight='bold')
    axx.set_ylabel(f"Depth ({depth_units})", fontsize=10, weight='bold')
    axx.grid(True, linestyle='--', linewidth=0.5)
    axx.legend(fontsize=8)

for j in range(i+1, len(axes)):
    axes[j].axis('off')

fig_sub.suptitle("Isoremoval Curves - Subplots", fontsize=16, weight='bold')
st.pyplot(fig_sub)

# -------------------------------------------------------------------------
# Suspended Solids Removal vs. Detention Time & Overflow Rate
# -------------------------------------------------------------------------
st.header("Suspended Solids Removal vs. Detention Time & Overflow Rate")

st.markdown("""
This section finds the **time** when each isoremoval curve meets the bottom (the specified max depth). 
Then, it approximates total removal at that time by a vertical integration of all isoremoval curves, 
and finally computes the corresponding **overflow rate** (m/d). 

We then plot:
1. **Removal % vs. Detention Time (hours)**  
2. **Removal % vs. Overflow Rate (m/d)**
""")

def find_time_for_max_depth(perc):
    """
    For the isoremoval curve 'perc', we have depth vs. time in 'interpolated_depths'.
    Solve for time where depth == plot_max_depth.
    """
    depth_series = interpolated_depths[perc].dropna()
    if depth_series.empty:
        return None
    try:
        d_vals = depth_series.values.astype(float)
        t_vals = depth_series.index.values.astype(float)
    except:
        return None

    # Because we do lots of "upside-down" logic, let's just do a direct interpolation
    # of t vs. d. We'll define fill_value="extrapolate" to ensure we can reach the domain.
    # We do want to see if the user wants strictly the point at max_depth.

    # We might do a check if the entire array is < max_depth or == max_depth, etc.
    d_min, d_max = d_vals.min(), d_vals.max()
    low, high = min(d_min, d_max), max(d_min, d_max)
    # If max_depth not in [low, high], let's do an extrapolation anyway
    # We'll not bail out here, we'll try to extrapolate
    try:
        f_td = interp1d(d_vals, t_vals, kind='linear', fill_value="extrapolate")
        candidate_t = f_td(plot_max_depth)
        if np.isnan(candidate_t) or candidate_t < 0:
            return None
        # also check time domain
        # if candidate_t > times_reference.max()+100: # some big tolerance
        #     return None
        return float(candidate_t)
    except:
        return None

def compute_vertical_removal_fraction(t_vertical):
    """
    We'll gather each % curve's depth at t_vertical,
    sort them from shallowest to deepest,
    and do a piecewise trapezoid in removal% from 0..max_depth.
    Returns approximate overall removal% (0-100).
    """
    pairs = []
    if t_vertical not in interpolated_depths.index:
        # If we want, we can do an extra interpolation among times_reference
        # but let's keep it simple
        return np.nan

    for R in percent_removal_reference:
        d = interpolated_depths.loc[t_vertical, R]
        if pd.isna(d) or d < 0 or d > plot_max_depth:
            continue
        pairs.append((R, d))

    if len(pairs) < 2:
        return np.nan

    pairs_sorted = sorted(pairs, key=lambda x: x[1])
    total_depth = plot_max_depth
    total_area = 0.0

    prev_removal, prev_depth = pairs_sorted[0]
    for i in range(1, len(pairs_sorted)):
        curr_removal, curr_depth = pairs_sorted[i]
        delta_depth = curr_depth - prev_depth
        if delta_depth < 0:
            continue
        avg_removal = (prev_removal + curr_removal)/2.0
        frac_of_col = delta_depth / total_depth
        total_area += avg_removal * frac_of_col
        prev_removal, prev_depth = curr_removal, curr_depth

    return total_area

results_list = []
for R in percent_removal_reference:
    t_intersect = find_time_for_max_depth(R)
    if t_intersect is None:
        continue
    if t_intersect <= 1e-9:
        continue
    # Overflow rate
    v_o = (plot_max_depth / t_intersect)*1440.0
    # Vertical line overall removal
    R_total = compute_vertical_removal_fraction(t_intersect)
    t_hours = t_intersect/60.0

    results_list.append({
        'Isoremoval_Curve_%': R,
        'Time_Intersect_Bottom_min': t_intersect,
        'Detention_Time_h': t_hours,
        'Overflow_Rate_m_d': v_o,
        'Overall_Removal_%': R_total
    })

if len(results_list) == 0:
    st.warning("""
No isoremoval curves intersect the specified maximum depth (or the code couldn't find a valid intersection)
even with extrapolation. 
If you still get no intersections, your data might produce 100% removal before the bottom, 
or something else might be going on.
""")
else:
    results_df = pd.DataFrame(results_list).sort_values('Detention_Time_h')
    st.subheader("Summary of Intersection Times & Computed Removals")
    st.dataframe(results_df.round(2))

    # Plot Overall_Removal_% vs. Detention Time
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

    # Plot Overall_Removal_% vs. Overflow Rate (m/d)
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
    **Note**:  
    1. We're applying linear extrapolation so that each curve can (in principle) 
       reach the maximum depth.  
    2. The vertical integration for "Overall Removal %" is a simple piecewise trapezoid 
       among the isoremoval curves we do have at that moment in time. 
    """)

st.success("Done! You can adjust inputs or upload different data as needed.")