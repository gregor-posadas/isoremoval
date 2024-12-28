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
# Load and parse data
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

if concentrations.shape[0] != len(depths) or concentrations.shape[1] != len(times):
    st.error("The shape of the concentrations matrix does not match the number of depths and times.")
    st.stop()

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
# Show user data
# -------------------------------------------------------------------------
st.subheader("Your Uploaded Data (Concentrations)")
st.write("Below is the raw data you uploaded:")
st.dataframe(df.style.format(precision=2))

# -------------------------------------------------------------------------
# Compute and show Percent Removals
# -------------------------------------------------------------------------
percent_removals = (initial_concentration - concentrations) / initial_concentration * 100
removal_df = pd.DataFrame(percent_removals, index=depths, columns=times)

st.subheader("Percent Removal (Table) vs. Time and Depth")
removal_df_display = removal_df.round(2)
removal_df_display.index.name = f"Depth ({depth_units})"
removal_df_display.columns.name = "Time (min)"
st.dataframe(removal_df_display)

# -------------------------------------------------------------------------
# User picks which % removal curves to plot
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
# Build the Interpolated Depths Table (no partial extrapolation)
# -------------------------------------------------------------------------
times_reference = np.arange(0, max(times) + 10, step=5)
interpolated_depths = pd.DataFrame(index=times_reference, columns=percent_removal_reference)

# Make a function for removal% vs. time for each depth
interp_funcs_over_time = {}
for d in removal_df.index:
    tvals = removal_df.columns.values.astype(float)
    rvals = removal_df.loc[d].values.astype(float)
    interp_funcs_over_time[d] = interp1d(
        tvals,
        rvals,
        kind='linear',
        fill_value="extrapolate"  # we allow time extrap, but not removal extrap
    )

# Now invert removal->depth for each time, but if out-of-range, set NaN
for perc in percent_removal_reference:
    depth_list = []
    for t_val in times_reference:
        if t_val == 0:
            # At time=0, let's define depth=0 for convenience
            depth_list.append(0.0)
            continue

        # Evaluate removal at each depth
        r_array = []
        d_array = []
        for d_val in removal_df.index:
            r_val = interp_funcs_over_time[d_val](t_val)
            r_array.append(r_val)
            d_array.append(d_val)

        r_array = np.array(r_array, dtype=float)
        d_array = np.array(d_array, dtype=float)

        # Keep only 0..100
        valid_mask = (r_array >= 0) & (r_array <= 100)
        vr = r_array[valid_mask]
        vd = d_array[valid_mask]
        if len(vr) < 2:
            depth_list.append(np.nan)
            continue

        # Sort by removal ascending
        idx_sort = np.argsort(vr)
        vr_sorted = vr[idx_sort]
        vd_sorted = vd[idx_sort]

        # If perc is out of [vr_sorted[0], vr_sorted[-1]], set NaN instead of forcing partial extrap
        if perc < vr_sorted[0] or perc > vr_sorted[-1]:
            depth_list.append(np.nan)
            continue

        # Otherwise do a standard linear interpolation within that domain
        idx_ = np.searchsorted(vr_sorted, perc)
        if idx_ == 0:
            depth_list.append(vd_sorted[0])
        elif idx_ >= len(vr_sorted):
            depth_list.append(vd_sorted[-1])
        else:
            r_lo, r_hi = vr_sorted[idx_-1], vr_sorted[idx_]
            d_lo, d_hi = vd_sorted[idx_-1], vd_sorted[idx_]
            if (r_hi == r_lo):
                depth_list.append(d_lo)
            else:
                slope = (d_hi - d_lo) / (r_hi - r_lo)
                cand_d = d_lo + slope * (perc - r_lo)
                depth_list.append(cand_d)

    depth_array = np.array(depth_list)
    # clamp final to [0, plot_max_depth] if you want
    depth_array = np.clip(depth_array, 0, plot_max_depth)
    interpolated_depths[perc] = depth_array

interpolated_depths = interpolated_depths.apply(pd.to_numeric, errors='coerce')

# -------------------------------------------------------------------------
# Final Step: Only extend the last segment if it hasn't reached bottom
# -------------------------------------------------------------------------
def extend_curve_to_bottom(t_vals, d_vals, bottom_depth):
    """
    Extend the curve from the last two valid points if the final depth < bottom_depth.
    Otherwise, do nothing. 
    """
    if len(t_vals) < 2:
        return t_vals, d_vals
    d_last = d_vals[-1]
    if d_last >= bottom_depth:
        return t_vals, d_vals
    # linear extension
    t2, d2 = t_vals[-1], d_vals[-1]
    t1, d1 = t_vals[-2], d_vals[-2]
    if t2 == t1:
        return t_vals, d_vals
    slope = (d2 - d1)/(t2 - t1)
    if abs(slope) < 1e-15:
        return t_vals, d_vals
    # Solve for t_ext so that depth=bottom_depth
    # bottom_depth = d2 + slope*(t_ext - t2)
    t_ext = t2 + (bottom_depth - d2)/slope
    if t_ext > t2:
        t_vals_ext = np.append(t_vals, t_ext)
        d_vals_ext = np.append(d_vals, bottom_depth)
        return t_vals_ext, d_vals_ext
    return t_vals, d_vals

# -------------------------------------------------------------------------
# Plot the main Isoremoval Curves
# -------------------------------------------------------------------------
st.subheader("Generated Isoremoval Curves")

fig, ax = plt.subplots(figsize=(14, 10))
plt.rcParams['text.usetex'] = False
cmap = plt.get_cmap('tab20')
colors = cmap(np.linspace(0, 1, len(percent_removal_reference)))

for perc, color in zip(percent_removal_reference, colors):
    depth_ser = interpolated_depths[perc].values.astype(float)
    time_ser = interpolated_depths.index.values.astype(float)

    mask = ~np.isnan(depth_ser)
    t_valid = time_ser[mask]
    d_valid = depth_ser[mask]
    if len(t_valid) < 2:
        continue
    # sort by time
    idxsort = np.argsort(t_valid)
    t_valid = t_valid[idxsort]
    d_valid = d_valid[idxsort]

    # extend if needed
    t_ext, d_ext = extend_curve_to_bottom(t_valid, d_valid, plot_max_depth)

    ax.plot(t_ext, d_ext, color=color, label=f'{perc}%')

ax.set_xlabel('Time (min)', fontsize=14, weight='bold')
ax.set_ylabel(f'Depth ({depth_units})', fontsize=14, weight='bold')
ax.set_title('Isoremoval Curves', fontsize=16, weight='bold')
ax.set_ylim(plot_max_depth, 0)
ax.grid(True, linestyle='--', linewidth=0.5)

# Clean up legend duplicates
handles, labels = ax.get_legend_handles_labels()
unique_labels = {}
new_handles = []
new_labels = []
for h, l in zip(handles, labels):
    if l not in unique_labels:
        unique_labels[l] = True
        new_handles.append(h)
        new_labels.append(l)

legend = ax.legend(
    new_handles, new_labels,
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
st.subheader("Interpolated Depths Table (No Partial Extrapolation)")
st.write("Depth (m) at which each % removal occurs, for each time in `times_reference` (then a final line extended if needed).")
interp_disp = interpolated_depths.round(3)
interp_disp.index.name = "Time (min)"
st.dataframe(interp_disp)

# -------------------------------------------------------------------------
# Subplots for each % removal (same final extension)
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
    d_serie = interpolated_depths[perc].values.astype(float)
    t_serie = interpolated_depths.index.values.astype(float)
    mask_ = ~np.isnan(d_serie)
    tt = t_serie[mask_]
    dd = d_serie[mask_]
    if len(tt) < 2:
        axx.set_title(f"{perc}% Removal (no data)")
        axx.invert_yaxis()
        axx.grid(True, linestyle='--', linewidth=0.5)
        continue

    idx_srt = np.argsort(tt)
    tt = tt[idx_srt]
    dd = dd[idx_srt]

    # final extension
    tt_ext, dd_ext = extend_curve_to_bottom(tt, dd, plot_max_depth)

    color_ = cmap_sub(i/len(percent_removal_reference))
    axx.plot(tt_ext, dd_ext, marker='o', linewidth=1.5,
             color=color_, markersize=3, label=f'{perc}%')
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
    We'll just do a direct time-vs-depth interpolation from the final
    (already extended) curve, then solve for depth=plot_max_depth.
    """
    depth_series = interpolated_depths[perc].dropna()
    if depth_series.empty:
        return None
    dvals = depth_series.values.astype(float)
    tvals = depth_series.index.values.astype(float)

    # sort by depth
    idx_ = np.argsort(dvals)
    d_sort = dvals[idx_]
    t_sort = tvals[idx_]

    # If the bottom is outside the min..max range, do a linear extrap
    if plot_max_depth < d_sort[0]:
        # from the first two points
        if len(d_sort) < 2:
            return None
        d1, d2 = d_sort[0], d_sort[1]
        t1, t2 = t_sort[0], t_sort[1]
        if d2 == d1:
            return None
        slope = (t2 - t1)/(d2 - d1)
        cand_t = t1 + slope*(plot_max_depth - d1)
        if cand_t < 0:
            return None
        return cand_t

    if plot_max_depth > d_sort[-1]:
        if len(d_sort) < 2:
            return None
        d1, d2 = d_sort[-2], d_sort[-1]
        t1, t2 = t_sort[-2], t_sort[-1]
        if d2 == d1:
            return None
        slope = (t2 - t1)/(d2 - d1)
        cand_t = t2 + slope*(plot_max_depth - d2)
        if cand_t < 0:
            return None
        return cand_t

    # normal in-range
    ixx = np.searchsorted(d_sort, plot_max_depth)
    if ixx == 0:
        return float(t_sort[0])
    if ixx >= len(d_sort):
        return float(t_sort[-1])
    d_lo, d_hi = d_sort[ixx-1], d_sort[ixx]
    t_lo, t_hi = t_sort[ixx-1], t_sort[ixx]
    if d_hi == d_lo:
        return float(t_lo)
    slope = (t_hi - t_lo)/(d_hi - d_lo)
    cand_t = t_lo + slope*(plot_max_depth - d_lo)
    if cand_t < 0:
        return None
    return float(cand_t)

def compute_vertical_removal_fraction(t_vertical):
    if t_vertical not in interpolated_depths.index:
        return np.nan
    pairs = []
    for R in percent_removal_reference:
        d_ = interpolated_depths.loc[t_vertical, R]
        if pd.isna(d_) or d_ < 0 or d_ > plot_max_depth:
            continue
        pairs.append((R, d_))
    if len(pairs) < 2:
        return np.nan
    # sort by depth
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
    t_int = find_time_for_max_depth(R)
    if t_int is None or t_int <= 1e-9:
        continue
    v_o = (plot_max_depth / t_int)*1440.0
    R_tot = compute_vertical_removal_fraction(t_int)
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
If your data or times_reference is sparse, or if removal hits 100% well above bottom, 
you may need a different approach.
""")
else:
    results_df = pd.DataFrame(results_list).sort_values('Detention_Time_h')
    st.subheader("Summary of Intersection Times & Computed Removals")
    st.dataframe(results_df.round(2))

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
    **Note**: 
    1. We preserve the original logic for the main interpolation, except we do **no** partial 
       extrapolation if the requested removal% is outside the measured range (we just use NaN).  
    2. We only add a final segment from the last point to the bottom if needed. 
    3. The summary of intersection times does a simple time-vs-depth interpolation to find 
       when each curve meets the bottom, with a piecewise trapezoid to estimate overall removal.
    """)

st.success("Done! You can adjust inputs or upload different data as needed.")