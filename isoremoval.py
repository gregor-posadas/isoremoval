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
# 1. Let the user upload CSV or Excel
# -------------------------------------------------------------------------
st.sidebar.header("Upload Data File")
uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx", "xls"]
)

# Provide a sample file for user reference
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
st.sidebar.markdown("""
**Expected Format:**  
- First column: Depth (m)  
- Subsequent columns: Times in minutes  
- The rows: Measured concentrations (mg/L)  

For CSV, the header row is the same (first column "Depth (m)" or similar, other columns are times in minutes), 
then each row has a depth in the first field and measured concentrations in the other fields.
""")

# -------------------------------------------------------------------------
# 2. Let user select initial concentration, depth units, etc.
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

# Maximum Depth Input
max_depth = st.sidebar.number_input(
    "Maximum Depth",
    min_value=0.0,
    value=0.0,
    step=0.1,
    help="Specify the maximum depth for the plots. If 0, uses the maximum depth from data."
)

# -------------------------------------------------------------------------
# 3. Load Data Function (Supports CSV or Excel)
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
# 4. Parsing function to get depths, times, concentrations
# -------------------------------------------------------------------------
def parse_input_data(df):
    try:
        # Attempt to convert everything except the first column to numeric
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Now we have numeric data for time columns, hopefully
        depths = df.iloc[:, 0].values.astype(float)  # first column = depth
        # The first row is header for times; we try to convert these to float
        # (the actual DataFrame columns are strings, e.g. '10', '20', '35' if from CSV)
        try:
            times_float = df.columns[1:].astype(float)
        except ValueError as e:
            st.error("Could not parse the column headers (times) as float. Make sure they are numeric.")
            return None, None, None

        times = times_float.values
        # The rest of the data
        concentrations = df.iloc[:, 1:].values.astype(float)

        return depths, times, concentrations
    except Exception as e:
        st.error(f"Error parsing data: {e}")
        return None, None, None

# -------------------------------------------------------------------------
# 5. Only proceed if a file is uploaded
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

# -------------------------------------------------------------------------
# 6. Validate shapes and set max depth
# -------------------------------------------------------------------------
if concentrations.shape[0] != len(depths) or concentrations.shape[1] != len(times):
    st.error("The shape of the concentrations matrix does not match the number of depths and times.")
    st.stop()

if max_depth > 0:
    plot_max_depth = max_depth
    if plot_max_depth < np.max(depths):
        st.warning(f"Specified maximum depth ({plot_max_depth} {depth_units}) is less than the max depth in data ({np.max(depths)}).")
else:
    plot_max_depth = np.max(depths)

# -------------------------------------------------------------------------
# 7. Compute Percent Removals
# -------------------------------------------------------------------------
percent_removals = (initial_concentration - concentrations) / initial_concentration * 100
removal_df = pd.DataFrame(percent_removals, index=depths, columns=times)

st.header("Generated Isoremoval Curves")

st.subheader("Percent Removal (table) vs. Time and Depth")
removal_df_display = removal_df.round(2)
removal_df_display.index.name = f"Depth ({depth_units})"
removal_df_display.columns.name = "Time (min)"
st.dataframe(removal_df_display)

# -------------------------------------------------------------------------
# 8. Choose which % removal curves to plot
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
    user_curves_list = [int(x.strip()) for x in default_curves.split(",")]

if len(user_curves_list) == 0:
    user_curves_list = [10, 20, 30, 40, 50, 60, 70, 80]

percent_removal_reference = sorted(list(set(user_curves_list)))

# -------------------------------------------------------------------------
# 9. Interpolate Depth vs. Time for each % removal
# -------------------------------------------------------------------------
times_reference = np.arange(0, max(times) + 10, step=5)

# We'll build a DataFrame indexed by times_reference, columns=the % removal
interpolated_depths = pd.DataFrame(index=times_reference, columns=percent_removal_reference)

# For each depth, define a function that returns removal% for time
interp_funcs_over_time = {}
for d in removal_df.index:
    # We have removal% at each time
    time_vals = removal_df.columns.values
    removal_vals = removal_df.loc[d].values
    # Force numeric
    time_vals = time_vals.astype(float)
    removal_vals = removal_vals.astype(float)

    interp_funcs_over_time[d] = interp1d(
        time_vals,
        removal_vals,
        kind='linear',
        fill_value="extrapolate"
    )

# Now invert that to find depth for each given % removal at each time in times_reference
for perc in percent_removal_reference:
    depth_list = []
    for t in times_reference:
        if t == 0:
            # By definition, no removal at time=0, so set depth=0 or np.nan
            depth_list.append(0.0)
            continue

        # Evaluate removal at this time for all depths
        local_removals = []
        local_depths = []
        for d in removal_df.index:
            r_val = interp_funcs_over_time[d](float(t))
            local_removals.append(r_val)
            local_depths.append(d)

        local_removals = np.array(local_removals, dtype=float)
        local_depths = np.array(local_depths, dtype=float)

        # Only consider 0 <= removal <= 100
        valid_mask = (local_removals >= 0) & (local_removals <= 100)
        vm_removals = local_removals[valid_mask]
        vm_depths = local_depths[valid_mask]

        # If we have fewer than 2 valid points, can't interpolate => NaN
        if len(vm_removals) < 2:
            depth_list.append(np.nan)
            continue

        # Interpolate depth vs. removal
        # i.e. removal -> depth
        try:
            depth_interp = interp1d(
                vm_removals,
                vm_depths,
                kind='linear',
                bounds_error=False,
                fill_value=np.nan
            )
            candidate_depth = depth_interp(float(perc))

            # Validate final depth
            if np.isnan(candidate_depth) or candidate_depth < 0 or candidate_depth > plot_max_depth:
                depth_list.append(np.nan)
            else:
                depth_list.append(candidate_depth)
        except:
            depth_list.append(np.nan)

    interpolated_depths[perc] = depth_list

interpolated_depths = interpolated_depths.apply(pd.to_numeric, errors='coerce')
interpolated_depths.replace([np.inf, -np.inf], np.nan, inplace=True)

# -------------------------------------------------------------------------
# 10. Plot the Isoremoval Curves
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 10))
plt.rcParams['text.usetex'] = False

cmap = plt.get_cmap('tab20')
colors = cmap(np.linspace(0, 1, len(percent_removal_reference)))

for perc, c in zip(percent_removal_reference, colors):
    d_series = interpolated_depths[perc].values.astype(float)
    t_series = interpolated_depths.index.values.astype(float)
    mask = (~np.isnan(d_series)) & (d_series >= 0)
    ax.plot(t_series[mask], d_series[mask], label=f'{perc}%', color=c, linewidth=1.5, marker='o', markersize=3)

ax.set_xlabel('Time (min)', fontsize=14, weight='bold')
ax.set_ylabel(f'Depth ({depth_units})', fontsize=14, weight='bold')
ax.set_title('Isoremoval Curves', fontsize=16, weight='bold')
ax.set_ylim(plot_max_depth, 0)  # invert
ax.grid(True, linestyle='--', linewidth=0.5)

plt.subplots_adjust(bottom=0.25)
legend = ax.legend(
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

st.pyplot(fig)

# -------------------------------------------------------------------------
# 11. Show the Interpolated Depths Table
# -------------------------------------------------------------------------
st.subheader("Interpolated Depths Table")
st.write("Depth (m) at which each % removal occurs, for each time in `times_reference`:")
interp_disp = interpolated_depths.round(3)
interp_disp.index.name = "Time (min)"
st.dataframe(interp_disp)

# -------------------------------------------------------------------------
# 12. Subplots for each % removal
# -------------------------------------------------------------------------
st.subheader("Subplots of Each % Removal")
n_sub = len(percent_removal_reference)
n_cols = 4
n_rows = (n_sub + n_cols - 1) // n_cols

fig_sub, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                             figsize=(5*n_cols, 4*n_rows),
                             constrained_layout=True)
axes = axes.flatten()

for i, perc in enumerate(percent_removal_reference):
    axx = axes[i]
    d_series = interpolated_depths[perc].values.astype(float)
    t_series = interpolated_depths.index.values.astype(float)
    mask = (~np.isnan(d_series)) & (d_series >= 0)
    axx.plot(t_series[mask], d_series[mask], marker='o', linewidth=1.5,
             color=cmap(i/len(percent_removal_reference)),
             markersize=3, label=f'{perc}%')
    axx.set_title(f'{perc}% Removal', fontsize=12, weight='bold')
    axx.invert_yaxis()
    axx.set_xlabel("Time (min)", fontsize=10, weight='bold')
    axx.set_ylabel(f"Depth ({depth_units})", fontsize=10, weight='bold')
    axx.grid(True, linestyle='--', linewidth=0.5)
    axx.legend(fontsize=8)

# Hide extra axes
for j in range(i+1, len(axes)):
    axes[j].axis('off')

fig_sub.suptitle("Isoremoval Curves - Subplots", fontsize=16, weight='bold')
st.pyplot(fig_sub)

# -------------------------------------------------------------------------
# 13. Suspended Solids Removal vs. Detention Time & Overflow Rate
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

# -- A safe function to find time at which a given % curve hits plot_max_depth
def find_time_for_max_depth(perc):
    """
    Interpolate the curve (depth vs. time) for 'perc' removal,
    and solve for time where depth == plot_max_depth.
    Return float (time in minutes) or None if no intersection.
    """
    # Extract the depth-series for this percent
    depth_series = interpolated_depths[perc].dropna()
    if depth_series.empty:
        return None

    # Convert to numeric
    try:
        d_vals = depth_series.values.astype(float)
        t_vals = depth_series.index.values.astype(float)
    except:
        return None

    # Check if plot_max_depth is within the min and max of d_vals
    d_min, d_max = d_vals.min(), d_vals.max()
    # Because we invert y-axis, sometimes d_min > d_max. Let's sort them:
    low, high = min(d_min, d_max), max(d_min, d_max)
    if plot_max_depth < low or plot_max_depth > high:
        return None

    # Create interpolation depth->time
    try:
        f_td = interp1d(
            d_vals,
            t_vals,
            kind='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        candidate_t = f_td(float(plot_max_depth))
        # Convert candidate_t to python float
        candidate_t = float(candidate_t)
    except:
        return None

    # Check validity
    if np.isnan(candidate_t):
        return None

    # Also ensure it's within the time domain
    if not (0 <= candidate_t <= times_reference.max() + 1e-6):
        return None

    return candidate_t

# -- A function to approximate total removal fraction at time t_vertical
def compute_vertical_removal_fraction(t_vertical):
    """
    We'll gather each % curve's depth at t_vertical, 
    sort them from shallowest to deepest, 
    and do a piecewise trapezoid integration in removal% 
    across the water column from depth=0 to depth=max_depth.
    Returns approximate overall removal% (0-100).
    """

    pairs = []
    for R in percent_removal_reference:
        d = interpolated_depths.loc[t_vertical, R] if t_vertical in interpolated_depths.index else np.nan
        # If not found exactly, we can do an interpolation:
        # (But let's keep it simple. We'll assume times_reference is fine enough.)
        if pd.isna(d) or d < 0 or d > plot_max_depth:
            continue
        pairs.append((R, d))

    # If no pairs, return None
    if len(pairs) < 2:
        return np.nan

    # Sort by depth ascending
    pairs_sorted = sorted(pairs, key=lambda x: x[1])
    total_depth = plot_max_depth
    total_area = 0.0

    prev_removal, prev_depth = pairs_sorted[0]
    # If the top is > prev_depth, that means from 0 to prev_depth we don't know removal?
    # For simplicity, we skip the region above the shallowest known point 
    # or assume removal=0 above that point. Adjust if needed.

    for i in range(1, len(pairs_sorted)):
        curr_removal, curr_depth = pairs_sorted[i]
        delta_depth = curr_depth - prev_depth
        if delta_depth < 0:
            # skip weird ordering
            continue
        avg_removal = (prev_removal + curr_removal)/2.0
        frac_of_col = delta_depth / total_depth
        total_area += avg_removal * frac_of_col
        prev_removal, prev_depth = curr_removal, curr_depth

    return total_area  # approximate total removal as a % (0-100)

# Gather results for each % curve that intersects max_depth
results_list = []
for R in percent_removal_reference:
    t_intersect = find_time_for_max_depth(R)
    if t_intersect is None:
        continue
    # Overflow rate (m/d)
    # v_o = (max_depth[m] / t[min]) * 1440 [min/day]
    if t_intersect <= 1e-9:
        continue
    v_o = (plot_max_depth / t_intersect) * 1440.0

    # vertical line overall removal
    R_total = compute_vertical_removal_fraction(t_intersect)
    # convert time to hours
    t_hours = t_intersect/60.0

    results_list.append({
        'Isoremoval_Curve_%': R,
        'Time_Intersect_Bottom_min': t_intersect,
        'Detention_Time_h': t_hours,
        'Overflow_Rate_m_d': v_o,
        'Overall_Removal_%': R_total
    })

if len(results_list) == 0:
    st.warning("No isoremoval curves intersect the specified maximum depth in the available data.")
else:
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values('Detention_Time_h').reset_index(drop=True)

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
    **Note**: This approach uses a simple piecewise trapezoid between adjacent
    % curves to approximate the total removal at a given time. In a real design scenario, 
    you may refine these assumptions or ensure the entire top region (from 0m down to the first
    measured curve) is accounted for appropriately.
    """)

st.success("Done! You can adjust inputs or upload different data as needed.")