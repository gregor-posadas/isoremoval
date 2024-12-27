import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.patches import FancyBboxPatch
import base64
from io import BytesIO

# -------------------------------------------------------------------------
# Set the page configuration
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Isoremoval Curves",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------------
# Title and description
# -------------------------------------------------------------------------
st.title("Isoremoval Curves Generator")

st.markdown("""
This application allows you to generate **Isoremoval Curves** based on your own data inputs. 
You can specify your own depths, times, concentrations, and initial concentration by uploading a properly formatted Excel file.

**New**: This version also demonstrates how to estimate **Suspended Solids Removal** vs. **Detention Time** and vs. **Overflow Rate** (m/d) 
by drawing vertical lines at the time each isoremoval curve intersects the user-selected **maximum depth**, then performing a vertical “piecewise” integration.
""")

# -------------------------------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------------------------------
st.sidebar.header("Input Parameters")

# Initial Concentration
initial_concentration = st.sidebar.number_input(
    "Initial Concentration (mg/L)",
    min_value=0.0,
    value=20.0,
    step=0.1
)

# Depth Units Selection
depth_units = st.sidebar.selectbox(
    "Select Units for Depth",
    options=["Meters (m)", "Feet (ft)", "Centimeters (cm)", "Inches (in)"],
    index=0
)

# Function to parse the uploaded Excel file
def parse_excel(file):
    try:
        df = pd.read_excel(file)
        if df.empty:
            st.error("The uploaded Excel file is empty.")
            return None, None, None
        # Assume the first column is Depths and the first row is header with times
        depths = df.iloc[:,0].values
        times = df.columns[1:].astype(float).values
        concentrations = df.iloc[:,1:].values
        
        # Validate data
        if not np.issubdtype(df.iloc[:,0].dtype, np.number):
            st.error("Depths should be numeric.")
            return None, None, None
        for col in df.columns[1:]:
            if not np.issubdtype(df[col].dtype, np.number):
                st.error(f"Time column '{col}' contains non-numeric values.")
                return None, None, None
                
        return depths, times, concentrations
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return None, None, None

# Provide a downloadable sample Excel file
def generate_sample_excel():
    # Define sample data
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

# Encode to base64 for download
def get_table_download_link():
    sample_excel = generate_sample_excel()
    b64 = base64.b64encode(sample_excel.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="isoremoval_sample.xlsx">Download Sample Excel File</a>'
    return href

# Upload Data File
st.sidebar.subheader("Upload Data File")
st.sidebar.markdown("""
Upload an Excel file containing depths, times, and concentrations.

**Expected Format:**  
- **First Row:** The first cell can be labeled (e.g., "Depth (m)"), followed by time points in minutes.  
- **First Column:** Depth values in meters corresponding to each row.  
- **Data Cells:** Concentration values (mg/L) for each depth and time.

**Example:**

| Depth (m) | 10 | 20 | 35 | 50 | 70 | 85 |
|-----------|----|----|----|----|----|----|
| 0.5       | 14 | 10 | 7  | 6.2| 5  | 4  |
| 1.0       | 15 | 13 |10.6|8.2 |7   |6   |
| 1.5       |15.4|14.2|12  |10  |7.8 |7   |
| 2.0       |16  |14.6|12.6|11  |9   |8   |
| 2.5       |17  |15  |13  |11.4|10  |8.8 |
""")
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx", "xls"])

# Provide a download link for the sample Excel file
st.sidebar.markdown(get_table_download_link(), unsafe_allow_html=True)

# Maximum Depth Input
max_depth = st.sidebar.number_input(
    "Maximum Depth",
    min_value=0.0,
    value=0.0,
    step=0.1,
    help="Specify the maximum depth for the plots. If left at 0, the program will use the maximum depth from the uploaded Excel file."
)

# If no file, stop
if uploaded_file is not None:
    depths, times, concentrations = parse_excel(uploaded_file)
    if depths is None or times is None or concentrations is None:
        st.stop()
else:
    st.sidebar.info("Please upload an Excel file to proceed.")
    st.stop()

# Validate shape
if concentrations.shape[0] != len(depths) or concentrations.shape[1] != len(times):
    st.error("The shape of the concentrations matrix does not match the number of depths and times.")
    st.stop()

# Determine maximum depth
if max_depth > 0:
    plot_max_depth = max_depth
    if plot_max_depth < np.max(depths):
        st.warning(
            f"Specified maximum depth ({plot_max_depth} {depth_units}) is less than the maximum depth in data ({np.max(depths)} {depth_units}). Some data may be excluded from the plot."
        )
else:
    plot_max_depth = np.max(depths)

# -------------------------------------------------------------------------
# Compute percent removals
# -------------------------------------------------------------------------
percent_removals = (initial_concentration - concentrations) / initial_concentration * 100
removal_df = pd.DataFrame(percent_removals, index=depths, columns=times)

# -------------------------------------------------------------------------
# 1. Display Table of Removal Efficiencies
# -------------------------------------------------------------------------
st.header("Generated Isoremoval Curves")
st.subheader("Percent Removal as a Function of Time and Depth")

removal_df_display = removal_df.copy()
removal_df_display.index.name = f"Depth ({depth_units})"
removal_df_display.columns.name = "Time (min)"

# Round for display
removal_df_display = removal_df_display.round(2)

st.dataframe(removal_df_display)

# -------------------------------------------------------------------------
# 2. User-Specified (or Default) % Removal Curves to Plot
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
    user_curves_list = [int(x.strip()) for x in default_curves.split(",")]

# Sort unique
percent_removal_reference = sorted(list(set(user_curves_list)))

# -------------------------------------------------------------------------
# 3. Interpolate for Isoremoval Curves
# -------------------------------------------------------------------------
# We'll create a higher-resolution time array for interpolation.
times_reference = np.arange(0, max(times) + 10, step=5)

# Prepare a DataFrame to hold interpolated depths for each % curve
interpolated_depths = pd.DataFrame(index=times_reference, columns=percent_removal_reference)

# Create interpolation over time for each depth
interp_funcs_over_time = {}
for depth_val in removal_df.index:
    # removal_df.loc[depth_val] are the removal% values vs. time
    interp_funcs_over_time[depth_val] = interp1d(
        removal_df.columns,
        removal_df.loc[depth_val],
        kind='linear',
        fill_value='extrapolate'
    )

# For each % removal in percent_removal_reference, invert to find depth vs time
# i.e. for each time in times_reference, we find the depth whose removal is "percent".
# We'll do it by re-interpolating over depth for that time.
for percent in percent_removal_reference:
    depths_list = []
    for time_val in times_reference:
        if time_val == 0:
            # At time=0, by definition, no removal has happened => depth=0 or NaN?
            # We'll just record 0 for continuity.
            depths_list.append(0.0)
            continue

        # Evaluate removal at this time for all depths
        removal_at_time_all_depths = []
        for d in removal_df.index:
            removal_at_time_all_depths.append(interp_funcs_over_time[d](time_val))
        removal_at_time_all_depths = np.array(removal_at_time_all_depths)

        # We only consider depth values for which removal% is between 0 and 100
        # (or 0 to 100+ if you'd like to allow overshoot). 
        valid_mask = (removal_at_time_all_depths >= 0) & (removal_at_time_all_depths <= 100)
        valid_depths = removal_df.index[valid_mask]
        valid_removals = removal_at_time_all_depths[valid_mask]

        if len(valid_removals) < 2:
            # Not enough points to interpolate => NaN
            depths_list.append(np.nan)
            continue

        # Interpolate depth vs. removal
        depth_vs_removal = interp1d(
            valid_removals,
            valid_depths,
            kind='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        # Attempt to find the depth where removal% = 'percent'
        calc_depth = depth_vs_removal(percent)

        # If outside the range or invalid, set to NaN
        if np.isnan(calc_depth) or calc_depth < 0 or calc_depth > plot_max_depth:
            depths_list.append(np.nan)
        else:
            depths_list.append(calc_depth)

    interpolated_depths[percent] = depths_list

# Clean up
interpolated_depths = interpolated_depths.apply(pd.to_numeric, errors='coerce')
interpolated_depths.replace([np.inf, -np.inf], np.nan, inplace=True)

# -------------------------------------------------------------------------
# 4. Plot the Isoremoval Curves
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 10))
plt.rcParams['text.usetex'] = False  # Disable LaTeX

cmap = plt.get_cmap('tab20')
colors = cmap(np.linspace(0, 1, len(percent_removal_reference)))

for percent, color in zip(percent_removal_reference, colors):
    times_with_origin = interpolated_depths.index
    depths_with_origin = interpolated_depths[percent].values.astype(float)
    mask = (~np.isnan(depths_with_origin)) & (depths_with_origin >= 0)
    ax.plot(
        times_with_origin[mask],
        depths_with_origin[mask],
        label=f'{percent:.0f}% Removal',
        color=color,
        linewidth=1.5,
        marker='o',
        markersize=3
    )

ax.set_xlabel('Time (min)', fontsize=14, weight='bold')
ax.set_ylabel(f'Depth ({depth_units})', fontsize=14, weight='bold')
ax.set_title('Isoremoval Curves', fontsize=16, weight='bold')
ax.set_ylim(plot_max_depth, 0)  # Invert Y-axis
ax.grid(color='gray', linestyle='--', linewidth=0.5)

plt.subplots_adjust(bottom=0.25)
legend = ax.legend(
    title='Percent Removal',
    loc='upper center',
    bbox_to_anchor=(0.5, -0.25),
    ncol=7,
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

# Add a little shadow effect behind the legend
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

plt.tight_layout()
st.pyplot(fig)

# -------------------------------------------------------------------------
# 5. Display the Interpolated Depths Table
# -------------------------------------------------------------------------
st.subheader("Interpolated Depths Table")
st.write("Each cell represents the depth at which a specific percent removal occurs at a given time.")

interpolated_depths_display = interpolated_depths.round(2)
interpolated_depths_display.index.name = "Time (min)"
st.dataframe(interpolated_depths_display)

# -------------------------------------------------------------------------
# 6. Isoremoval Subplots for Each % Removal (optional visualization)
# -------------------------------------------------------------------------
st.subheader("Isoremoval Subplots for Each Percent Removal")

n_subplots = len(percent_removal_reference)
n_cols = 4
n_rows = (n_subplots + n_cols - 1) // n_cols

fig_sub, axes = plt.subplots(
    nrows=n_rows,
    ncols=n_cols,
    figsize=(5 * n_cols, 4 * n_rows),
    constrained_layout=True
)

axes = axes.flatten()
for idx, percent in enumerate(percent_removal_reference):
    ax_sub = axes[idx]
    times_sub = interpolated_depths.index
    depths_sub = interpolated_depths[percent].values.astype(float)
    mask_sub = (~np.isnan(depths_sub)) & (depths_sub >= 0)

    ax_sub.plot(
        times_sub[mask_sub],
        depths_sub[mask_sub],
        label=f'{percent:.0f}% Removal',
        color=cmap(idx / len(percent_removal_reference)),
        linewidth=1.5,
        marker='o',
        markersize=3
    )
    ax_sub.set_title(f'{percent:.0f}% Removal', fontsize=12, weight='bold')
    ax_sub.set_xlabel('Time (min)', fontsize=10, weight='bold')
    ax_sub.set_ylabel(f'Depth ({depth_units})', fontsize=10, weight='bold')
    ax_sub.invert_yaxis()
    ax_sub.grid(color='gray', linestyle='--', linewidth=0.5)
    ax_sub.legend(fontsize=8, frameon=True)

for ax_off in axes[n_subplots:]:
    ax_off.axis('off')

fig_sub.suptitle('Isoremoval Curves', fontsize=16, weight='bold')
st.pyplot(fig_sub)

# -------------------------------------------------------------------------
# 7. Additional: Suspended Solids Removal vs. Detention Time & Overflow Rate
# -------------------------------------------------------------------------
st.header("Suspended Solids Removal vs. Detention Time / Overflow Rate")

st.markdown("""
Below is a simplified demonstration of how to:  
1. Determine **the time** at which a specific isoremoval curve intersects the **maximum depth** (user-specified).  
2. Compute the corresponding **overflow rate** in m/d.  
3. Draw a **vertical line** at that time to intersect all other isoremoval curves and approximate the **overall** removal fraction by a piecewise method.  
4. Repeat for each isoremoval curve that does intersect the maximum depth.  
5. Plot the resulting removal fraction vs. **time** (converted to hours) and vs. **overflow rate** (m/d).  
""")

# A helper function to find the exact time at which a given % removal curve hits max_depth
# We'll use an interpolation in "time" => "depth" for that curve to solve for time where depth = max_depth.
def find_time_for_max_depth(percent):
    """
    For the isoremoval curve 'percent', we have depth vs. time in 'interpolated_depths'.
    We want to find the time t (in minutes) such that depth(t) = max_depth.
    We'll do a 1D interpolation of t vs. depth and invert it.
    Returns None if it does not intersect in the range [min_time, max_time].
    """
    # Pull out the series of depths for this isoremoval line
    depth_series = interpolated_depths[percent].dropna()
    if depth_series.empty:
        return None

    # If the entire line is above/below max_depth, no intersection
    # We only consider times from 0 up to the maximum time in times_reference
    t_vals = depth_series.index.values
    d_vals = depth_series.values

    # If there's no sign change around (d_vals - max_depth), no intersection
    # But let's do a safe approach: we can create an interpolation function time=>depth
    # then invert. We'll do a quick check for min/max first.
    d_min, d_max = d_vals.min(), d_vals.max()
    if (d_max < plot_max_depth) or (d_min > plot_max_depth):
        # This means the curve never crosses max_depth in the dataset
        return None

    # We'll define a direct 1D interpolation time->depth
    # Then invert for time->(where depth=plot_max_depth).
    f_td = interp1d(d_vals, t_vals, kind='linear', bounds_error=False, fill_value=np.nan)
    t_intersect = f_td(plot_max_depth)
    # Check if t_intersect is in the domain
    if np.isnan(t_intersect):
        return None
    # Also ensure it's within min and max time range
    if t_intersect < 0 or t_intersect > times_reference.max():
        return None

    return float(t_intersect)

# A function to compute the overall removal fraction at a given time t_vertical
# by "vertical integration" across all user-specified isoremoval curves.
def compute_vertical_removal_fraction(t_vertical):
    """
    We'll do a piecewise trapezoid approximation from the top (shallow depth)
    to the bottom (max_depth). For each consecutive pair of % removal lines
    (sorted from smallest to largest), we find the depth of each at t_vertical.
    Then each "band" of depth contributes a fraction of total depth times
    the average removal in that band.
    
    Returns removal fraction in % (0 to 100+ possible).
    """
    # Get each % curve's depth at t_vertical
    # We'll skip any that are NaN or <0 or > max_depth
    pairs = []
    for R in percent_removal_reference:
        d = interpolated_depths.loc[t_vertical, R]
        if np.isnan(d):
            continue
        if d < 0:
            continue
        if d > plot_max_depth:
            continue
        pairs.append((R, d))

    # If no pairs, return None or 0
    if len(pairs) < 2:
        return np.nan

    # Sort by depth ascending (top to bottom)
    # Typically smaller depth => smaller removal in typical scenarios,
    # but let's rely on the actual depth to ensure top->bottom ordering.
    pairs_sorted = sorted(pairs, key=lambda x: x[1])  # ascending by depth
    # Example: [ (R1, d1), (R2, d2), ... ] where d1 < d2 < ...

    total_depth = plot_max_depth
    total_area = 0.0  # We'll do area in "removal% * fraction_of_depth"
    prev_removal, prev_depth = pairs_sorted[0]

    # If the top is above the first known depth, we might have a region from 0 to d1 with removal= ??? 
    # This can get complicated. For simplicity, we start from the shallowest known depth in pairs_sorted.
    # If we wanted to start from depth=0 at 0% removal, we could add an artificial pair (0,0).

    for i in range(1, len(pairs_sorted)):
        curr_removal, curr_depth = pairs_sorted[i]
        # Depth band
        delta_depth = (curr_depth - prev_depth)
        if delta_depth < 0:
            # Possibly reversed order or something unexpected, skip
            continue
        # Average removal for the band
        avg_removal = (prev_removal + curr_removal) / 2.0
        # Fraction of total depth
        frac_of_col = delta_depth / total_depth
        # Contribution
        total_area += avg_removal * frac_of_col
        # Move forward
        prev_removal, prev_depth = curr_removal, curr_depth

    # If there's still some depth from the last known curve to the bottom, 
    # you might define an extrapolation or assume last removal extends to bottom.
    # For now, let's assume no additional info => no additional removal from the last depth to bottom.

    return total_area  # This is the approximate "overall" removal %, 0-100 range.

# We'll now gather the relevant data for each isoremoval curve that DOES intersect max_depth.
results_list = []
for R in percent_removal_reference:
    t_intersect = find_time_for_max_depth(R)
    if t_intersect is None:
        continue  # This % curve never hits the bottom or out of range
    # Overflow rate in m/d, given T in minutes
    # v_o = ( (max_depth [m]) / (t [min]) ) * (1440 [min/day])
    v_o = (plot_max_depth / t_intersect) * 1440.0
    # Now compute the total removal fraction at that time (vertical line)
    R_total = compute_vertical_removal_fraction(t_intersect)
    # Convert time to hours
    t_hours = t_intersect / 60.0

    results_list.append({
        'Isoremoval_Curve_%': R,
        'Time_Intersect_Bottom_min': t_intersect,
        'Detention_Time_h': t_hours,
        'Overflow_Rate_m_d': v_o,
        'Overall_Removal_%': R_total
    })

if len(results_list) == 0:
    st.warning("No isoremoval curves intersect the specified maximum depth within the available data.")
else:
    results_df = pd.DataFrame(results_list)
    st.dataframe(results_df.round(2))

    # Now let's plot Overall_Removal_% vs. Detention_Time_h
    fig_removal_time, ax_rt = plt.subplots(figsize=(7,5))
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
    st.pyplot(fig_removal_time)

    # Plot Overall_Removal_% vs. Overflow_Rate_m_d
    fig_removal_vo, ax_vo = plt.subplots(figsize=(7,5))
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
    st.pyplot(fig_removal_vo)

st.markdown("""
**Notes:**  
1. These two plots (Removal vs. Time, Removal vs. Overflow Rate) reflect only those isoremoval curves 
   that actually intersect the bottom (i.e., maximum depth).  
2. The overall removal fraction is computed by a simple piecewise trapezoid method along a vertical line at the chosen time.  
3. You can refine or modify this approach to match the exact method you need (e.g., the example step-by-step summation you provided).  
""")