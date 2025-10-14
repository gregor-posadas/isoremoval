import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from matplotlib.patches import FancyBboxPatch
import base64
from io import BytesIO

# -------------------------------------------------------------------------
# Set page config
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Isoremoval Curves Generator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------------
# Title
# -------------------------------------------------------------------------
st.title("Isoremoval Curves Generator")

st.markdown("""
This tool analyzes settling column data to generate isoremoval curves and calculate turbidity removal efficiency.
Upload your data file and adjust parameters in the sidebar to begin.
""")

with st.expander("❓ Quick Start Guide", expanded=False):
    st.markdown("""
    ### First Time Using This Tool?
    
    **Step 1: Prepare Your Data**
    - Format: CSV or Excel file
    - First column: Depths (in meters, feet, cm, or inches)
    - Column headers: Time points (in minutes, must be numbers)
    - Data cells: Turbidity measurements (NTU)
    - [Download the sample file](#) to see the expected format
    
    **Step 2: Upload & Configure**
    1. Upload your data file using the sidebar
    2. Enter your **initial turbidity** (before particles were added)
    3. Select your depth units
    4. Adjust the maximum depth if needed (0 = use data's max)
    
    **Step 3: Adjust Processing (Optional)**
    - **Enable Smoothing**: Reduces noise in your measurements (recommended for real lab data)
    - **Smoothing Strength**: Higher values = smoother curves (start with default)
    - **Enforce Monotonicity**: Prevents curves from going backward (recommended: ON)
    - **Auto-filter Curves**: Only shows curves within your data range (recommended: ON)
    
    **Step 4: Choose Curves to Plot**
    - Enter percentages separated by commas (e.g., "30,40,50,60,70")
    - Tool will warn you if curves are outside your data range
    
    ---
    
    ### Common Issues & Solutions
    
    | Problem | Solution |
    |---------|----------|
    | "No curves to plot" error | Disable auto-filter OR choose percentages within your removal range |
    | Jagged, noisy curves | Increase smoothing strength (try 1.5 or 2.0) |
    | Curves look too smooth | Decrease smoothing strength (try 0.3-0.5) |
    | Missing final plots | Curves may not reach the bottom; check your data range |
    | Wrong removal values | Double-check your initial turbidity value! |
    | "Times must be numeric" error | Column headers must be numbers (not "10min") |
    
    ---
    
    ### Tips for Best Results
    - ✅ Make sure your initial turbidity is accurate - all calculations depend on it!
    - ✅ Run your settling test long enough to see meaningful changes
    - ✅ Take measurements at consistent time intervals
    - ✅ Use the Data Quality Assessment to check if your test duration was sufficient
    """)

st.markdown("---")

# -------------------------------------------------------------------------
# Sidebar: Provide sample file & example
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
    buf = BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return buf

def get_sample_link():
    sample = generate_sample_excel()
    b64 = base64.b64encode(sample.read()).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="sample_isoremoval.xlsx">Download Sample Excel</a>'

st.sidebar.markdown(get_sample_link(), unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel",
    type=["csv","xlsx","xls"]
)

# -------------------------------------------------------------------------
# Sidebar: Input Parameters
# -------------------------------------------------------------------------
st.sidebar.header("Input Parameters")

init_conc = st.sidebar.number_input(
    "Initial Turbidity (NTU)",
    min_value=0.0,
    value=20.0,
    step=0.1
)

depth_units = st.sidebar.selectbox(
    "Depth Units",
    ["Meters (m)", "Feet (ft)", "Centimeters (cm)", "Inches (in)"],
    index=0
)

max_depth_val = st.sidebar.number_input(
    "Max Depth",
    min_value=0.0,
    value=0.0,
    step=0.1,
    help="0 => use maximum depth from data"
)

# -------------------------------------------------------------------------
# 1. Load & parse
# -------------------------------------------------------------------------
def load_data(file):
    if file is None:
        return None
    ext = file.name.split(".")[-1].lower()
    if ext in ["xlsx","xls"]:
        return pd.read_excel(file)
    elif ext in ["csv"]:
        return pd.read_csv(file)
    else:
        st.error("Unsupported file format.")
        return None

if not uploaded_file:
    st.info("Upload a file to continue.")
    st.stop()

df = load_data(uploaded_file)
if df is None:
    st.stop()

def parse_data(dframe):
    try:
        for col in dframe.columns[1:]:
            dframe[col] = pd.to_numeric(dframe[col], errors='coerce')
        depths_ = dframe.iloc[:,0].values.astype(float)
        try:
            times_ = dframe.columns[1:].astype(float)
        except:
            st.error("Times must be numeric in column headers.")
            return None,None,None
        cvals_ = dframe.iloc[:,1:].values.astype(float)
        return depths_, times_, cvals_
    except Exception as e:
        st.error(f"Error parsing data: {e}")
        return None,None,None

depths, times, concentrations = parse_data(df)
if depths is None or times is None or concentrations is None:
    st.stop()

if concentrations.shape!=(len(depths), len(times)):
    st.error("Data shape mismatch.")
    st.stop()

# -------------------------------------------------------------------------
# NEW: Assess Data Quality and Stratification
# -------------------------------------------------------------------------
def assess_data_stratification(conc_matrix, depths_arr, init_c):
    """Assess how well-stratified the settling data is"""
    # Look at final time point
    final_concs = conc_matrix[:, -1]
    conc_range = np.max(final_concs) - np.min(final_concs)
    conc_variation = conc_range / np.max(final_concs) * 100
    
    # Calculate removals at final time
    final_removals = (init_c - final_concs) / init_c * 100
    removal_range = np.max(final_removals) - np.min(final_removals)
    
    # Check for non-monotonic behavior
    issues = []
    for i in range(len(depths_arr)):
        for j in range(1, conc_matrix.shape[1]):
            if conc_matrix[i, j] > conc_matrix[i, j-1]:
                issues.append((depths_arr[i], j))
    
    return {
        'conc_range': conc_range,
        'conc_variation': conc_variation,
        'removal_range': removal_range,
        'non_monotonic_count': len(issues),
        'stratification_quality': 'High' if removal_range > 25 else 'Medium' if removal_range > 10 else 'Low'
    }

quality = assess_data_stratification(concentrations, depths, init_conc)

# Display data quality assessment
with st.expander("📊 Data Quality Assessment", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Stratification Quality", quality['stratification_quality'])
        if quality['stratification_quality'] == 'Low':
            st.warning("⚠️ Low stratification detected")
    
    with col2:
        st.metric("Removal Range", f"{quality['removal_range']:.1f}%")
        if quality['removal_range'] < 10:
            st.info("Small removal range - test may need longer duration")
    
    with col3:
        st.metric("Turbidity Variation", f"{quality['conc_variation']:.1f}%")
    
    if quality['non_monotonic_count'] > 0:
        st.warning(f"⚠️ {quality['non_monotonic_count']} non-monotonic data points detected (measurement noise)")
    
    if quality['stratification_quality'] == 'Low':
        st.info("""
        **Low Stratification Recommendations:**
        - The test may not have run long enough for significant settling
        - Consider using higher smoothing values
        - Focus on curves that fall within your data's removal range
        - Results may be less reliable than highly-stratified data
        """)

# -------------------------------------------------------------------------
# Data Processing Options (with smart defaults)
# -------------------------------------------------------------------------
st.sidebar.header("Data Processing Options")

# Auto-adjust defaults based on data quality
default_smoothing = 1.5 if quality['stratification_quality'] == 'Low' else 0.8 if quality['stratification_quality'] == 'Medium' else 0.5
default_enable_smoothing = quality['non_monotonic_count'] > 0

enable_smoothing = st.sidebar.checkbox(
    "Enable Data Smoothing",
    value=default_enable_smoothing,
    help="Recommended for noisy data"
)

smoothing_sigma = st.sidebar.slider(
    "Smoothing Strength",
    min_value=0.1,
    max_value=3.0,
    value=default_smoothing,
    step=0.1,
    help=f"Recommended: {default_smoothing} for this dataset",
    disabled=not enable_smoothing
)

enforce_monotonicity = st.sidebar.checkbox(
    "Enforce Monotonicity",
    value=True,
    help="Prevent curves from reversing direction"
)

filter_curves = st.sidebar.checkbox(
    "Auto-filter Curves",
    value=True,
    help="Only show curves within the data's removal range"
)

# -------------------------------------------------------------------------
# Apply smoothing if enabled
# -------------------------------------------------------------------------
if enable_smoothing:
    concentrations_smoothed = np.copy(concentrations)
    for i in range(len(depths)):
        concentrations_smoothed[i, :] = gaussian_filter1d(concentrations[i, :], sigma=smoothing_sigma)
    concentrations_to_use = concentrations_smoothed
else:
    concentrations_to_use = concentrations

if max_depth_val>0:
    plot_max = max_depth_val
    if plot_max<np.max(depths):
        st.warning("User-specified max depth < data's actual max depth.")
else:
    plot_max = np.max(depths)

st.subheader("Your Uploaded Data (Turbidity)")
st.markdown("""
**What is this?** This is your raw experimental data showing turbidity (NTU) measurements at different depths and times.
- **Rows**: Different depths in your settling column (shallow → deep)
- **Columns**: Different times during your experiment
- **Values**: Turbidity measurements in NTU
- **Expected pattern**: In a good settling test, turbidity should generally decrease over time as particles settle out
""")
st.dataframe(df.style.format(precision=2))

# -------------------------------------------------------------------------
# 2. Compute Removals
# -------------------------------------------------------------------------
removals = (init_conc - concentrations_to_use)/init_conc*100
removal_df = pd.DataFrame(removals, index=depths, columns=times)
removal_df.index.name = f"Depth ({depth_units})"
removal_df.columns.name = "Time (min)"

# Calculate min/max removal at each time
min_removal_by_time = removal_df.min(axis=0)
max_removal_by_time = removal_df.max(axis=0)

st.subheader("Percent Removal (Table)")
st.markdown("""
**What is this?** This table converts your turbidity measurements into **percent removal** at each depth and time.

**Calculation**: `% Removal = (Initial Turbidity - Measured Turbidity) / Initial Turbidity × 100`

**How to read it:**
- **Higher %** = More turbidity removed = Better settling at that location/time
- **Lower %** = Less turbidity removed = Particles still suspended
- **Typical pattern**: Removal % increases with both time (left to right) and depth (top to bottom)

💡 **Example**: If you see 65% at depth 50m and time 30min, that means 65% of the original turbidity has been removed at that depth after 30 minutes.
""")
st.dataframe(removal_df.round(2))

# -------------------------------------------------------------------------
# 3. Build isoremoval curves
# -------------------------------------------------------------------------
st.sidebar.subheader("Which % Curves to Plot?")
default_str = "10,20,30,40,50,60,70,80"
user_str = st.sidebar.text_input("Comma %", value=default_str)
try:
    user_list = [int(x.strip()) for x in user_str.split(",")]
    user_list = [u for u in user_list if 1 <= u < 100]
except:
    user_list = [10,20,30,40,50,60,70,80]

# Filter curves based on data range if enabled
if filter_curves:
    data_min = removal_df.min().min()
    data_max = removal_df.max().max()
    user_list_filtered = [u for u in user_list if data_min <= u <= data_max]
    if len(user_list_filtered) < len(user_list):
        excluded = set(user_list) - set(user_list_filtered)
        st.info(f"ℹ️ Auto-filtered curves outside data range ({data_min:.1f}% - {data_max:.1f}%): {sorted(excluded)}")
    percent_list = sorted(list(set(user_list_filtered)))
else:
    percent_list = sorted(list(set(user_list)))

if len(percent_list) == 0:
    st.error("No curves to plot within data range. Adjust your curve selection or disable auto-filtering.")
    st.stop()

times_ref = np.arange(0, max(times)+10, step=5)
iso_depths = pd.DataFrame(index=times_ref, columns=percent_list)

interp_time_removal = {}
for d_ in removal_df.index:
    t_ = removal_df.columns.values.astype(float)
    r_ = removal_df.loc[d_].values.astype(float)
    interp_time_removal[d_] = interp1d(t_, r_, kind='linear', fill_value="extrapolate")

def invert_removal_depth(perc, tval):
    all_r = []
    all_d = []
    for d_ in removal_df.index:
        rr = interp_time_removal[d_](tval)
        all_r.append(rr)
        all_d.append(d_)

    all_r = np.array(all_r)
    all_d = np.array(all_d)

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
    return np.clip(cand, 0, plot_max)

for p_ in percent_list:
    depth_list = []
    for t_ in times_ref:
        if t_ == 0.0:
            depth_list.append(0.0)
            continue
        dd_ = invert_removal_depth(p_, t_)
        depth_list.append(dd_)

    arr_ = np.array(depth_list)
    arr_ = np.clip(arr_, 0, plot_max)
    
    # Enforce monotonicity if enabled
    if enforce_monotonicity:
        for i in range(1, len(arr_)):
            if not np.isnan(arr_[i]) and not np.isnan(arr_[i-1]):
                if arr_[i] < arr_[i-1]:
                    arr_[i] = arr_[i-1]
    
    iso_depths[p_] = arr_

def extend_final_segment(tvals, dvals, bottom):
    if len(tvals) < 2:
        return tvals, dvals
    if dvals[-1] >= bottom:
        return tvals, dvals

    d2 = dvals[-1]
    d1 = dvals[-2]
    t2 = tvals[-1]
    t1 = tvals[-2]
    if abs(t2 - t1) < 1e-12:
        return tvals, dvals

    slope = (d2 - d1) / (t2 - t1)
    if abs(slope) < 1e-12:
        return tvals, dvals

    t_ext = t2 + (bottom - d2) / slope
    if t_ext > t2:
        tv_new = np.append(tvals, t_ext)
        dv_new = np.append(dvals, bottom)
        return tv_new, dv_new
    return tvals, dvals

st.subheader("Generated Isoremoval Curves")

with st.expander("ℹ️ Understanding Isoremoval Curves", expanded=False):
    st.markdown("""
    ### What are isoremoval curves?
    Each curve represents a **constant percentage of turbidity removal**. They show how particles with different settling 
    velocities move through the column over time.
    
    ### How to read this graph:
    - **X-axis (Time)**: How long the settling process has been running
    - **Y-axis (Depth)**: How deep into the column you are
    - **Each curve**: Represents a specific % of removal (e.g., the "50%" curve)
    - **Following a curve**: Shows where that removal percentage occurs at different times
    
    ### What does a "good" curve look like?
    - ✅ **Smooth and monotonic**: Depth should always increase or stay constant as time increases (never go backwards)
    - ✅ **Clear separation**: Different % curves should be spread out, not bunched together
    - ✅ **Reaches the bottom**: Curves should extend to your maximum depth
    
    ### Physical interpretation:
    - **Steeper curves** = Particles settling faster (high settling velocity)
    - **Flatter curves** = Particles settling slower (low settling velocity)
    - **Curves close together** = Limited settling occurred (short test duration or low initial turbidity)
    
    ### Why are they useful?
    Engineers use these curves to design settling tanks and clarifiers in water treatment plants. They help determine:
    - How large a settling tank needs to be
    - What flow rate will achieve desired removal
    - How long water needs to stay in the tank
    
    ### Troubleshooting:
    - **Curves look jagged?** → Increase smoothing strength in the sidebar
    - **Missing curves?** → They may be outside your data's removal range; try different percentages
    - **Curves bunched together?** → Low stratification; test may need longer duration
    """)

fig, ax = plt.subplots(figsize=(14, 10))
cmap_ = plt.get_cmap('tab20')
colors_ = cmap_(np.linspace(0, 1, len(percent_list)))

for p_, col_ in zip(percent_list, colors_):
    d_ser = iso_depths[p_].values.astype(float)
    t_ser = iso_depths.index.values.astype(float)
    mask__ = ~np.isnan(d_ser)
    t_ok = t_ser[mask__]
    d_ok = d_ser[mask__]
    if len(t_ok) < 2:
        continue

    sidx = np.argsort(t_ok)
    t_ok = t_ok[sidx]
    d_ok = d_ok[sidx]

    t_ext, d_ext = extend_final_segment(t_ok, d_ok, plot_max)
    ax.plot(t_ext, d_ext, color=col_, label=f"{p_}%", linewidth=2.5)

ax.set_xlabel("Time (min)", fontsize=14, weight='bold')
ax.set_ylabel(f"Depth ({depth_units})", fontsize=14, weight='bold')
ax.set_title("Isoremoval Curves", fontsize=16, weight='bold')
ax.set_ylim(plot_max, 0)
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.subplots_adjust(bottom=0.25)

handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(
    handles, labels,
    title="Percent Removal",
    loc='upper center',
    bbox_to_anchor=(0.5, -0.25),
    ncol=min(6, len(percent_list)), fontsize=8, title_fontsize=10,
    frameon=True
)
legend.get_title().set_weight('bold')
st.pyplot(fig)

st.subheader("Interpolated Depths Table")

with st.expander("📖 How This Table is Created (Step-by-Step)", expanded=False):
    st.markdown("""
    ### Why Do We Need Interpolation?
    
    Your raw data gives you removal percentages at **specific depths** (e.g., 13m, 25m, 37m, etc.). 
    But what if you want to know where **exactly 50% removal** occurs? It might be between two of your measurement depths!
    
    **Interpolation** lets us estimate values between our actual measurements. This is the tedious part that this tool automates for you!
    
    ---
    
    ### Step-by-Step Process
    
    **For each time point and each desired removal percentage (e.g., 50%), here's what happens:**
    
    **Step 1: Gather Data for This Time**
    - Look at one column in the "Percent Removal Table" (one time point)
    - Example: At t=20 minutes, you might have:
      - Depth 13m → 45% removal
      - Depth 25m → 48% removal
      - Depth 37m → 52% removal  ← **We want 50%, which is between 25m and 37m!**
      - Depth 49m → 55% removal
      - (etc.)
    
    **Step 2: Find Where Your Desired % Falls**
    - We want to find where 50% removal occurs
    - We see it falls between depth 25m (48% removal) and depth 37m (52% removal)
    
    **Step 3: Linear Interpolation**
    - Use the formula: `Depth = Depth₁ + (Depth₂ - Depth₁) × (Target% - %₁) / (%₂ - %₁)`
    - Plug in values:
      - Depth₁ = 25m, %₁ = 48%
      - Depth₂ = 37m, %₂ = 52%
      - Target% = 50%
    - Calculate: `Depth = 25 + (37-25) × (50-48) / (52-48) = 25 + 12 × 0.5 = 31m`
    
    **Step 4: Record the Result**
    - At t=20 minutes, 50% removal occurs at depth = 31m
    - This goes in the table: Row "20", Column "50%", Value "31.0"
    
    **Step 5: Repeat for All Times and All Percentages**
    - Do this for every time point (every row)
    - Do this for every requested removal % (every column)
    - The result is the complete "Interpolated Depths Table" below
    
    ---
    
    ### From Table to Curves
    
    **How this becomes an isoremoval curve:**
    - Take one column (e.g., the 50% column)
    - Plot all those depths vs. their corresponding times
    - Connect the dots → You get the "50% isoremoval curve"!
    
    **That's why this table is crucial:** Each column becomes one curve in your plot.
    
    ---
    
    ### Why This is Tedious by Hand
    
    Imagine doing this manually:
    - 15 time points × 8 removal percentages = **120 interpolations**
    - Each interpolation requires:
      - Finding which two depths bracket your target %
      - Doing the math
      - Recording the result
    - **Total time: 1-2 hours of tedious calculations!**
    
    This tool does it all in seconds. 🎉
    """)

st.markdown("""
**What is this?** This table shows the **depth** where each removal percentage occurs at different times.

**How to read it:**
- **Rows**: Different time points during your experiment
- **Columns**: Different removal percentages (the curves you requested)
- **Values**: The depth (in your chosen units) where that removal % occurs at that time

💡 **Example**: If you see "45.2" at time=20min and column=50%, that means the 50% removal level 
is located at a depth of 45.2m after 20 minutes.
""")
st.dataframe(iso_depths.round(3))

# -------------------------------------------------------------------------
# Overall Removal calculations
# -------------------------------------------------------------------------
st.header("Turbidity Removal vs. Detention Time & Overflow Rate")

with st.expander("📖 How These Plots are Generated (Step-by-Step)", expanded=True):
    st.markdown("""
    ### Overview
    
    These plots show the relationship between **overall turbidity removal** and two design parameters. 
    Here's **exactly** how each data point is calculated and plotted.
    
    ---
    
    ## Step-by-Step Calculation Process
    
    ### For Each Isoremoval Curve (e.g., 30%, 40%, 50%, etc.):
    
    #### **Step 1: Find When the Curve Hits the Bottom**
    
    Look at the isoremoval curve (from the plot above) and find when it reaches your maximum depth.
    
    **Example:** 
    - Your maximum depth is 85m
    - The 50% isoremoval curve reaches 85m at time = 32.5 minutes
    - **This time (32.5 min) becomes the "detention time" for this curve**
    
    **Why?** This represents how long water would need to stay in a settling tank to achieve 
    that removal percentage across the full depth.
    
    ---
    
    #### **Step 2: Calculate Detention Time (hours)**
    
    Convert the time from minutes to hours:
    
    ```
    Detention Time (hours) = Time to reach bottom (min) / 60
    ```
    
    **Example:**
    - Time = 32.5 minutes
    - Detention Time = 32.5 / 60 = **0.542 hours**
    
    ---
    
    #### **Step 3: Calculate Overflow Rate**
    
    The overflow rate represents how fast water is flowing through per unit area:
    
    ```
    Overflow Rate (m/d) = (Column Depth / Time in minutes) × 1440 min/day
    ```
    
    **Example:**
    - Column Depth = 85m
    - Time = 32.5 minutes
    - Overflow Rate = (85 / 32.5) × 1440 = **3,764 m/d**
    
    **Why 1440?** That's the number of minutes in a day (60 min/hr × 24 hr/day), which converts our rate to m/day.
    
    ---
    
    #### **Step 4: Calculate Overall Removal at This Time**
    
    Now we need to calculate what the **overall removal** is at time = 32.5 minutes.
    
    **This is the most complex step:**
    
    **4a) Gather removal data at t=32.5 min:**
    - Go back to your Percent Removal table
    - For each depth, find what the removal % is at time = 32.5 minutes
      - At depth 13m: 62% removal
      - At depth 25m: 65% removal
      - At depth 37m: 67% removal
      - At depth 49m: 69% removal
      - At depth 61m: 71% removal
      - At depth 73m: 73% removal
      - At depth 85m: 75% removal
    
    **4b) Integrate across depths:**
    - We use **trapezoidal integration** to find the average removal across all depths
    - Formula: `Overall Removal = (1/H) × ∫ R(z) dz`
    - Where H = total depth, R(z) = removal at depth z
    
    **Simplified explanation:**
    - Imagine the column is divided into layers
    - Each layer has a certain removal %
    - We calculate the weighted average based on layer thickness
    - Result: **Overall Removal = 68.5%**
    
    **4c) Add boundary condition:**
    - At the very top (depth = 0), removal is always 100% (no particles have entered yet)
    - This is included in the integration
    
    ---
    
    #### **Step 5: Record the Data Point**
    
    Now we have one complete data point:
    - **Detention Time**: 0.542 hours
    - **Overflow Rate**: 3,764 m/d  
    - **Overall Removal**: 68.5%
    
    This creates:
    - One point on the "Detention Time vs. Removal" plot: (0.542, 68.5)
    - One point on the "Overflow Rate vs. Removal" plot: (3764, 68.5)
    
    ---
    
    #### **Step 6: Repeat for All Curves**
    
    Do Steps 1-5 for every isoremoval curve you plotted:
    - 30% curve → Calculate detention time, overflow rate, overall removal → Plot point
    - 40% curve → Calculate detention time, overflow rate, overall removal → Plot point
    - 50% curve → Calculate detention time, overflow rate, overall removal → Plot point
    - (etc.)
    
    ---
    
    #### **Step 7: Connect the Dots**
    
    - Sort all points by detention time (or overflow rate)
    - Connect them with lines
    - You now have your two engineering plots!
    
    ---
    
    ## What the Results Mean
    
    ### Detention Time Plot
    - **Shows**: How removal improves with longer settling time
    - **Trend**: Should go **UP** (more time = more removal)
    - **Engineering use**: "To get 70% removal, I need detention time ≥ X hours"
    
    ### Overflow Rate Plot  
    - **Shows**: How removal decreases with faster flow
    - **Trend**: Should go **DOWN** (faster flow = less removal)
    - **Engineering use**: "To get 70% removal, I need overflow rate ≤ Y m/day"
    
    ---
    
    ## Why This Matters
    
    These relationships let engineers:
    
    1. **Size settling tanks**: 
       - "I need 80% removal at 50,000 m³/day flow"
       - Use detention time plot → Find required detention time (say, 2 hours)
       - Calculate: Tank Volume = Flow × Detention Time = 50,000 × 2/24 = 4,167 m³
    
    2. **Calculate surface area**:
       - Use overflow rate plot → Find maximum overflow rate (say, 1,000 m/day)
       - Calculate: Area = Flow / Overflow Rate = 50,000 / 1,000 = 50 m²
    
    3. **Predict performance**:
       - "If I change my flow rate, how will removal change?"
       - Use these plots to estimate new performance
    
    Your lab data could actually design a real treatment plant! 🏗️
    """)

def find_time_for_bottom(p_):
    srs = iso_depths[p_].dropna()
    if srs.empty:
        return None

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

def depth_of_curve_at_time(p_, tval):
    allr = []
    alld = []
    for d_ in removal_df.index:
        rr = interp_time_removal[d_](tval)
        allr.append(rr)
        alld.append(d_)

    allr = np.array(allr)
    alld = np.array(alld)

    mask_ = (allr >= 0) & (allr <= 100)
    rr_ = allr[mask_]
    dd_ = alld[mask_]
    if len(rr_) < 2:
        return np.nan

    idx_s = np.argsort(rr_)
    rr_s = rr_[idx_s]
    dd_s = dd_[idx_s]

    if p_ < rr_s[0] or p_ > rr_s[-1]:
        return np.nan

    ixx = np.searchsorted(rr_s, p_)
    if ixx == 0:
        return dd_s[0]
    elif ixx >= len(rr_s):
        return dd_s[-1]

    r_lo, r_hi = rr_s[ixx - 1], rr_s[ixx]
    dd_lo, dd_hi = dd_s[ixx - 1], dd_s[ixx]
    if abs(r_hi - r_lo) < 1e-12:
        return dd_lo

    slope = (dd_hi - dd_lo) / (r_hi - r_lo)
    cand = dd_lo + slope * (p_ - r_lo)
    return np.clip(cand, 0, plot_max)

def compute_overall_removal(tval):
    """
    Compute overall removal using proper settling column theory.
    Integration accounts for removal across the entire depth at time tval.
    """
    # Collect removal percentages at each depth at this specific time
    depth_removal_pairs = []
    
    # For each measured depth, calculate what the removal is at time tval
    for d_ in removal_df.index:
        r_ = interp_time_removal[d_](tval)
        if 0 <= r_ <= 100:  # Valid removal percentage
            depth_removal_pairs.append((d_, r_))
    
    if len(depth_removal_pairs) < 2:
        return np.nan
    
    # Sort by depth
    depth_removal_pairs = sorted(depth_removal_pairs, key=lambda x: x[0])
    
    # Add top boundary (depth=0, removal=100%)
    depth_removal_pairs.insert(0, (0.0, 100.0))
    
    # If no point at bottom, extrapolate to bottom
    if depth_removal_pairs[-1][0] < plot_max:
        # Linear extrapolation to bottom
        if len(depth_removal_pairs) >= 2:
            d1, r1 = depth_removal_pairs[-2]
            d2, r2 = depth_removal_pairs[-1]
            if d2 > d1:
                slope = (r2 - r1) / (d2 - d1)
                r_bottom = r2 + slope * (plot_max - d2)
                r_bottom = np.clip(r_bottom, 0, 100)
                depth_removal_pairs.append((plot_max, r_bottom))
    
    # Perform trapezoidal integration: Overall Removal = (1/H) * ∫ R(z) dz
    H = plot_max
    integral = 0.0
    
    for i in range(1, len(depth_removal_pairs)):
        d1, r1 = depth_removal_pairs[i-1]
        d2, r2 = depth_removal_pairs[i]
        
        # Trapezoidal rule
        delta_d = d2 - d1
        avg_removal = (r1 + r2) / 2.0
        integral += avg_removal * delta_d
    
    overall_removal = integral / H
    return np.clip(overall_removal, 0, 100)

results = []
for p_ in percent_list:
    t_bot = find_time_for_bottom(p_)
    if t_bot is None or t_bot < 1e-12:
        continue

    vo = (plot_max / t_bot) * 1440.0
    R_tot = compute_overall_removal(t_bot)
    
    # RELAXED: Accept even if NaN, but skip only truly invalid values
    if np.isnan(R_tot):
        continue

    t_hrs = t_bot / 60.0
    results.append({
        "Isoremoval_Curve_%": p_,
        "Time_Intersect_Bottom_min": round(t_bot, 2),
        "Detention_Time_h": round(t_hrs, 2),
        "Overflow_Rate_m_d": round(vo, 2),
        "Overall_Removal_%": round(R_tot, 2)
    })

if not results:
    st.warning("Unable to compute detention times and overflow rates. This may occur if curves don't reach the bottom depth or if there's insufficient data for calculations.")
else:
    final_df = pd.DataFrame(results).sort_values("Detention_Time_h")
    st.subheader("Summary of Intersection Times & Computed Removals")
    st.dataframe(final_df)

    fig_dt, ax_dt = plt.subplots(figsize=(7, 5))
    ax_dt.plot(final_df['Detention_Time_h'], final_df['Overall_Removal_%'],
               marker='o', color='blue', linewidth=2, markersize=8)
    ax_dt.set_xlabel("Detention Time (hours)", fontsize=12)
    ax_dt.set_ylabel("Overall Removal (%)", fontsize=12)
    ax_dt.set_title("Turbidity Removal vs. Detention Time", fontsize=14, weight='bold')
    ax_dt.grid(True, alpha=0.3)
    st.pyplot(fig_dt)

    fig_vo, ax_vo = plt.subplots(figsize=(7, 5))
    ax_vo.plot(final_df['Overflow_Rate_m_d'], final_df['Overall_Removal_%'],
               marker='s', linestyle='--', color='red', linewidth=2, markersize=8)
    ax_vo.set_xlabel("Overflow Rate (m/d)", fontsize=12)
    ax_vo.set_ylabel("Overall Removal (%)", fontsize=12)
    ax_vo.set_title("Turbidity Removal vs. Overflow Rate", fontsize=14, weight='bold')
    ax_vo.grid(True, alpha=0.3)
    st.pyplot(fig_vo)

st.success("✓ Analysis complete!")
