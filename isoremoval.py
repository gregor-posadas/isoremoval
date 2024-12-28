import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.patches import FancyBboxPatch
import base64
from io import BytesIO

# -------------------------------------------------------------------------
# Set page config
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Isoremoval Curves",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------------
# Title and Intro
# -------------------------------------------------------------------------
st.title("Isoremoval Curves Generator (CSV or Excel) - With Second Interpolation for Overall Removal")

st.markdown("""
This application:

1. Reads depths, times, and concentrations (mg/L).
2. Builds isoremoval curves for a selected range of removal percentages.
3. **Crucially**: When calculating Overall_Removal_% at the time a curve hits the bottom, 
   we do an **on-the-fly second interpolation** of all other curves at that exact time—ensuring 
   no reliance on a discrete `times_reference` row.
""")

# -------------------------------------------------------------------------
# Sidebar
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

st.sidebar.markdown("""
**Expected Format (example):**

| Depth (m) |  10 |  20 |  35 |  50 |  70 |  85 |
|-----------|-----|-----|-----|-----|-----|-----|
|  0.5      |14   |10   |7    |6.2  |5    |4    |
|  1.0      |15   |13   |10.6 |8.2  |7    |6    |
|  1.5      |15.4 |14.2 |12   |10   |7.8  |7    |
|  2.0      |16   |14.6 |12.6 |11   |9    |8    |
|  2.5      |17   |15   |13   |11.4 |10   |8.8  |
""")

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx", "xls"]
)

# Input Parameters
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
    help="If 0, uses the maximum depth from data."
)

# -------------------------------------------------------------------------
# 1. Load / parse
# -------------------------------------------------------------------------
def load_data(uploaded_file):
    if uploaded_file is None:
        return None
    ext = uploaded_file.name.split(".")[-1].lower()
    if ext in ["xlsx", "xls"]:
        return pd.read_excel(uploaded_file)
    elif ext == "csv":
        return pd.read_csv(uploaded_file)
    else:
        st.error("Unsupported file format.")
        return None

def parse_input_data(df):
    try:
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        depths = df.iloc[:,0].values.astype(float)
        try:
            times_float = df.columns[1:].astype(float)
        except:
            st.error("Could not parse column headers as numeric times.")
            return None, None, None
        times = times_float.values
        concentrations = df.iloc[:,1:].values.astype(float)
        return depths, times, concentrations
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None

# Wait for file
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
    st.error("Shapes do not match number of depths/times.")
    st.stop()

if max_depth > 0:
    plot_max_depth = max_depth
    if plot_max_depth < np.max(depths):
        st.warning(
            f"Specified max depth {plot_max_depth} {depth_units} < data's max depth {np.max(depths)}"
        )
else:
    plot_max_depth = np.max(depths)

# -------------------------------------------------------------------------
# Show user data
# -------------------------------------------------------------------------
st.subheader("Your Uploaded Data (Concentrations)")
st.write("Below is the raw data you uploaded:")
st.dataframe(df)

# -------------------------------------------------------------------------
# 2. Compute Removals
# -------------------------------------------------------------------------
percent_removals = (initial_concentration - concentrations)/initial_concentration*100
removal_df = pd.DataFrame(percent_removals, index=depths, columns=times)
removal_df.index.name = f"Depth ({depth_units})"
removal_df.columns.name = "Time (min)"

st.subheader("Percent Removal (Table) vs. Time and Depth")
st.dataframe(removal_df.round(2))

# -------------------------------------------------------------------------
# 3. Build isoremoval curves with times_reference
# -------------------------------------------------------------------------
st.sidebar.subheader("Select Which % Removal Curves to Plot")
default_curves = "10,20,30,40,50,60,70,80"
user_input = st.sidebar.text_input(
    "Comma-separated % values:",
    value=default_curves
)
try:
    user_list = [int(x.strip()) for x in user_input.split(",")]
    user_list = [v for v in user_list if 1 <= v < 100]
except:
    user_list = [10,20,30,40,50,60,70,80]

if not user_list:
    user_list = [10,20,30,40,50,60,70,80]

percent_removal_reference = sorted(list(set(user_list)))

times_reference = np.arange(0, max(times)+10, step=5)
interpolated_depths = pd.DataFrame(index=times_reference, columns=percent_removal_reference)

# For each depth, define time->removal function
interp_time_removal = {}
for d in removal_df.index:
    t_ = removal_df.columns.values.astype(float)
    r_ = removal_df.loc[d].values.astype(float)
    interp_time_removal[d] = interp1d(
        t_, r_, kind='linear', fill_value="extrapolate"
    )

# invert removal->depth for each time
for perc in percent_removal_reference:
    depth_list = []
    for tval in times_reference:
        if tval==0:
            depth_list.append(0.0)
            continue
        # Evaluate removal at all depths
        r_array, d_array = [], []
        for d_ in removal_df.index:
            rr = interp_time_removal[d_](tval)
            r_array.append(rr)
            d_array.append(d_)

        r_array = np.array(r_array)
        d_array = np.array(d_array)
        mask_ = (r_array>=0)&(r_array<=100)
        rr_val = r_array[mask_]
        dd_val = d_array[mask_]
        if len(rr_val)<2:
            depth_list.append(np.nan)
            continue
        idx_s = np.argsort(rr_val)
        rr_s = rr_val[idx_s]
        dd_s = dd_val[idx_s]
        if perc<rr_s[0] or perc>rr_s[-1]:
            depth_list.append(np.nan)
            continue
        ixx = np.searchsorted(rr_s, perc)
        if ixx==0:
            depth_list.append(dd_s[0])
        elif ixx>=len(rr_s):
            depth_list.append(dd_s[-1])
        else:
            r_lo,r_hi = rr_s[ixx-1], rr_s[ixx]
            d_lo,d_hi = dd_s[ixx-1], dd_s[ixx]
            if abs(r_hi-r_lo)<1e-12:
                depth_list.append(d_lo)
            else:
                slope = (d_hi-d_lo)/(r_hi-r_lo)
                d_cand = d_lo + slope*(perc-r_lo)
                depth_list.append(d_cand)
    depth_array = np.array(depth_list)
    depth_array = np.clip(depth_array, 0, plot_max_depth)
    interpolated_depths[perc] = depth_array

# quick function to extend final segment
def extend_curve_to_bottom(tvals,dvals,bottom):
    if len(tvals)<2:
        return tvals,dvals
    if dvals[-1]>=bottom:
        return tvals,dvals
    d1,d2 = dvals[-2], dvals[-1]
    t1,t2 = tvals[-2], tvals[-1]
    if abs(t2-t1)<1e-12:
        return tvals,dvals
    slope = (d2-d1)/(t2-t1)
    if abs(slope)<1e-15:
        return tvals,dvals
    t_ext = t2 + (bottom - d2)/slope
    if t_ext>t2:
        tvals_ext = np.append(tvals, t_ext)
        dvals_ext = np.append(dvals, bottom)
        return tvals_ext,dvals_ext
    return tvals,dvals

# -------------------------------------------------------------------------
# Plot the curves
# -------------------------------------------------------------------------
st.subheader("Generated Isoremoval Curves")

fig, ax = plt.subplots(figsize=(14,10))
colors_ = plt.get_cmap('tab20')(np.linspace(0,1,len(percent_removal_reference)))

for perc,col in zip(percent_removal_reference,colors_):
    d_series = interpolated_depths[perc].values.astype(float)
    t_series = interpolated_depths.index.values.astype(float)
    mask__ = ~np.isnan(d_series)
    tt = t_series[mask__]
    dd = d_series[mask__]
    if len(tt)<2:
        continue
    isrt = np.argsort(tt)
    tt = tt[isrt]
    dd = dd[isrt]
    # extend
    tt_ext,dd_ext = extend_curve_to_bottom(tt,dd, plot_max_depth)
    ax.plot(tt_ext, dd_ext, color=col, label=f"{perc}%")

ax.set_xlabel("Time (min)", fontsize=14, weight='bold')
ax.set_ylabel(f"Depth ({depth_units})", fontsize=14, weight='bold')
ax.set_title("Isoremoval Curves", fontsize=16, weight='bold')
ax.set_ylim(plot_max_depth,0)
ax.grid(True, linestyle='--', linewidth=0.5)
plt.subplots_adjust(bottom=0.25)
handles,labels = ax.get_legend_handles_labels()
legend = ax.legend(
    handles,labels,
    title='Percent Removal',
    loc='upper center',
    bbox_to_anchor=(0.5, -0.25),
    ncol=6,fontsize=8, title_fontsize=10,frameon=True
)
legend.get_title().set_weight('bold')
legend.get_frame().set_facecolor('#f9f9f9')
legend.get_frame().set_edgecolor('gray')
legend.get_frame().set_linewidth(1.5)
legend.get_frame().set_boxstyle('round,pad=0.3,rounding_size=0.2')
legend.get_frame().set_alpha(0.9)
legend_box = legend.get_frame()
shadow_box = FancyBboxPatch(
    (legend_box.get_x()-0.02, legend_box.get_y()-0.02),
    legend_box.get_width()+0.04,
    legend_box.get_height()+0.04,
    boxstyle='round,pad=0.3,rounding_size=0.2',
    linewidth=0,
    color='gray', alpha=0.2, zorder=0
)
ax.add_patch(shadow_box)

st.pyplot(fig)

# Show Interpolated Depths Table
st.subheader("Interpolated Depths Table")
st.write("Depth (m) for each % removal, times_reference, final extension if needed.")
st.dataframe(interpolated_depths.round(3))

# -------------------------------------------------------------------------
# Subplots
# -------------------------------------------------------------------------
st.subheader("Isoremoval Subplots")
n_sub = len(percent_removal_reference)
n_cols=4
n_rows=(n_sub + n_cols -1)//n_cols
fig_sub, axes = plt.subplots(n_rows,n_cols, figsize=(5*n_cols,4*n_rows), constrained_layout=True)
axes = axes.flatten()
for i,(perc,col) in enumerate(zip(percent_removal_reference,colors_)):
    axx=axes[i]
    dd_ = interpolated_depths[perc].values.astype(float)
    tt_ = interpolated_depths.index.values.astype(float)
    mask_ = ~np.isnan(dd_)
    tval=tt_[mask_]
    dval=dd_[mask_]
    if len(tval)<2:
        axx.set_title(f"{perc}% (no data)")
        axx.invert_yaxis()
        axx.grid(True,linestyle='--',linewidth=0.5)
        continue
    sidx=np.argsort(tval)
    tval=tval[sidx]
    dval=dval[sidx]
    #ext
    tval2,dval2=extend_curve_to_bottom(tval,dval,plot_max_depth)
    axx.plot(tval2,dval2,marker='o',color=col,linewidth=1.5,markersize=3,label=f"{perc}%")
    axx.invert_yaxis()
    axx.set_xlabel("Time(min)",fontsize=10,weight='bold')
    axx.set_ylabel(f"Depth({depth_units})",fontsize=10,weight='bold')
    axx.set_title(f"{perc}% Removal",fontsize=12,weight='bold')
    axx.grid(True,linestyle='--',linewidth=0.5)
    axx.legend(fontsize=8)

for j in range(i+1,len(axes)):
    axes[j].axis('off')

fig_sub.suptitle("Isoremoval Curves - Subplots",fontsize=16,weight='bold')
st.pyplot(fig_sub)

# -------------------------------------------------------------------------
# 4. "Second Interpolation" for Overall_Removal_%
# -------------------------------------------------------------------------
st.header("Suspended Solids Removal vs. Detention Time & Overflow Rate")

st.markdown("""
Now we do a second interpolation for every other curve at the exact time 
a given curve intersects the bottom, ensuring we can compute Overall_Removal_% 
via a vertical line from 0..max_depth.
""")

def find_time_bottom(perc):
    """
    We do time-vs-depth interpolation from the (already extended) curve, 
    then solve for depth=plot_max_depth.
    """
    depth_series = interpolated_depths[perc].dropna()
    if depth_series.empty:
        return None
    dvals=depth_series.values
    tvals=depth_series.index.values.astype(float)
    idx_=np.argsort(dvals)
    d_s = dvals[idx_]
    t_s = tvals[idx_]
    if plot_max_depth<d_s[0]:
        if len(d_s)<2: return None
        d1,d2 = d_s[0], d_s[1]
        t1,t2 = t_s[0], t_s[1]
        if abs(d2-d1)<1e-12: return None
        slope=(t2-t1)/(d2-d1)
        cand_t = t1 + slope*(plot_max_depth - d1)
        return cand_t if cand_t>=0 else None
    if plot_max_depth>d_s[-1]:
        if len(d_s)<2: return None
        d1,d2=d_s[-2], d_s[-1]
        t1,t2=t_s[-2],t_s[-1]
        if abs(d2-d1)<1e-12: return None
        slope=(t2-t1)/(d2-d1)
        cand_t=t2+slope*(plot_max_depth-d2)
        return cand_t if cand_t>=0 else None
    ixx=np.searchsorted(d_s,plot_max_depth)
    if ixx==0: return float(t_s[0])
    if ixx>=len(d_s):return float(t_s[-1])
    d_lo,d_hi=d_s[ixx-1],d_s[ixx]
    t_lo,t_hi=t_s[ixx-1],t_s[ixx]
    if abs(d_hi-d_lo)<1e-12:
        return float(t_lo)
    slope=(t_hi-t_lo)/(d_hi-d_lo)
    cand_t=t_lo+slope*(plot_max_depth-d_lo)
    return cand_t if cand_t>=0 else None

def depth_of_curve_at_time(perc, tval):
    """
    "Second interpolation": We gather removal vs. depth at time tval by scanning
    all depths, then invert. We'll do the same logic as building a row in 'interpolated_depths'
    but on the fly.
    """
    # Evaluate removal at each depth from 'removal_df'
    r_, d_ = [], []
    for dd in removal_df.index:
        # time->removal
        rr = interp_time_removal[dd](tval)
        r_.append(rr)
        d_.append(dd)
    r_ = np.array(r_)
    d_ = np.array(d_)

    # filter 0..100
    mask_ = (r_>=0)&(r_<=100)
    r_ok = r_[mask_]
    d_ok = d_[mask_]
    if len(r_ok)<2:
        return np.nan
    idx_s=np.argsort(r_ok)
    r_s=r_ok[idx_s]
    d_s=d_ok[idx_s]
    if perc<r_s[0] or perc>r_s[-1]:
        return np.nan
    ixx=np.searchsorted(r_s,perc)
    if ixx==0: return d_s[0]
    if ixx>=len(r_s): return d_s[-1]
    r_lo,r_hi=r_s[ixx-1], r_s[ixx]
    dd_lo,dd_hi=d_s[ixx-1], d_s[ixx]
    if abs(r_hi-r_lo)<1e-12:
        return dd_lo
    slope=(dd_hi-dd_lo)/(r_hi-r_lo)
    cand_depth=dd_lo + slope*(perc-r_lo)
    cand_depth=np.clip(cand_depth,0,plot_max_depth)
    return cand_depth

def compute_vertical_removal_fraction_on_the_fly(tval):
    """
    We'll gather each curve's depth at time tval using 'depth_of_curve_at_time',
    then do the piecewise vertical integration from top to bottom.
    We'll also optionally add top(=100%@depth=0) and bottom(=0%@depth=max_depth).
    """
    pairs=[]
    # add top, bottom if you want
    pairs.append((100.0, 0.0)) 
    pairs.append((0.0, plot_max_depth))

    for R in percent_removal_reference:
        dd_ = depth_of_curve_at_time(R, tval)
        if np.isnan(dd_):
            continue
        if dd_>=0 and dd_<=plot_max_depth:
            pairs.append((R, dd_))

    # Now sort by depth ascending
    pairs_sorted = sorted(pairs, key=lambda x: x[1])
    totalH=plot_max_depth
    if len(pairs_sorted)<2:
        return np.nan
    # Use the formula: R_total = R_a + sum( (h_i/H)*(R_b - R_a) )
    R_total=0.0
    prevR, prevD = pairs_sorted[0]
    # We'll accumulate the "band" increments
    for i in range(1,len(pairs_sorted)):
        curR, curD = pairs_sorted[i]
        h_i = curD - prevD
        if h_i<0:
            continue
        # the "increment" is (h_i/H)*(curR - prevR)
        R_total += (h_i/totalH)*(curR - prevR)
        prevR, prevD = curR, curD

    # final add the initial curve's R
    R_total += pairs_sorted[0][0]

    # clamp
    R_total = np.clip(R_total, 0, 100)
    return R_total

results_list=[]
for R in percent_removal_reference:
    # find time
    t_int = find_time_bottom(R)
    if t_int is None or t_int<=1e-9:
        continue
    # compute overflow rate
    v_o = (plot_max_depth/t_int)*1440.0
    # now do the second interpolation for overall removal
    R_total = compute_vertical_removal_fraction_on_the_fly(t_int)
    t_h = t_int/60.0
    results_list.append({
        'Isoremoval_Curve_%': R,
        'Time_Intersect_Bottom_min': t_int,
        'Detention_Time_h': t_h,
        'Overflow_Rate_m_d': v_o,
        'Overall_Removal_%': R_total
    })

if not results_list:
    st.warning("No isoremoval curves intersect bottom or no valid data.")
else:
    final_df = pd.DataFrame(results_list).sort_values('Detention_Time_h').reset_index(drop=True)
    st.subheader("Summary of Intersection Times & Computed Removals (Second Interpolation)")
    st.dataframe(final_df.round(2))

    # Plots
    fig_rt, ax_rt = plt.subplots(figsize=(7,5))
    ax_rt.plot(final_df['Detention_Time_h'], final_df['Overall_Removal_%'],
               marker='o',linestyle='-',color='blue')
    ax_rt.set_xlabel("Detention Time (hours)", fontsize=12)
    ax_rt.set_ylabel("Overall Removal (%)", fontsize=12)
    ax_rt.set_title("Suspended Solids Removal vs. Detention Time", fontsize=14, weight='bold')
    ax_rt.grid(True)
    st.pyplot(fig_rt)

    fig_vo, ax_vo = plt.subplots(figsize=(7,5))
    ax_vo.plot(final_df['Overflow_Rate_m_d'], final_df['Overall_Removal_%'],
               marker='s',linestyle='--',color='red')
    ax_vo.set_xlabel("Overflow Rate (m/d)", fontsize=12)
    ax_vo.set_ylabel("Overall Removal (%)", fontsize=12)
    ax_vo.set_title("Suspended Solids Removal vs. Overflow Rate", fontsize=14, weight='bold')
    ax_vo.grid(True)
    st.pyplot(fig_vo)

    st.markdown("""
    **Note**: We do a second interpolation (`depth_of_curve_at_time`) for each curve 
    at the exact time the selected curve hits bottom, ensuring we can form a "vertical line" 
    from top(100%) to bottom(0%) and integrate. 
    That is why we now see Overall_Removal_% for all curves that truly intersect the bottom.
    """)

st.success("Done! You can adjust inputs or upload different data as needed.")