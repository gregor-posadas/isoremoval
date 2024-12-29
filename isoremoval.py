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
    page_title="Isoremoval Curves (100% top boundary only)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------------
# Title and Intro
# -------------------------------------------------------------------------
st.title("Isoremoval Curves Generator (with 100% Top Boundary, No 0% at Bottom)")

st.markdown("""
1. Reads a table of depths × times → concentrations (mg/L).  
2. Computes isoremoval curves for selected removal percentages.  
3. For **Overall Removal** at the time a curve intersects the bottom, we do a **second interpolation** 
   of **all** curves at that exact time, from the bottom curve to a 100% boundary at the top (depth=0).  
4. We do **not** add an artificial 0% boundary at the bottom.  
""")

# -------------------------------------------------------------------------
# Sidebar: Provide sample file & example
# -------------------------------------------------------------------------
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

st.sidebar.header("Upload Data File")
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
    "Initial Concentration (mg/L)",
    min_value=0.0,
    value=20.0,
    step=0.1
)

depth_units = st.sidebar.selectbox(
    "Depth Units",
    ["Meters (m)","Feet (ft)","Centimeters (cm)","Inches (in)"],
    index=0
)

max_depth_val = st.sidebar.number_input(
    "Max Depth",
    min_value=0.0,
    value=0.0,
    step=0.1,
    help="0 => use maximum depth in data"
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
        st.error("Unsupported format.")
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
    st.error("Shape mismatch in data.")
    st.stop()

if max_depth_val>0:
    plot_max = max_depth_val
    if plot_max<np.max(depths):
        st.warning("User-specified max depth < data's actual max depth.")
else:
    plot_max = np.max(depths)

# -------------------------------------------------------------------------
# Show user data
# -------------------------------------------------------------------------
st.subheader("Your Uploaded Data (Concentrations)")
st.dataframe(df.style.format(precision=2))

# -------------------------------------------------------------------------
# 2. Compute Removals
# -------------------------------------------------------------------------
removals = (init_conc - concentrations)/init_conc*100
removal_df = pd.DataFrame(removals, index=depths, columns=times)
removal_df.index.name = f"Depth ({depth_units})"
removal_df.columns.name = "Time (min)"
st.subheader("Percent Removal (Table)")
st.dataframe(removal_df.round(2))

# -------------------------------------------------------------------------
# 3. Build isoremoval curves
# -------------------------------------------------------------------------
st.sidebar.subheader("Which % Curves to Plot?")
default_str = "10,20,30,40,50,60,70,80"
user_str = st.sidebar.text_input("Comma %", value=default_str)
try:
    user_list = [int(x.strip()) for x in user_str.split(",")]
    user_list = [u for u in user_list if 1<=u<100]
except:
    user_list = [10,20,30,40,50,60,70,80]

percent_list = sorted(list(set(user_list)))
times_ref = np.arange(0, max(times)+10, step=5)
iso_depths = pd.DataFrame(index=times_ref, columns=percent_list)

# time->removal for each depth
interp_time_removal={}
for d_ in removal_df.index:
    t_ = removal_df.columns.values.astype(float)
    r_ = removal_df.loc[d_].values.astype(float)
    interp_time_removal[d_] = interp1d(t_, r_, kind='linear', fill_value="extrapolate")

# invert removal->depth for each time
for perc in percent_list:
    d_list=[]
    for tval in times_ref:
        if tval==0:
            d_list.append(0.0)
            continue
        # Evaluate removal at all depths
        rtemp=[]
        dtemp=[]
        for dd_ in removal_df.index:
            rr = interp_time_removal[dd_](tval)
            rtemp.append(rr)
            dtemp.append(dd_)
        rtemp=np.array(rtemp)
        dtemp=np.array(dtemp)
        mask_=(rtemp>=0)&(rtemp<=100)
        rr_ = rtemp[mask_]
        dd_ = dtemp[mask_]
        if len(rr_)<2:
            d_list.append(np.nan)
            continue
        idx_s = np.argsort(rr_)
        rr_s=rr_[idx_s]
        dd_s=dd_[idx_s]
        if perc<rr_s[0] or perc>rr_s[-1]:
            d_list.append(np.nan)
            continue
        ixx = np.searchsorted(rr_s,perc)
        if ixx==0:
            d_list.append(dd_s[0])
        elif ixx>=len(rr_s):
            d_list.append(dd_s[-1])
        else:
            r_lo,r_hi=rr_s[ixx-1], rr_s[ixx]
            dep_lo,dep_hi=dd_s[ixx-1], dd_s[ixx]
            if abs(r_hi-r_lo)<1e-12:
                d_list.append(dep_lo)
            else:
                slope=(dep_hi-dep_lo)/(r_hi-r_lo)
                d_cand=dep_lo + slope*(perc-r_lo)
                d_cand=np.clip(d_cand,0,plot_max)
                d_list.append(d_cand)
    arr_ = np.array(d_list)
    arr_ = np.clip(arr_,0,plot_max)
    iso_depths[perc] = arr_

def extend_final_segment(tvals,dvals,bottom):
    if len(tvals)<2:
        return tvals,dvals
    if dvals[-1]>=bottom:
        return tvals,dvals
    d2=dvals[-1]
    d1=dvals[-2]
    t2=tvals[-1]
    t1=tvals[-2]
    if abs(t2-t1)<1e-12:
        return tvals,dvals
    slope=(d2-d1)/(t2-t1)
    if abs(slope)<1e-12:
        return tvals,dvals
    t_ext=t2+(bottom-d2)/slope
    if t_ext>t2:
        tv_new=np.append(tvals,t_ext)
        dv_new=np.append(dvals,bottom)
        return tv_new,dv_new
    return tvals,dvals

st.subheader("Generated Isoremoval Curves")
fig, ax = plt.subplots(figsize=(14,10))
cmap_ = plt.get_cmap('tab20')
colors_=cmap_(np.linspace(0,1,len(percent_list)))
for perc,col in zip(percent_list,colors_):
    dcol = iso_depths[perc].values.astype(float)
    tcol = iso_depths.index.values.astype(float)
    mask__=~np.isnan(dcol)
    tt_=tcol[mask__]
    dd_=dcol[mask__]
    if len(tt_)<2:
        continue
    sidx=np.argsort(tt_)
    tt_=tt_[sidx]
    dd_=dd_[sidx]
    # extend
    tt_e,dd_e = extend_final_segment(tt_,dd_, plot_max)
    ax.plot(tt_e,dd_e,color=col,label=f"{perc}%")

ax.set_xlabel("Time(min)",fontsize=14,weight='bold')
ax.set_ylabel(f"Depth({depth_units})",fontsize=14,weight='bold')
ax.set_title("Isoremoval Curves",fontsize=16,weight='bold')
ax.set_ylim(plot_max,0)
ax.grid(True,linestyle='--',linewidth=0.5)
plt.subplots_adjust(bottom=0.25)
handles,labels = ax.get_legend_handles_labels()
legend=ax.legend(
    handles,labels,
    title="Percent Removal",
    loc='upper center',
    bbox_to_anchor=(0.5,-0.25),
    ncol=6,fontsize=8, title_fontsize=10,frameon=True
)
legend.get_title().set_weight('bold')
legend.get_frame().set_facecolor('#f9f9f9')
legend.get_frame().set_edgecolor('gray')
legend.get_frame().set_linewidth(1.5)
legend.get_frame().set_boxstyle('round,pad=0.3,rounding_size=0.2')
legend.get_frame().set_alpha(0.9)
legend_box = legend.get_frame()
shadow_box=FancyBboxPatch(
    (legend_box.get_x()-0.02,legend_box.get_y()-0.02),
    legend_box.get_width()+0.04,
    legend_box.get_height()+0.04,
    boxstyle='round,pad=0.3,rounding_size=0.2',
    linewidth=0,
    color='gray', alpha=0.2,zorder=0
)
ax.add_patch(shadow_box)
st.pyplot(fig)

st.subheader("Interpolated Depths Table")
st.dataframe(iso_depths.round(3))

st.subheader("Isoremoval Subplots")
n_sub=len(percent_list)
n_cols=4
n_rows=(n_sub+n_cols-1)//n_cols
fig_sub, axes_sub = plt.subplots(n_rows,n_cols, figsize=(5*n_cols,4*n_rows), constrained_layout=True)
axes_sub=axes_sub.flatten()
for i,(p_,co_) in enumerate(zip(percent_list, colors_)):
    axx=axes_sub[i]
    dv_=iso_depths[p_].values.astype(float)
    tv_=iso_depths.index.values.astype(float)
    ms_=~np.isnan(dv_)
    tv2=tv_[ms_]
    dv2=dv_[ms_]
    if len(tv2)<2:
        axx.set_title(f"{p_}% (no data)")
        axx.invert_yaxis()
        axx.grid(True,linestyle='--',linewidth=0.5)
        continue
    sidx_=np.argsort(tv2)
    tv2=tv2[sidx_]
    dv2=dv2[sidx_]
    tv_ex,dv_ex = extend_final_segment(tv2,dv2,plot_max)
    axx.plot(tv_ex,dv_ex, marker='o',color=co_,linewidth=1.5,markersize=3,label=f"{p_}%")
    axx.invert_yaxis()
    axx.set_title(f"{p_}% Removal",fontsize=12,weight='bold')
    axx.set_xlabel("Time(min)",fontsize=10,weight='bold')
    axx.set_ylabel(f"Depth({depth_units})",fontsize=10,weight='bold')
    axx.grid(True,linestyle='--',linewidth=0.5)
    axx.legend(fontsize=8)

for j in range(i+1,len(axes_sub)):
    axes_sub[j].axis('off')
fig_sub.suptitle("Isoremoval Curves - Subplots",fontsize=16,weight='bold')
st.pyplot(fig_sub)

# -------------------------------------------------------------------------
# Second interpolation for Overall Removal
# with a 100% top boundary, NO 0% at bottom
# -------------------------------------------------------------------------
st.header("Suspended Solids Removal vs. Detention Time & Overflow Rate (100% top boundary only)")

# 1. Find intersection time with bottom
def find_time_bottom(perc):
    srs=iso_depths[perc].dropna()
    if srs.empty:return None
    dvals=srs.values
    tvals=srs.index.values.astype(float)
    idx_=np.argsort(dvals)
    d_s=dvals[idx_]
    t_s=tvals[idx_]
    # handle if bottom outside range
    if plot_max<d_s[0]:
        if len(d_s)<2:return None
        d1,d2 = d_s[0], d_s[1]
        t1,t2 = t_s[0], t_s[1]
        if abs(d2-d1)<1e-12:return None
        slope=(t2-t1)/(d2-d1)
        cand=t1 + slope*(plot_max-d1)
        return cand if cand>=0 else None
    if plot_max>d_s[-1]:
        if len(d_s)<2:return None
        d1,d2=d_s[-2], d_s[-1]
        t1,t2=t_s[-2],t_s[-1]
        if abs(d2-d1)<1e-12:return None
        slope=(t2-t1)/(d2-d1)
        cand=t2+slope*(plot_max-d2)
        return cand if cand>=0 else None
    ixx=np.searchsorted(d_s,plot_max)
    if ixx==0:return float(t_s[0])
    if ixx>=len(d_s):return float(t_s[-1])
    dl,dh=d_s[ixx-1], d_s[ixx]
    tl,th=t_s[ixx-1], t_s[ixx]
    if abs(dh-dl)<1e-12:
        return float(tl)
    slope=(th-tl)/(dh-dl)
    return tl + slope*(plot_max-dl)

# 2. On-the-fly function: depth_of_curve_at_time
def depth_of_curve_at_time(perc, tval):
    """
    Evaluate for each depth d in removal_df,
    time->removal. Then invert to find depth s.t. removal=perc.
    """
    rlist=[]
    dlist=[]
    for d_ in removal_df.index:
        r_ = interp_time_removal[d_](tval)
        rlist.append(r_)
        dlist.append(d_)
    rlist=np.array(rlist)
    dlist=np.array(dlist)
    mask_=(rlist>=0)&(rlist<=100)
    rr=rlist[mask_]
    dd=dlist[mask_]
    if len(rr)<2:
        return np.nan
    idx_s=np.argsort(rr)
    rr_s=rr[idx_s]
    dd_s=dd[idx_s]
    if perc<rr_s[0] or perc>rr_s[-1]:
        return np.nan
    ixx=np.searchsorted(rr_s,perc)
    if ixx==0:return dd_s[0]
    if ixx>=len(rr_s):return dd_s[-1]
    r_lo,r_hi=rr_s[ixx-1], rr_s[ixx]
    d_lo,d_hi=dd_s[ixx-1], dd_s[ixx]
    if abs(r_hi-r_lo)<1e-12:
        return d_lo
    slope=(d_hi-d_lo)/(r_hi-r_lo)
    cand_depth=d_lo + slope*(perc-r_lo)
    cand_depth=np.clip(cand_depth,0,plot_max)
    return cand_depth

# 3. vertical integration
def compute_overall_removal_top_only(tval):
    """
    We gather each real isoremoval curve at time tval, plus top=100% at depth=0,
    then do a piecewise integration from shallow to deep.
    No 0% boundary at bottom.
    """
    pairs=[]
    # add 100% top boundary
    pairs.append((100.0, 0.0))

    # gather real curves
    for p_ in percent_list:
        dd_ = depth_of_curve_at_time(p_, tval)
        if not np.isnan(dd_) and 0<=dd_<=plot_max:
            pairs.append((p_, dd_))

    # if <2, can't integrate
    if len(pairs)<2:
        return np.nan
    # sort by depth ascending
    pairs_sorted=sorted(pairs, key=lambda x: x[1])
    H=plot_max
    R_total=0.0
    prevR, prevD = pairs_sorted[0]
    for i in range(1,len(pairs_sorted)):
        curR, curD = pairs_sorted[i]
        deltaD=curD-prevD
        if deltaD<0:
            continue
        # piece
        R_total += (deltaD/H)*(curR-prevR)
        prevR, prevD = curR, curD
    # add the initial removal from top boundary
    R_total += pairs_sorted[0][0]
    return np.clip(R_total,0,100)

results=[]
for p_ in percent_list:
    t_bott = find_time_bottom(p_)
    if t_bott is None or t_bott<1e-12:
        continue
    # Overflow
    v_o = (plot_max/t_bott)*1440.0
    # overall removal
    R_tot = compute_overall_removal_top_only(t_bott)
    t_h = t_bott/60.0
    results.append({
        "Isoremoval_Curve_%": p_,
        "Time_Intersect_Bottom_min": round(t_bott,2),
        "Detention_Time_h": round(t_h,2),
        "Overflow_Rate_m_d": round(v_o,2),
        "Overall_Removal_%": round(R_tot,2)
    })

if not results:
    st.warning("No curves intersect bottom or no valid data.")
else:
    final_df = pd.DataFrame(results).sort_values("Detention_Time_h")
    st.subheader("Summary of Intersection Times & Computed Removals (100% top boundary only)")
    st.dataframe(final_df)

    fig_dt, ax_dt = plt.subplots(figsize=(7,5))
    ax_dt.plot(final_df['Detention_Time_h'], final_df['Overall_Removal_%'],
               marker='o', color='blue', linewidth=1.5)
    ax_dt.set_xlabel("Detention Time (hours)", fontsize=12)
    ax_dt.set_ylabel("Overall Removal (%)", fontsize=12)
    ax_dt.set_title("Suspended Solids Removal vs. Detention Time", fontsize=14, weight='bold')
    ax_dt.grid(True)
    st.pyplot(fig_dt)

    fig_vo, ax_vo = plt.subplots(figsize=(7,5))
    ax_vo.plot(final_df['Overflow_Rate_m_d'], final_df['Overall_Removal_%'],
               marker='s', linestyle='--', color='red', linewidth=1.5)
    ax_vo.set_xlabel("Overflow Rate (m/d)", fontsize=12)
    ax_vo.set_ylabel("Overall Removal (%)", fontsize=12)
    ax_vo.set_title("Suspended Solids Removal vs. Overflow Rate", fontsize=14, weight='bold')
    ax_vo.grid(True)
    st.pyplot(fig_vo)

st.success("Done! We keep 100% at top=0m only, no 0% at bottom.")
