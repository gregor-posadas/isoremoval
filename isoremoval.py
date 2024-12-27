import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.patches import FancyBboxPatch
import base64
from io import BytesIO

# Set the page configuration
st.set_page_config(
    page_title="Isoremoval Curves",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title of the app
st.title("Isoremoval Curves Generator")

st.markdown("""
This application allows you to generate Isoremoval Curves based on your own data inputs. 
You can specify your own depths, times, concentrations, and initial concentration by uploading a properly formatted Excel file.
""")

# Sidebar for user inputs
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
    # Define the data
    data = {
        'Depth (m)': [0.5, 1.0, 1.5, 2.0, 2.5],
        10: [14, 15, 15.4, 16, 17],
        20: [10, 13, 14.2, 14.6, 15],
        35: [7, 10.6, 12, 12.6, 13],
        50: [6.2, 8.2, 10, 11, 11.4],
        70: [5, 7, 7.8, 9, 10],
        85: [4, 6, 7, 8, 8.8]
    }

    # Create DataFrame
    df = pd.DataFrame(data)
    columns = ['Depth (m)', 10, 20, 35, 50, 70, 85]
    df = df[columns]

    # Convert DataFrame to Excel in memory
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

if uploaded_file is not None:
    depths, times, concentrations = parse_excel(uploaded_file)
    if depths is None or times is None or concentrations is None:
        st.stop()
else:
    st.sidebar.info("Please upload an Excel file to proceed.")
    st.stop()

# Validate inputs
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

# Main processing
st.header("Generated Isoremoval Curves")

# Disable LaTeX rendering
plt.rcParams['text.usetex'] = False

# Compute percent removals
percent_removals = (initial_concentration - concentrations) / initial_concentration * 100
removal_df = pd.DataFrame(percent_removals, index=depths, columns=times)

##########################################################################
# 1. Display Table of Removal Efficiencies for Each Time and Depth
##########################################################################
st.subheader("Percent Removal as a Function of Time and Depth")

# We will label rows as "Depth (user-selected units)" and columns as "Time (min)"
removal_df_display = removal_df.copy()
removal_df_display.index.name = f"Depth ({depth_units})"
removal_df_display.columns.name = "Time (min)"

# Limit to 2 decimal places
removal_df_display = removal_df_display.round(2)

st.dataframe(removal_df_display)

##########################################################################
# 2. User-Specified (or Default) Percent Removal Curves to Plot
##########################################################################
st.sidebar.subheader("Select Which % Removal Curves to Plot")

default_curves = "10,20,30,40,50,60,70,80"
user_curves_input = st.sidebar.text_input(
    "Enter comma-separated % values (between 1 and 99):", 
    value=default_curves
)

# Parse the user input and fall back to default if parsing fails or empty
try:
    user_curves_list = [int(x.strip()) for x in user_curves_input.split(",")]
    # Keep only valid values between 1 and 99
    user_curves_list = [val for val in user_curves_list if 1 <= val < 100]
except:
    user_curves_list = [int(x.strip()) for x in default_curves.split(",")]

if len(user_curves_list) == 0:
    user_curves_list = [int(x.strip()) for x in default_curves.split(",")]

# Sort and remove duplicates
percent_removal_reference = sorted(list(set(user_curves_list)))

##########################################################################
# Interpolation for Isoremoval Curves
##########################################################################
times_reference = np.arange(0, max(times) + 10, step=5)  # Increased resolution for times

# Prepare a DataFrame to hold interpolated depths
interpolated_depths = pd.DataFrame(index=times_reference, columns=percent_removal_reference)

# Create interpolation function for each depth (over time)
interp_funcs_over_time = {}
for depth in removal_df.index:
    interp_funcs_over_time[depth] = interp1d(
        removal_df.columns, 
        removal_df.loc[depth], 
        kind='linear', 
        fill_value='extrapolate'
    )

# For each percent removal in user-specified list, find depth vs. time curve
for percent in percent_removal_reference:
    depths_list = []
    for time_val in times_reference:
        if time_val == 0:
            # At time=0, we can define the "depth" as 0 for all removal (or NaN).
            # For a typical scenario, no removal has occurred at time=0, so depth=0.
            interpolated_depth = 0
        else:
            # Evaluate the removal function at this time for all depths
            depths_for_time = np.array([interp_funcs_over_time[d](time_val) for d in removal_df.index])
            # We only consider values that are between 0% and 100% removal
            valid_mask = (depths_for_time >= 0) & (depths_for_time <= 100)
            valid_depths = removal_df.index[valid_mask]
            valid_percent_removals = depths_for_time[valid_mask]

            if len(valid_percent_removals) > 1:
                interp_func_over_depth = interp1d(
                    valid_percent_removals, 
                    valid_depths, 
                    kind='linear', 
                    bounds_error=False, 
                    fill_value='extrapolate'
                )
                interpolated_depth = interp_func_over_depth(percent)
            else:
                interpolated_depth = np.nan

        # Discard if outside the plot range or NaN
        if np.isnan(interpolated_depth) or interpolated_depth < 0 or interpolated_depth > plot_max_depth:
            interpolated_depth = np.nan

        depths_list.append(interpolated_depth)

    interpolated_depths[percent] = depths_list

# Clean up infinite or invalid values
interpolated_depths = interpolated_depths.apply(pd.to_numeric, errors='coerce')
interpolated_depths.replace([np.inf, -np.inf], np.nan, inplace=True)

##########################################################################
# Plotting the Isoremoval Curves
##########################################################################
fig, ax = plt.subplots(figsize=(14, 10))

# High-contrast colormap with sufficient distinct colors
cmap = plt.get_cmap('tab20')  # 'tab20' has 20 distinct colors
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

# Set plot labels, title, and grid
ax.set_xlabel('Time (min)', fontsize=14, weight='bold')
ax.set_ylabel(f'Depth ({depth_units})', fontsize=14, weight='bold')
ax.set_title('Isoremoval Curves', fontsize=16, weight='bold')
ax.set_ylim(plot_max_depth, 0)  # Invert y-axis to have depth increasing downward
ax.grid(color='gray', linestyle='--', linewidth=0.5)

# Adjust the bottom margin to accommodate the legend
plt.subplots_adjust(bottom=0.25)

# Legend - Spread horizontally with multiple columns
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

# Add a shadow effect to the legend
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

##########################################################################
# 3. Display the Interpolated Depths Table (Limited to 2 decimals)
##########################################################################
st.subheader("Interpolated Depths Table")
st.write("Each cell represents the depth at which a specific percent removal occurs at a given time.")

# Round the table for better readability (limit to 2 decimals)
interpolated_depths_display = interpolated_depths.round(2)
interpolated_depths_display.index.name = "Time (min)"
st.dataframe(interpolated_depths_display)

##########################################################################
# Additional Subplots
##########################################################################
st.subheader("Isoremoval Subplots for Each Percent Removal")

# Define the figure and subplots
n_subplots = len(percent_removal_reference)
n_cols = 4  # Adjust columns as desired
n_rows = (n_subplots + n_cols - 1) // n_cols  # Ceiling division

fig_sub, axes = plt.subplots(
    nrows=n_rows,
    ncols=n_cols,
    figsize=(5 * n_cols, 4 * n_rows),
    constrained_layout=True
)

# Flatten axes for easier indexing
axes = axes.flatten()

for idx, percent in enumerate(percent_removal_reference):
    ax_sub = axes[idx]
    times_with_origin = interpolated_depths.index
    depths_with_origin = interpolated_depths[percent].values.astype(float)

    mask = (~np.isnan(depths_with_origin)) & (depths_with_origin >= 0)
    ax_sub.plot(
        times_with_origin[mask],
        depths_with_origin[mask],
        label=f'{percent:.0f}% Removal',
        color=cmap(idx / len(percent_removal_reference)),
        linewidth=1.5,
        marker='o',
        markersize=3
    )
    ax_sub.set_title(f'{percent:.0f}% Removal', fontsize=12, weight='bold')
    ax_sub.set_xlabel('Time (min)', fontsize=10, weight='bold')
    ax_sub.set_ylabel(f'Depth ({depth_units})', fontsize=10, weight='bold')
    ax_sub.invert_yaxis()  # Depth increases downward
    ax_sub.grid(color='gray', linestyle='--', linewidth=0.5)
    ax_sub.legend(fontsize=8, frameon=True)

# Turn off unused axes if subplots don't fill the grid
for ax in axes[n_subplots:]:
    ax.axis('off')

# Add a super title for the entire plot
fig_sub.suptitle('Isoremoval Curves', fontsize=16, weight='bold')
st.pyplot(fig_sub)
