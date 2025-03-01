import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import matplotlib.lines as mlines  # For custom legend entries

# Step 1: Get all the files that contain 'violin_plot_data' and '3D_last' in their filenames
file_pattern = '*violin_plot_data*3Dlast*.csv'
file_list = glob.glob(file_pattern)

# Ensure we have the expected number of files
assert len(file_list) == 3, f"Expected 3 files, but found {len(file_list)}."

# Step 2: Combine all datasets into one DataFrame
all_data = pd.DataFrame()

for file in file_list:
    # Load the data
    data = pd.read_csv(file)

    # Extract the filename (without path and extension) to use as label
    filename = os.path.basename(file).replace('_', ' ').replace('.csv', '')

    # Remove 'violin plot data' and '3D' from the label
    filename_clean = filename.replace('violin plot data', '').replace('3D', '').strip()

    # Remove 'last' from the label
    data['Dataset'] = filename_clean.replace('last', '').strip()

    # Append the current file's data to the combined DataFrame
    all_data = pd.concat([all_data, data], ignore_index=True)

# Step 3: Create a single violin plot with all the datasets
plt.figure(figsize=(10, 8))

sns.violinplot(x='Dataset', y='Value', hue='Type', data=all_data, palette="Set2", split=True, inner="quart", gap=0.05)

# Step 4: Customize the plot
plt.title('Full MC Data vs Surrogate Estimates (Three-input Case)', fontsize=16)
plt.ylabel('Flood Extent ($m^2$)', fontsize=14)

# Increase the size of the x-axis title ('Dataset')
plt.xlabel('Surrogate Model', fontsize=16)  # Enlarging the "Dataset" label itself

# Increase the size of the x-axis tick labels
plt.xticks(rotation=45, ha='right', fontsize=12)

plt.grid(True, linestyle='--', alpha=0.6)

# Step 5: Add custom legend for mean and quartile lines

# Create a custom line for the mean and quartile
mean_line = mlines.Line2D([], [], color='black', linestyle='--', label='Median')
quartile_line = mlines.Line2D([], [], color='black', linestyle=':', label='Interquartile')

# Generate legend entries for True MC Data and Surrogate Estimate
handles, labels = plt.gca().get_legend_handles_labels()

# Append mean and quartile to the existing legend
handles.extend([mean_line, quartile_line])

# Move the legend outside the plot
plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), title='Legend')

# Step 6: Save the figure
plt.tight_layout()
plt.savefig('comparison_violin_plot_3D_last.png', dpi=300)  # Save as PNG with 300 dpi resolution

# Show the plot
plt.show()
