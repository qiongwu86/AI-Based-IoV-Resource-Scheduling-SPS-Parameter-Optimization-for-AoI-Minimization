import matplotlib.pyplot as plt
import re
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Extract AoI from avg_aoi_per_epoch_*.txt files (CSV format with 50 data points)
def extract_aoi_history(file_path):
    aoi_values = []
    try:
        with open(file_path, 'r') as f:
            next(f)  # Skip header (Epoch,Avg_AoI)
            for line in f:
                try:
                    # Split by comma, take second value (Avg_AoI)
                    aoi = float(line.strip().split(',')[1])
                    aoi_values.append(aoi)
                except (IndexError, ValueError):
                    print(f"Warning: Invalid format in {file_path}: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return []
    if len(aoi_values) != 50:
        print(f"Warning: {file_path} contains {len(aoi_values)} data points, expected 50")
    return aoi_values

# Extract AoI from avg_aoi_stats.txt or ddpg_stats.txt and compress to 50 points
def extract_aoi_stats(file_path):
    aoi_values = []
    try:
        with open(file_path, 'r') as f:
            next(f)  # Skip header
            for line in f:
                try:
                    # Split by comma, take second value (Min_AoI or Avg_AoI)
                    aoi = float(line.strip().split(',')[1])
                    aoi_values.append(aoi)
                except (IndexError, ValueError):
                    print(f"Warning: Invalid format in {file_path}: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return []

    # Check if we have 150 data points
    if len(aoi_values) != 150:
        print(f"Warning: {file_path} contains {len(aoi_values)} data points, expected 150")

    # Compress 150 points to 50 by averaging every 3 points
    if len(aoi_values) == 150:
        aoi_values = np.array(aoi_values).reshape(-1, 3).mean(axis=1)

    if len(aoi_values) != 50:
        print(f"Warning: After compression, {file_path} contains {len(aoi_values)} data points, expected 50")
    return aoi_values

# File paths
history_files = ['avg_aoi_per_epoch_gpt4.1.txt', 'avg_aoi_per_epoch_gemini-2.5-flash.txt', 'avg_aoi_per_epoch_grok3.txt']
avg_aoi_file = 'avg_aoi_stats.txt'
ddpg_file = 'ddpg_stats.txt'

# Extract data
aoi_data = {}
for file in history_files:
    aoi_data[file] = extract_aoi_history(file)
aoi_data['avg_aoi'] = extract_aoi_stats(avg_aoi_file)
aoi_data['ddpg'] = extract_aoi_stats(ddpg_file)

# Check for empty data
if not all(len(data) > 0 for data in aoi_data.values()):
    print("Error: Data extraction failed for one or more files, cannot plot")
    exit()

# Create epochs (0 to 49, for 50 data points)
epochs = list(range(50))

# Plotting
fig, ax_main = plt.subplots(figsize=(10, 6))

# Plot AoI trends for GPT-4.1, Gemini-2.5-Flash, and Grok 3 in main plot
model_names = {
    'avg_aoi_per_epoch_gpt4.1.txt': 'GPT-4.1',
    'avg_aoi_per_epoch_gemini-2.5-flash.txt': 'Gemini-2.5-Flash',
    'avg_aoi_per_epoch_grok3.txt': 'Grok 3',
    'avg_aoi': 'Genetic Algorithm',
    'ddpg': 'DDPG'
}
for label, aoi_values in aoi_data.items():
    if label in ['avg_aoi', 'ddpg']:  # Skip GA average and DDPG for main plot
        continue
    # Plot raw data
    ax_main.plot(epochs, aoi_values, marker='o', label=model_names[label])

# Add horizontal line at AoI = 59.7803 in main plot
ax_main.axhline(y=59.7803, color='r', linestyle='--', label='Optimal AoI')

# Create inset plot for DDPG and GA average
ax_inset = inset_axes(ax_main, width="50%", height="50%", loc='right')

# Plot DDPG and GA average in inset
for label in ['ddpg', 'avg_aoi']:
    aoi_values = aoi_data[label]
    ax_inset.plot(epochs, aoi_values, marker='o', markersize=3, linewidth=0.8,
                  label=model_names[label], color='purple' if label == 'ddpg' else 'green')

# Add horizontal line at AoI = 59.7803 in inset
ax_inset.axhline(y=59.7803, color='r', linestyle='--', label='Optimal AoI')

# Customize inset plot
ax_inset.grid(True, linestyle='--', alpha=0.5)
ax_inset.set_xlabel('Epoch', fontsize=8, labelpad=2)
ax_inset.set_ylabel('AoI', fontsize=8, labelpad=2)
ax_inset.legend(fontsize=8)
ax_inset.set_xticks(range(0, 51, 10))

# Adjust inset appearance
ax_inset.set_facecolor('white')
ax_inset.patch.set_edgecolor('black')
ax_inset.patch.set_linewidth(0.75)
ax_inset.patch.set_alpha(0.9)
ax_inset.set_zorder(10)

# Dynamically set y-axis range for inset to prevent clipping
inset_values = []
for label in ['ddpg', 'avg_aoi']:
    inset_values.extend(aoi_data[label])
if inset_values:
    ax_inset.set_ylim(min(inset_values) * 0.95, max(inset_values) * 1.05)

# Adjust inset tick label size
ax_inset.tick_params(axis='both', which='major', labelsize=7)

# Customize main plot
ax_main.set_xlabel('Epoch')
ax_main.set_ylabel('AoI')
ax_main.grid(True)
ax_main.legend()
ax_main.set_xticks(range(0, 51, 2))
plt.tight_layout()

# Save plot
plt.savefig('optimal_aoi_trends_with_ddpg_ga_avg_inset.png', dpi=600, bbox_inches='tight')

# Show plot
plt.show()
