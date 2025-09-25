import matplotlib.pyplot as plt
import re
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Extract AoI from history_*.txt files (250 data points, compress to 50 by averaging every 5 points)
def extract_aoi_history(file_path):
    aoi_values = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    # Extract AoI value from line format: "..., AoI = XX.XXXX"
                    parts = line.strip().split(', ')
                    for part in parts:
                        if part.startswith('AoI = '):
                            aoi = float(part.split(' = ')[1])
                            aoi_values.append(aoi)
                            break
                except (IndexError, ValueError):
                    print(f"Warning: Invalid format in {file_path}: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return []
    
    # Check if we have 250 data points
    if len(aoi_values) != 250:
        print(f"Warning: {file_path} contains {len(aoi_values)} data points, expected 250")
    
    # Compress 250 points to 50 by averaging every 5 points
    if len(aoi_values) == 250:
        aoi_values = np.array(aoi_values).reshape(-1, 5).mean(axis=1)
    
    if len(aoi_values) != 50:
        print(f"Warning: After compression, {file_path} contains {len(aoi_values)} data points, expected 50")
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
history_files = ['history_gpt4.1.txt', 'history_gemini2.5pro.txt', 'history_grok3.txt']
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

# Set global font sizes for better readability when scaled down
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 18
})

# Plot AoI trends for GPT-4.1, Gemini-2.5-Flash, and Grok 3 in main plot
model_names = {
    'history_gpt4.1.txt': 'GPT-4.1',
    'history_gemini2.5pro.txt': 'Gemini-2.5-Pro',
    'history_grok3.txt': 'Grok 3',
    'avg_aoi': 'Genetic Algorithm',
    'ddpg': 'DDPG'
}
for label, aoi_values in aoi_data.items():
    if label in ['avg_aoi', 'ddpg']:  # Skip GA average and DDPG for main plot
        continue
    # Plot raw data
    ax_main.plot(epochs, aoi_values, marker='o', label=model_names[label])

# Add horizontal line at AoI = 59.7803 in main plot
ax_main.axhline(y=34.0125, color='r', linestyle='--', label='Optimal AoI')

# Create inset plot for DDPG and GA average
ax_inset = inset_axes(ax_main, width="50%", height="50%", loc='right')

# Plot DDPG and GA average in inset
for label in ['ddpg', 'avg_aoi']:
    aoi_values = aoi_data[label]
    ax_inset.plot(epochs, aoi_values, marker='o', markersize=3, linewidth=0.8,
                  label=model_names[label], color='purple' if label == 'ddpg' else 'green')

# Add horizontal line at AoI = 34.0125 in inset
ax_inset.axhline(y=34.0125, color='r', linestyle='--', label='Optimal AoI')

# Customize inset plot
ax_inset.grid(True, linestyle='--', alpha=0.5)
ax_inset.set_xlabel('Epoch', fontsize=12, labelpad=2)
ax_inset.set_ylabel('AoI', fontsize=12, labelpad=2)
ax_inset.legend(fontsize=12)
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
ax_inset.tick_params(axis='both', which='major', labelsize=12)

# Customize main plot
ax_main.set_xlabel('Epoch', fontsize=14)
ax_main.set_ylabel('AoI', fontsize=14)
ax_main.grid(True)
ax_main.legend()
# Adjust x-axis ticks for better readability with 50 epochs
ax_main.set_xticks(range(0, 51, 5))
ax_main.set_xlim(-1, 50)
# Set tick label size for main plot to match global settings
ax_main.tick_params(axis='both', which='major', labelsize=14)

# Dynamically set y-axis range for main plot to better show LLM data
main_values = []
for label, aoi_values in aoi_data.items():
    if label not in ['avg_aoi', 'ddpg']:  # Only LLM data for main plot
        main_values.extend(aoi_values)
if main_values:
    ax_main.set_ylim(min(main_values) * 0.95, max(main_values) * 1.05)
plt.tight_layout()

# Save plot
plt.savefig('optimal_aoi_trends_with_ddpg_ga_avg_inset.png', dpi=600, bbox_inches='tight')

# Show plot
plt.show()
