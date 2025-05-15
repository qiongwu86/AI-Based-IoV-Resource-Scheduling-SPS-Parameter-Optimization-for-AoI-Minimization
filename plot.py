import matplotlib.pyplot as plt
import re
import numpy as np

# Moving average smoothing function
def moving_average(data, window_size=3):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Extract AoI from history*.txt files
def extract_aoi_history(file_path):
    aoi_values = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Match AoI values, accommodating varying decimal places (e.g., 60.0663 or 62.9241)
                match = re.search(r'AoI = (\d+\.\d+)', line)
                if match:
                    aoi_values.append(float(match.group(1)))
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return []
    if len(aoi_values) != 50:
        print(f"Warning: {file_path} contains {len(aoi_values)} data points, expected 50")
    return aoi_values

# Extract AoI from min_aoi_stats.txt
def extract_aoi_min(file_path):
    aoi_values = []
    try:
        with open(file_path, 'r') as f:
            next(f)  # Skip header
            for line in f:
                try:
                    # Split by comma, take second value (Min_AoI)
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

# File paths
history_files = ['history_gpt4.1.txt', 'history_deepseekv3.txt', 'history_grok3.txt']
min_aoi_file = 'min_aoi_stats.txt'

# Extract data
aoi_data = {}
for file in history_files:
    aoi_data[file] = extract_aoi_history(file)
aoi_data['min_aoi'] = extract_aoi_min(min_aoi_file)

# Check for empty data
if not all(aoi_data.values()):
    print("Error: Data extraction failed for one or more files, cannot plot")
    exit()

# Create epochs (0 to 49, for 50 data points)
epochs = list(range(50))

# Moving average window size
window_size = 3
# Adjust epochs for smoothed data (due to reduced points from moving average)
smoothed_epochs = list(range(window_size-1, 50))

# Plotting
plt.figure(figsize=(10, 6))

# Plot smoothed AoI trends for each file
model_names = {
    'history_gpt4.1.txt': 'GPT-4.1',
    'history_deepseekv3.txt': 'DeepSeek V3',
    'history_grok3.txt': 'Grok 3',
    'min_aoi': 'Genetic Algorithm'
}
for label, aoi_values in aoi_data.items():
    # Apply moving average
    smoothed_aoi = moving_average(aoi_values, window_size=window_size)
    # Plot smoothed curve
    plt.plot(smoothed_epochs, smoothed_aoi, marker='o', label=f'{model_names[label]} (Smoothed)')

# Add horizontal line at AoI = 59.7803
plt.axhline(y=59.7803, color='r', linestyle='--', label='Optimal AoI')

# Customize plot
plt.xlabel('Epoch')
plt.ylabel('AoI')
# plt.title('Smoothed AoI Trends')
plt.grid(True)
plt.legend()
plt.xticks(range(0, 52, 2))  # Show every 2nd epoch
plt.tight_layout()

# Save plot
plt.savefig('optimal_aoi_trends.png', dpi=600, bbox_inches='tight')

# Show plot
plt.show()
