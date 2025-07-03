import matplotlib.pyplot as plt
import re
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# Moving average smoothing function
def moving_average(data, window_size=3):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


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


# Extract AoI from min_aoi_stats.txt or ddpg_stats.txt and compress to 50 points
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
history_files = ['history_gpt4.1.txt', 'history_deepseekv3.txt', 'history_grok3.txt']
min_aoi_file = 'min_aoi_stats.txt'
ddpg_file = 'ddpg_stats.txt'

# Extract data
aoi_data = {}
for file in history_files:
    aoi_data[file] = extract_aoi_history(file)
aoi_data['min_aoi'] = extract_aoi_min(min_aoi_file)
aoi_data['ddpg'] = extract_aoi_min(ddpg_file)

# Check for empty data
if not all(len(data) > 0 for data in aoi_data.values()):
    print("Error: Data extraction failed for one or more files, cannot plot")
    exit()

# Create epochs (0 to 49, for 50 data points)
epochs = list(range(50))

# Moving average window size
window_size = 3
# Adjust epochs for smoothed data (due to reduced points from moving average)
smoothed_epochs = list(range(window_size - 1, 50))

# Plotting
fig, ax_main = plt.subplots(figsize=(10, 6))

# Plot smoothed AoI trends for GPT-4.1, DeepSeek V3, Grok 3, and Genetic Algorithm
model_names = {
    'history_gpt4.1.txt': 'GPT-4.1',
    'history_deepseekv3.txt': 'DeepSeek V3',
    'history_grok3.txt': 'Grok 3',
    'min_aoi': 'Genetic Algorithm',
    'ddpg': 'DDPG'
}
for label, aoi_values in aoi_data.items():
    if label == 'ddpg':  # Skip DDPG for main plot
        continue
    # Apply moving average
    smoothed_aoi = moving_average(aoi_values, window_size=window_size)
    # Plot smoothed curve
    ax_main.plot(smoothed_epochs, smoothed_aoi, marker='o', label=f'{model_names[label]} (Smoothed)')

# Add horizontal line at AoI = 59.7803 in main plot
ax_main.axhline(y=59.7803, color='r', linestyle='--', label='Optimal AoI')

# Create inset plot for DDPG
# Position: [x, y, width, height] in normalized figure coordinates (0 to 1)
# Centered in the middle, adjust size as needed
ax_inset = inset_axes(ax_main, width="40%", height="40%", loc='right')

# 在子图中绘制 DDPG，减小数据点大小
ddpg_values = aoi_data['ddpg']
smoothed_ddpg = moving_average(ddpg_values, window_size=window_size)
ax_inset.plot(smoothed_epochs, smoothed_ddpg, marker='o', markersize=3, linewidth=0.8,
              label=f'{model_names["ddpg"]} (Smoothed)', color='purple')

# 在子图中添加 AoI = 59.7803 的水平线
ax_inset.axhline(y=59.7803, color='r', linestyle='--', label='Optimal AoI')

ax_inset.grid(True, linestyle='--', alpha=0.5)  # 细网格线，增加美观度
ax_inset.set_xlabel('Epoch', fontsize=8, labelpad=2)
ax_inset.set_ylabel('AoI', fontsize=8, labelpad=2)

ax_inset.legend(fontsize=8)
ax_inset.set_xticks(range(0, 52, 10))  # 减少刻度以保持清晰

# 添加子图边框和背景
ax_inset.set_facecolor('white')  # 白色背景
ax_inset.patch.set_edgecolor('black')  # 黑色边框
ax_inset.patch.set_linewidth(0.75)  # 边框线宽
ax_inset.patch.set_alpha(0.9)  # 轻微透明
ax_inset.set_zorder(10)  # 确保子图在主图之上

# 动态设置子图的 y 轴范围，防止数据被裁剪
if len(smoothed_ddpg) > 0:
    ax_inset.set_ylim(min(smoothed_ddpg) * 0.95, max(smoothed_ddpg) * 1.05)

# 调整子图的刻度标签大小
ax_inset.tick_params(axis='both', which='major', labelsize=7)
# Customize main plot
ax_main.set_xlabel('Epoch')
ax_main.set_ylabel('AoI')
ax_main.grid(True)
ax_main.legend()
ax_main.set_xticks(range(0, 52, 2))  # Show every 2nd epoch
plt.tight_layout()

# Save plot
plt.savefig('optimal_aoi_trends_with_ddpg_inset.png', dpi=500, bbox_inches='tight')

# Show plot
plt.show()
