import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Data
components = ["Ray Cluster Init", "Metadata Init", "Create Worker & Load Model Weights", 
               "Profile Memory Usage", "Init KV Cache"]

# Times for each component for 25G model and 140G model
times_25g = np.array([2, 1, 12, 3, 1])
times_140g = np.array([2, 1, 37, 6, 1])

# Colors for components (green for first 3, yellow for last 2)
colors = ['#7EA6E0', '#7EA6E0', '#7EA6E0', '#FFD966', '#FFD966']

# Patterns for each component
patterns = ['||', '//', 'xx', 'x', '--']

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 5))

# Create stacked horizontal bars
y_pos = [0, 0.3]  # 0 for 25G, 1 for 140G
bar_height = 0.2

# First model (25G)
left = 0
for i, time in enumerate(times_25g):
    ax.barh(y_pos[0], time, height=bar_height, left=left, 
            color='white', edgecolor=colors[i], hatch=patterns[i], linewidth=1.5)
    left += time

# Second model (140G)
left = 0
for i, time in enumerate(times_140g):
    ax.barh(y_pos[1], time, height=bar_height, left=left, 
            color='white', edgecolor=colors[i], hatch=patterns[i], linewidth=1.5)
    left += time

# Add grid lines
ax.grid(axis='x', linestyle='--', alpha=0.7)

# Labels and title
ax.set_yticks(y_pos)
ax.set_yticklabels(['13B', '70B'], fontsize=12)
ax.set_xlabel('Time (seconds)', fontsize=12)
# ax.set_title('LLM Serving Engine Initialization Time Breakdown', fontsize=14)

# Create legend patches
legend_elements = [
    Patch(facecolor='white', edgecolor=colors[0],  label='don\'t require full GPU memory: '),
    Patch(facecolor='white', edgecolor=colors[0], hatch=patterns[0], label=components[0]),
    Patch(facecolor='white', edgecolor=colors[1], hatch=patterns[1], label=components[1]),
    Patch(facecolor='white', edgecolor=colors[2], hatch=patterns[2], label=components[2]),
    Patch(facecolor='white', edgecolor=colors[3],  label='require full GPU memory:  '),
    Patch(facecolor='white', edgecolor=colors[3], hatch=patterns[3], label=components[3]),
    Patch(facecolor='white', edgecolor=colors[4], hatch=patterns[4], label=components[4])
]

# Add legend
legend = ax.legend(handles=legend_elements, loc='upper center', 
                   bbox_to_anchor=(0.5, 1.5), ncol=2, frameon=False, 
                   fontsize=14)

# Get the text elements of the legend and make headers bold
texts = legend.get_texts()
texts[0].set_fontweight('bold')  # Make first header bold
texts[4].set_fontweight('bold')  # Make second header bold

# # Add text at the end of bars showing total time
# ax.text(sum(times_25g) + 1, y_pos[0], f'Total: {sum(times_25g)}s',
#        va='center', ha='left', fontsize=10)
# ax.text(sum(times_140g) + 1, y_pos[1], f'Total: {sum(times_140g)}s',
#        va='center', ha='left', fontsize=10)

# Set proper limits
ax.set_xlim(0, max(sum(times_25g), sum(times_140g)) * 1.1)

plt.tight_layout()
plt.savefig('llm_init_time_breakdown.png', dpi=300, bbox_inches='tight')
plt.savefig('llm_init_time_breakdown.eps', dpi=300, bbox_inches='tight')
plt.show()