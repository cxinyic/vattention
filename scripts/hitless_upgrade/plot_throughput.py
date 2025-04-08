import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read both CSV files
# df_baseline = pd.read_csv("logs/multi_model_upgrade/multi_model_upgrade/model_0_Yi-Coder-1.5B/bs_30/gpu_0_1_to_0_1_2_3/throughput_metrics.csv")
df_inplace = pd.read_csv("logs/multi_model_upgrade_overlap/bs_40/gpu_0_1_2_3_to_0_1_2_3/overlap/throughput_metrics.csv")
# df_baseline = pd.read_csv("logs/multi_model_upgrade_overlap/bs_60/gpu_0_1_to_0_1_2_3/no_serve/throughput_metrics.csv")
df_baseline = df_inplace
# Function to calculate throughput directly from consecutive points
def calculate_point_throughput(df, min_time_diff=1.1, max_time_diff=3.1, max_throughput=None, gap_interval=0.5):
    """
    Calculate throughput directly from consecutive data points.
    
    Args:
        df: DataFrame with elapsed_time and tokens_generated columns
        min_time_diff: Minimum time difference to consider (skip points closer than this)
        max_time_diff: Maximum time gap before filling with zeros
        max_throughput: Optional cap on maximum throughput value
        gap_interval: Interval to insert zero points in large gaps
    
    Returns:
        DataFrame with time and throughput columns
    """
    # Sort data by elapsed_time to ensure proper ordering
    df = df.sort_values('elapsed_time').reset_index(drop=True)
    
    # Initialize lists to store results
    times = []
    throughputs = []
    
    # Calculate throughput between consecutive points
    i = 0
    while i < len(df) - 1:
        prev_time = df['elapsed_time'].iloc[i]
        prev_tokens = df['tokens_generated'].iloc[i]
        
        # Find the next point that is at least min_time_diff away
        next_idx = i + 1
        while next_idx < len(df) and (df['elapsed_time'].iloc[next_idx] - prev_time) < min_time_diff:
            next_idx += 1
        
        if next_idx < len(df):
            curr_time = df['elapsed_time'].iloc[next_idx]
            curr_tokens = df['tokens_generated'].iloc[next_idx]
            
            time_diff = curr_time - prev_time
            token_diff = curr_tokens - prev_tokens
            
            # Check if there's a large gap that needs filling with zeros
            if time_diff > max_time_diff:
                # Calculate throughput for the initial segment
                if token_diff > 0:
                    initial_throughput = token_diff / min_time_diff if min_time_diff > 0 else 0
                    if max_throughput is not None:
                        initial_throughput = min(initial_throughput, max_throughput)
                else:
                    initial_throughput = 0
                
                # Add the initial throughput point
                times.append(prev_time + min_time_diff/2)
                throughputs.append(initial_throughput)
                
                # Fill the gap with zeros at regular intervals
                gap_start = prev_time + gap_interval
                while gap_start < curr_time:
                    times.append(gap_start)
                    throughputs.append(0)
                    gap_start += gap_interval
            else:
                # Normal case - calculate throughput
                mid_time = (prev_time + curr_time) / 2
                
                if token_diff > 0:
                    throughput = token_diff / time_diff
                    if max_throughput is not None:
                        throughput = min(throughput, max_throughput)
                else:
                    throughput = 0
                
                times.append(mid_time)
                throughputs.append(throughput)
            
            # Move to the next valid point
            i = next_idx
        else:
            # No more points beyond min_time_diff
            break
    
    # Create a DataFrame with the results
    result_df = pd.DataFrame({
        'time': times,
        'throughput': throughputs
    })
    
    return result_df

# Filter the dataframes to only include data before 100 seconds
df_baseline_filtered = df_baseline[df_baseline['elapsed_time'] < 100]
df_inplace_filtered = df_inplace[df_inplace['elapsed_time'] < 100]

# Calculate throughput with the point-by-point approach
# Using 95th percentile as the cap for maximum throughput
throughput_baseline_raw = calculate_point_throughput(df_baseline_filtered, min_time_diff=1.1, max_time_diff=3.1, gap_interval=0.5)
throughput_inplace_raw = calculate_point_throughput(df_inplace_filtered, min_time_diff=1.1, max_time_diff=3.1, gap_interval=0.5)

# Calculate 95th percentile for each dataset to cap extreme values
baseline_cap = throughput_baseline_raw['throughput'].quantile(0.95)
inplace_cap = throughput_inplace_raw['throughput'].quantile(0.95)

# Recalculate with the caps
throughput_baseline_df = calculate_point_throughput(df_baseline_filtered, min_time_diff=1.1, max_time_diff=3.1, 
                                                  max_throughput=baseline_cap, gap_interval=0.5)
throughput_inplace_df = calculate_point_throughput(df_inplace_filtered, min_time_diff=1.1, max_time_diff=3.1, 
                                                 max_throughput=inplace_cap, gap_interval=0.5)

# Add zero throughput values from time 0 to the first data point
# First, get the first time point in each dataset
first_baseline_time = throughput_baseline_df['time'].min()
first_inplace_time = throughput_inplace_df['time'].min()

# Create dataframes with zero throughput from time 0 to first data point (with 1-second intervals)
if first_baseline_time > 0:
    baseline_zeros = pd.DataFrame({
        'time': np.arange(0, first_baseline_time, 1.0),
        'throughput': np.zeros(len(np.arange(0, first_baseline_time, 1.0)))
    })
    throughput_baseline_df = pd.concat([baseline_zeros, throughput_baseline_df]).reset_index(drop=True)

if first_inplace_time > 0:
    inplace_zeros = pd.DataFrame({
        'time': np.arange(0, first_inplace_time, 1.0),
        'throughput': np.zeros(len(np.arange(0, first_inplace_time, 1.0)))
    })
    throughput_inplace_df = pd.concat([inplace_zeros, throughput_inplace_df]).reset_index(drop=True)

# Apply a small amount of smoothing to make the plot more readable
# Using a simple rolling mean with a small window
throughput_baseline_df['smoothed'] = throughput_baseline_df['throughput'].rolling(window=3, center=True).mean().fillna(throughput_baseline_df['throughput'])
throughput_inplace_df['smoothed'] = throughput_inplace_df['throughput'].rolling(window=3, center=True).mean().fillna(throughput_inplace_df['throughput'])

# Set larger font sizes for all plots
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 18,  # Larger x and y labels
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20
})

# Plot baseline throughput as a separate figure
plt.figure(figsize=(12, 6))
plt.plot(throughput_baseline_df['time'], throughput_baseline_df['smoothed'], 
         linestyle='-', linewidth=2, color='orange', alpha=0.8)

# plt.title('Baseline Throughput')
plt.xlabel('Time (seconds)')
plt.ylabel('Throughput (tokens/second)')
plt.grid(True, alpha=0.3, linestyle='--')
plt.xlim(0, 100)  # Explicitly set x-axis limit to 100 seconds

# Get the y-value at x=40 for the baseline data
y_at_30 = throughput_baseline_df.loc[throughput_baseline_df['time'].sub(39.5).abs().idxmin(), 'smoothed']

# Add an arrow at time=40 directly on the line
plt.annotate('Upgrade starts', xy=(39.5, y_at_30), xytext=(39.5, y_at_30 * 1.3),
             arrowprops=dict(facecolor='gray', shrink=0.05, width=2, headwidth=8, 
                             edgecolor='gray'),  
             fontsize=14, color='gray', ha='center', weight='bold')

plt.tight_layout()
plt.savefig('throughput_baseline.pdf', dpi=300)

# Plot in-place throughput as a separate figure
plt.figure(figsize=(12, 6))
plt.plot(throughput_inplace_df['time'], throughput_inplace_df['smoothed'], 
         linestyle='-', linewidth=2, color='green', alpha=0.8)

# plt.title('In-place Throughput')
plt.xlabel('Time (seconds)')
plt.ylabel('Throughput (tokens/second)')
plt.grid(True, alpha=0.3, linestyle='--')
plt.xlim(0, 100)  # Explicitly set x-axis limit to 100 seconds

# Get the y-value at x=40 for the in-place data
y_at_30 = throughput_inplace_df.loc[throughput_inplace_df['time'].sub(40).abs().idxmin(), 'smoothed']

# Add an arrow at time=40 directly on the line
plt.annotate('Upgrade starts', xy=(40, y_at_30), xytext=(40, y_at_30 * 1.3),
             arrowprops=dict(facecolor='gray', shrink=0.05, width=2, headwidth=8, 
                             edgecolor='gray'), 
             fontsize=14, color='gray', ha='center', weight='bold')

plt.tight_layout()
plt.savefig('throughput_inplace.pdf', dpi=300)

# FIGURE: Baseline total tokens comparison (before 80s)
plt.figure(figsize=(12, 6))
plt.plot(df_baseline_filtered['elapsed_time'], df_baseline_filtered['tokens_generated'], 
         marker='o', linestyle='-', linewidth=2, color='orange')
plt.title('Baseline Total Tokens Generated (Before 80s) - Yi-Coder-1.5B (GPU 0,1 → 0,1,2,3)')
plt.xlabel('Elapsed Time (seconds)')
plt.ylabel('Total Tokens Generated')
plt.grid(True, alpha=0.3, linestyle='--')
plt.xlim(0, 100)  # Explicitly set x-axis limit to 100 seconds

# Get the y-value at x=40 for the baseline tokens data
closest_point = df_baseline_filtered.loc[df_baseline_filtered['elapsed_time'].sub(40).abs().idxmin()]
x_at_30 = closest_point['elapsed_time']
y_at_30 = closest_point['tokens_generated']

# Add an arrow at time=40 directly on the line
plt.annotate('Upgrade starts', xy=(x_at_30, y_at_30), xytext=(x_at_30, y_at_30 * 1.2),
             arrowprops=dict(facecolor='gray', shrink=0.05, width=2, headwidth=8),
             fontsize=14, color='gray', ha='center')

plt.tight_layout()
plt.savefig('tokens_generated_baseline.pdf', dpi=300)

# FIGURE: In-place total tokens comparison (before 80s)
plt.figure(figsize=(12, 6))
plt.plot(df_inplace_filtered['elapsed_time'], df_inplace_filtered['tokens_generated'], 
         marker='s', linestyle='-', linewidth=2, color='green')
plt.title('In-place Total Tokens Generated (Before 80s) - Yi-Coder-1.5B (GPU 0,1 → 0,1,2,3)')
plt.xlabel('Elapsed Time (seconds)')
plt.ylabel('Total Tokens Generated')
plt.grid(True, alpha=0.3, linestyle='--')
plt.xlim(0, 100)  # Explicitly set x-axis limit to 100 seconds

# Get the y-value at x=40 for the in-place tokens data
closest_point = df_inplace_filtered.loc[df_inplace_filtered['elapsed_time'].sub(40).abs().idxmin()]
x_at_30 = closest_point['elapsed_time']
y_at_30 = closest_point['tokens_generated']

# Add an arrow at time=40 directly on the line
plt.annotate('Upgrade starts', xy=(x_at_30, y_at_30), xytext=(x_at_30, y_at_30 * 1.2),
             arrowprops=dict(facecolor='gray', shrink=0.05, width=2, headwidth=8),
             fontsize=14, color='gray', ha='center')

plt.tight_layout()
plt.savefig('tokens_generated_inplace.pdf', dpi=300)

plt.show()

print("Baseline throughput (before 80s) saved as 'throughput_baseline_80s.pdf'")
print("In-place throughput (before 80s) saved as 'throughput_inplace_80s.pdf'")
print("Baseline tokens generated (before 80s) saved as 'tokens_generated_baseline_80s.pdf'")
print("In-place tokens generated (before 80s) saved as 'tokens_generated_inplace_80s.pdf'")