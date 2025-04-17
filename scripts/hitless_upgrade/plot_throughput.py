import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read both CSV files
df_baseline = pd.read_csv("logs/hitless_upgrade/bs_100/gpu_2_4/throughput/no_serve/kickout_immediately/by_arrival_time/by_arrival_time/throughput_metrics.csv")
df_swiftserve = pd.read_csv("logs/hitless_upgrade/bs_100/gpu_2_4/throughput/decode_only/kickout_immediately/by_arrival_time/by_prefill_status/throughput_metrics.csv")

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

# Filter the dataframes to only include data before 300 seconds
df_baseline_filtered = df_baseline[df_baseline['elapsed_time'] < 300]
df_swiftserve_filtered = df_swiftserve[df_swiftserve['elapsed_time'] < 300]

# Calculate throughput with the point-by-point approach
# Using 95th percentile as the cap for maximum throughput
throughput_baseline_raw = calculate_point_throughput(df_baseline_filtered, min_time_diff=1.1, max_time_diff=3.1, gap_interval=0.5)
throughput_swiftserve_raw = calculate_point_throughput(df_swiftserve_filtered, min_time_diff=1.1, max_time_diff=3.1, gap_interval=0.5)

# Calculate 95th percentile for each dataset to cap extreme values
baseline_cap = throughput_baseline_raw['throughput'].quantile(0.95)
swiftserve_cap = throughput_swiftserve_raw['throughput'].quantile(0.95)

# Recalculate with the caps
throughput_baseline_df = calculate_point_throughput(df_baseline_filtered, min_time_diff=1.1, max_time_diff=3.1, 
                                                  max_throughput=baseline_cap, gap_interval=0.5)
throughput_swiftserve_df = calculate_point_throughput(df_swiftserve_filtered, min_time_diff=1.1, max_time_diff=3.1, 
                                                 max_throughput=swiftserve_cap, gap_interval=0.5)

# Add zero throughput values from time 0 to the first data point
# First, get the first time point in each dataset
first_baseline_time = throughput_baseline_df['time'].min()
first_swiftserve_time = throughput_swiftserve_df['time'].min()

# Create dataframes with zero throughput from time 0 to first data point (with 1-second intervals)
if first_baseline_time > 0:
    baseline_zeros = pd.DataFrame({
        'time': np.arange(0, first_baseline_time, 1.0),
        'throughput': np.zeros(len(np.arange(0, first_baseline_time, 1.0)))
    })
    throughput_baseline_df = pd.concat([baseline_zeros, throughput_baseline_df]).reset_index(drop=True)

if first_swiftserve_time > 0:
    swiftserve_zeros = pd.DataFrame({
        'time': np.arange(0, first_swiftserve_time, 1.0),
        'throughput': np.zeros(len(np.arange(0, first_swiftserve_time, 1.0)))
    })
    throughput_swiftserve_df = pd.concat([swiftserve_zeros, throughput_swiftserve_df]).reset_index(drop=True)

# Apply a small amount of smoothing to make the plot more readable
# Using a simple rolling mean with a small window
throughput_baseline_df['smoothed'] = throughput_baseline_df['throughput'].rolling(window=3, center=True).mean().fillna(throughput_baseline_df['throughput'])
throughput_swiftserve_df['smoothed'] = throughput_swiftserve_df['throughput'].rolling(window=3, center=True).mean().fillna(throughput_swiftserve_df['throughput'])

# Set larger font sizes for all plots
plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 20,  # Larger x and y labels
    'axes.titlesize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 20
})

# Create combined throughput plot with both lines
plt.figure(figsize=(12, 9))

# Plot baseline in orange (original color from your code)
plt.plot(throughput_baseline_df['time'], throughput_baseline_df['smoothed'], 
         linestyle='-', linewidth=2, color='orange', alpha=0.8, label='Terminate and Restart')

# Plot SwiftServe in green (original color from your code)
plt.plot(throughput_swiftserve_df['time'], throughput_swiftserve_df['smoothed'], 
         linestyle='-', linewidth=2, color='green', alpha=0.8, label='SwiftServe')

plt.xlabel('Time (seconds)')
plt.ylabel('Throughput (tokens/second)')
plt.grid(True, alpha=0.3, linestyle='--')
plt.xlim(0, 300)  # Explicitly set x-axis limit to 300 seconds
plt.legend(loc='lower right')

# Get the y-values at x=68 for adding annotations
y_baseline_68 = throughput_baseline_df.loc[throughput_baseline_df['time'].sub(68).abs().idxmin(), 'smoothed']
y_swiftserve_68 = throughput_swiftserve_df.loc[throughput_swiftserve_df['time'].sub(68).abs().idxmin(), 'smoothed']

# Determine the maximum value of both lines at the specific x=68 position
max_y_at_68 = max(y_baseline_68, y_swiftserve_68)

# Set a constant vertical offset for the arrow text (30 tokens/second higher than the data point)
const_throughput_offset = 300
arrow_y_position_68 = max_y_at_68 + const_throughput_offset

# Add an arrow at time=68 pointing downward with shorter height
plt.annotate('Upgrade Starts', 
             xy=(68, max_y_at_68 + 5),  # Position 5 tokens/second above the higher line
             xytext=(68, arrow_y_position_68),  # Position text at constant height above the point
             arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8, edgecolor='black'),  
             fontsize=14, color='black', ha='center', weight='bold')

# Get the y-values at x=110 for adding "Upgrade Finished" annotations
y_baseline_112 = throughput_baseline_df.loc[throughput_baseline_df['time'].sub(110).abs().idxmin(), 'smoothed']
y_swiftserve_112 = throughput_swiftserve_df.loc[throughput_swiftserve_df['time'].sub(110).abs().idxmin(), 'smoothed']

# Determine the maximum value of both lines at the specific x=110 position
max_y_at_112 = max(y_baseline_112, y_swiftserve_112)

# Use the same constant vertical offset for the arrow text (30 tokens/second higher than the data point)
arrow_y_position_112 = max_y_at_112 + const_throughput_offset

# Add an arrow at time=110 pointing downward with shorter height
plt.annotate('Upgrade Finishes', 
             xy=(110, max_y_at_112 + 5),  # Position 5 tokens/second above the higher line
             xytext=(110, arrow_y_position_112),  # Position text at constant height above the point
             arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8, edgecolor='black'),  
             fontsize=14, color='black', ha='center', weight='bold')

plt.tight_layout()

# Save the combined figure
plt.savefig('combined_throughput_comparison.pdf', dpi=300)
plt.savefig('combined_throughput_comparison.png', dpi=300)

# Optional: Create combined tokens generated plot
plt.figure(figsize=(12, 6))

# Plot baseline in orange (original color from your code)
plt.plot(df_baseline_filtered['elapsed_time'], df_baseline_filtered['tokens_generated'], 
         marker='o', linestyle='-', linewidth=2, color='orange', label='Terminate and Restart')

# Plot SwiftServe in green (original color from your code)
plt.plot(df_swiftserve_filtered['elapsed_time'], df_swiftserve_filtered['tokens_generated'], 
         marker='s', linestyle='-', linewidth=2, color='green', label='SwiftServe')

plt.title('Total Tokens Generated Comparison')
plt.xlabel('Elapsed Time (seconds)')
plt.ylabel('Total Tokens Generated')
plt.grid(True, alpha=0.3, linestyle='--')
plt.xlim(0, 300)  # Explicitly set x-axis limit to 300 seconds
plt.legend(loc='lower right')

# Get the y-values at x=68 for adding annotations
closest_baseline_68 = df_baseline_filtered.loc[df_baseline_filtered['elapsed_time'].sub(68).abs().idxmin()]
closest_swiftserve_68 = df_swiftserve_filtered.loc[df_swiftserve_filtered['elapsed_time'].sub(68).abs().idxmin()]
y_baseline_68 = closest_baseline_68['tokens_generated']
y_swiftserve_68 = closest_swiftserve_68['tokens_generated']

# Instead of using a percentage of the maximum value, use a constant offset
# Determine the maximum value of both lines at the specific x=68 position
max_y_at_68 = max(y_baseline_68, y_swiftserve_68)

# Set a constant vertical offset for the arrow text (300 tokens higher than the data point)
const_vertical_offset = 300
arrow_tokens_position = max_y_at_68 + const_vertical_offset

# Add an arrow at time=68 pointing downward with shorter height
plt.annotate('Upgrade starts', 
             xy=(68, max_y_at_68 + 50),  # Position 50 tokens above the higher line
             xytext=(68, arrow_tokens_position),  # Position text at constant height above the point
             arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8, edgecolor='black'),
             fontsize=14, color='black', ha='center', weight='bold')

# Get the y-values at x=110 for adding "Upgrade Finished" annotations
closest_baseline_112 = df_baseline_filtered.loc[df_baseline_filtered['elapsed_time'].sub(110).abs().idxmin()]
closest_swiftserve_112 = df_swiftserve_filtered.loc[df_swiftserve_filtered['elapsed_time'].sub(110).abs().idxmin()]
y_baseline_112 = closest_baseline_112['tokens_generated']
y_swiftserve_112 = closest_swiftserve_112['tokens_generated']

# Determine the maximum value of both lines at the specific x=110 position
max_y_at_112 = max(y_baseline_112, y_swiftserve_112)

# Use the same constant vertical offset for the arrow text (300 tokens higher than the data point)
arrow_tokens_position_112 = max_y_at_112 + const_vertical_offset

# Add an arrow at time=110 pointing downward with shorter height
plt.annotate('Upgrade Finished', 
             xy=(110, max_y_at_112 + 50),  # Position 50 tokens above the higher line
             xytext=(110, arrow_tokens_position_112),  # Position text at constant height above the point
             arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8, edgecolor='black'),
             fontsize=14, color='black', ha='center', weight='bold')

plt.tight_layout()

# Save the combined tokens figure
plt.savefig('combined_tokens_comparison.pdf', dpi=300)
plt.savefig('combined_tokens_comparison.png', dpi=300)

plt.show()

print("Combined throughput comparison saved as 'combined_throughput_comparison.pdf/png'")
print("Combined tokens generated comparison saved as 'combined_tokens_comparison.pdf/png'")