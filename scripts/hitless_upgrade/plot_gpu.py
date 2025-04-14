import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot_utilization_metrics(baseline_file, swiftserve_file, output_prefix=None, 
                            start_idx=10, end_idx=140, interval=0.5):
    """
    Plot GPU and memory utilization from two CSV files and compare them.
    
    Args:
        baseline_file: Path to the baseline CSV file
        swiftserve_file: Path to the SwiftServe CSV file
        output_prefix: Optional prefix for saving the plots
        start_idx: Starting data point index to plot
        end_idx: Ending data point index to plot
        interval: Time interval between data points in seconds
    """
    # Read data from CSV files
    try:
        baseline_data = pd.read_csv(baseline_file)
        swiftserve_data = pd.read_csv(swiftserve_file)
        
        print(f"Successfully loaded data files:")
        print(f"  - Baseline: {baseline_file} ({len(baseline_data)} samples)")
        print(f"  - SwiftServe: {swiftserve_file} ({len(swiftserve_data)} samples)")
    except Exception as e:
        print(f"Error loading data files: {e}")
        return
    
    # Check if the memory utilization column exists
    has_memory_data = 'avg_memory_utilization' in baseline_data.columns and 'avg_memory_utilization' in swiftserve_data.columns
    
    if not has_memory_data:
        print("Warning: Memory utilization data not found in files. Only plotting GPU utilization.")
    
    # Slice the data to get only the specified range
    if start_idx < len(baseline_data) and end_idx <= len(baseline_data):
        baseline_data = baseline_data.iloc[start_idx:end_idx]
    else:
        print(f"Warning: Requested range ({start_idx}:{end_idx}) exceeds baseline data length ({len(baseline_data)})")
        baseline_data = baseline_data.iloc[min(start_idx, len(baseline_data)-1):min(end_idx, len(baseline_data))]
        
    if start_idx < len(swiftserve_data) and end_idx <= len(swiftserve_data):
        swiftserve_data = swiftserve_data.iloc[start_idx:end_idx]
    else:
        print(f"Warning: Requested range ({start_idx}:{end_idx}) exceeds SwiftServe data length ({len(swiftserve_data)})")
        swiftserve_data = swiftserve_data.iloc[min(start_idx, len(swiftserve_data)-1):min(end_idx, len(swiftserve_data))]
    
    # Create time indices with the specified interval
    baseline_data['time'] = np.arange(len(baseline_data)) * interval
    swiftserve_data['time'] = np.arange(len(swiftserve_data)) * interval
    
    # Plot GPU utilization
    plot_single_metric(
        baseline_data, 
        swiftserve_data, 
        'avg_gpu_utilization', 
        'GPU Utilization Comparison: Baseline vs SwiftServe',
        'Average GPU Utilization (%)',
        output_prefix + '_gpu_utilization.png' if output_prefix else None
    )
    
    # Plot memory utilization if available
    if has_memory_data:
        plot_single_metric(
            baseline_data, 
            swiftserve_data, 
            'avg_memory_utilization', 
            'Memory Utilization Comparison: Baseline vs SwiftServe',
            'Average Memory Utilization (%)',
            output_prefix + '_memory_utilization.png' if output_prefix else None
        )
    else:
        print("Skipping memory utilization plot due to missing data.")

def plot_single_metric(baseline_data, swiftserve_data, metric_column, title, ylabel, output_file=None):
    """
    Create a plot for a single metric comparing baseline vs swiftserve.
    
    Args:
        baseline_data: DataFrame with baseline data
        swiftserve_data: DataFrame with swiftserve data
        metric_column: Column name of the metric to plot
        title: Plot title
        ylabel: Y-axis label
        output_file: Optional file path to save the plot
    """
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot the data
    plt.plot(baseline_data['time'], baseline_data[metric_column], 
             label='Baseline', color='#1f77b4', linewidth=2)
    plt.plot(swiftserve_data['time'], swiftserve_data[metric_column], 
             label='SwiftServe', color='#ff7f0e', linewidth=2)
    
    # Add horizontal lines for the means
    baseline_mean = baseline_data[metric_column].mean()
    swiftserve_mean = swiftserve_data[metric_column].mean()
    plt.axhline(y=baseline_mean, color='#1f77b4', linestyle='--', alpha=0.7)
    plt.axhline(y=swiftserve_mean, color='#ff7f0e', linestyle='--', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Add legend
    plt.legend(fontsize=12)
    
    # Compute statistics
    baseline_stats = {
        'mean': baseline_mean,
        'median': baseline_data[metric_column].median(),
        'min': baseline_data[metric_column].min(),
        'max': baseline_data[metric_column].max(),
        'std': baseline_data[metric_column].std()
    }
    
    swiftserve_stats = {
        'mean': swiftserve_mean,
        'median': swiftserve_data[metric_column].median(),
        'min': swiftserve_data[metric_column].min(),
        'max': swiftserve_data[metric_column].max(),
        'std': swiftserve_data[metric_column].std()
    }
    
    # Display statistics on the plot
    stats_text = (
        f"Baseline - Mean: {baseline_stats['mean']:.2f}%, "
        f"Max: {baseline_stats['max']:.2f}%\n"
        f"SwiftServe - Mean: {swiftserve_stats['mean']:.2f}%, "
        f"Max: {swiftserve_stats['max']:.2f}%"
    )
    plt.annotate(stats_text, xy=(0.02, 0.02), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                 fontsize=10, va='bottom')
    
    # Enhance visual appearance
    plt.tight_layout()
    
    # Save the plot if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    
    # Print detailed statistics
    print(f"\nDetailed Statistics for {metric_column}:")
    print("-" * 50)
    print(f"Baseline:")
    for key, value in baseline_stats.items():
        print(f"  {key}: {value:.2f}")
    
    print(f"\nSwiftServe:")
    for key, value in swiftserve_stats.items():
        print(f"  {key}: {value:.2f}")
    
    # Calculate improvement
    if baseline_stats['mean'] > 0:
        util_improvement = ((swiftserve_stats['mean'] / baseline_stats['mean']) - 1) * 100
        print(f"\nImprovement: {util_improvement:.2f}%")
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot GPU and memory utilization from two CSV files")
    parser.add_argument("baseline_file", help="Path to baseline CSV file")
    parser.add_argument("swiftserve_file", help="Path to SwiftServe CSV file")
    parser.add_argument("--output", "-o", help="Prefix for saved output plots (optional)")
    parser.add_argument("--start", type=int, default=10, help="Starting data point index")
    parser.add_argument("--end", type=int, default=130, help="Ending data point index")
    parser.add_argument("--interval", type=float, default=0.5, help="Time interval between data points (seconds)")
    
    args = parser.parse_args()
    
    plot_utilization_metrics(
        args.baseline_file, 
        args.swiftserve_file, 
        args.output,
        args.start,
        args.end,
        args.interval
    )