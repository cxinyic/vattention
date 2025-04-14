import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_metric_cdf(latency_files: dict, metric: str, output_path: str, title: str):
    """
    Plot CDF for a specific latency metric from multiple files.
    
    Args:
        latency_files: Dict mapping strategy names to csv file paths
        metric: Which metric to plot ('e2e_latency_seconds', 'ttft_seconds', or 'tpot_seconds')
        output_path: Path to save the plot
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    
    for label, file_path in latency_files.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if metric in df.columns:
                # When reading the CSV file, pandas should automatically convert empty strings to NaN
                # But let's make sure by explicitly checking for them
                df[metric] = pd.to_numeric(df[metric], errors='coerce')  # This will convert empty strings and non-numeric values to NaN
                filtered_df = df.dropna(subset=[metric])
                
                if len(filtered_df) > 0:
                    values = sorted(filtered_df[metric].values)
                    n = len(values)
                    cumulative_prob = np.arange(1, n + 1) / n
                    
                    line = plt.plot(values, cumulative_prob, '-', linewidth=2, label=label)[0]
                    color = line.get_color()
                    
                    # Add percentile lines and print statistics
                    percentiles = [50, 90, 95, 99]
                    print(f"\nStatistics for {label} - {metric}:")
                    for i, p in enumerate(percentiles):
                        percentile_value = np.percentile(values, p)
                        plt.axvline(x=percentile_value, color=color, linestyle='--', alpha=0.2)
                        print(f"p{p}: {percentile_value:.2f}s")
                    
                    # Print mean and count
                    print(f"Mean: {filtered_df[metric].mean():.2f}s")
                    print(f"Count: {len(filtered_df)} (of {len(df)} total)")
                    
                    # If data was filtered, print how many points were dropped
                    if len(filtered_df) < len(df):
                        print(f"Note: {len(df) - len(filtered_df)} data points were filtered out due to missing {metric} values")
                else:
                    print(f"Warning: No valid data for {metric} in {file_path} after filtering out missing values")
            else:
                print(f"Warning: Metric {metric} not found in {file_path}")
        else:
            print(f"Warning: File not found - {file_path}")
    
    # Customize x-axis label based on metric
    if metric == "e2e_latency_seconds":
        plt.xlabel('E2E Latency (seconds)', fontsize=20)  # Adjust the number as needed
    elif metric == "ttft_seconds":
        plt.xlabel('TTFT (seconds)', fontsize=20)
    elif metric == "tpot_seconds":
        plt.xlabel('Time per Output Token (seconds)', fontsize=20)
    else:
        plt.xlabel(f'{metric} (seconds)', fontsize=20)
        
    plt.ylabel('CDF', fontsize=20)
    # plt.title(title)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=16,loc='lower right')
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the plot
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    latency_files = {
        "Baseline": "logs/hitless_upgrade/bs_100/trace/no_serve/kickout_immediately/by_arrival_time/by_arrival_time/latencies.csv",
        "Overlap with FCFS": "logs/hitless_upgrade/bs_100/trace/decode_only/kickout_immediately/by_arrival_time/by_arrival_time/latencies.csv",
        "Overlap with Upgrade Scheduler": "logs/hitless_upgrade/bs_100/trace/prefill_only/kickout_immediately/by_arrival_time/by_prefill_status/latencies.csv",
    }
    
    # Create output directory
    output_dir = "logs/hitless_upgrade/bs_32/gpu_2_to_4/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate separate plots for each metric
    plot_metric_cdf(
        latency_files,
        metric="e2e_latency_seconds",
        output_path=f"{output_dir}/e2e_latency_cdf.pdf",
        title="End-to-End Latency CDF Comparison"
    )
    
    plot_metric_cdf(
        latency_files,
        metric="ttft_seconds",
        output_path=f"{output_dir}/ttft_cdf.pdf",
        title="Time to First Token CDF Comparison"
    )
    
    plot_metric_cdf(
        latency_files,
        metric="tpot_seconds",
        output_path=f"{output_dir}/tpot_cdf.pdf",
        title="Time per Output Token CDF Comparison"
    )

if __name__ == "__main__":
    main()