import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_combined_metrics(latency_files: dict, output_path: str, dataset_name: str = "Prefill Heavy Gen"):
    """
    Plot TTFT and E2E latency CDFs side by side as subfigures with a shared legend at the top
    and a dataset label on the left.
    
    Args:
        latency_files: Dict mapping strategy names to csv file paths
        output_path: Path to save the plot
        dataset_name: Name of the dataset to display on the left side
    """
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Store lines for the legend
    legend_lines = []
    legend_labels = []
    
    # Plot TTFT on the left subplot
    for label, file_path in latency_files.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            metric = "ttft_seconds"
            
            if metric in df.columns:
                df[metric] = pd.to_numeric(df[metric], errors='coerce')
                filtered_df = df.dropna(subset=[metric])
                
                if len(filtered_df) > 0:
                    values = sorted(filtered_df[metric].values)
                    n = len(values)
                    cumulative_prob = np.arange(1, n + 1) / n
                    
                    line = ax1.plot(values, cumulative_prob, '-', linewidth=2, label=label)[0]
                    color = line.get_color()
                    
                    # Store for combined legend
                    if label not in legend_labels:
                        legend_lines.append(line)
                        legend_labels.append(label)
                    
                    # Add percentile lines
                    percentiles = [50, 90, 95, 99]
                    print(f"\nStatistics for {label} - {metric}:")
                    for p in percentiles:
                        percentile_value = np.percentile(values, p)
                        ax1.axvline(x=percentile_value, color=color, linestyle='--', alpha=0.2)
                        print(f"p{p}: {percentile_value:.2f}s")
                    
                    print(f"Mean: {filtered_df[metric].mean():.2f}s")
                    print(f"Count: {len(filtered_df)} (of {len(df)} total)")
                    
                    if len(filtered_df) < len(df):
                        print(f"Note: {len(df) - len(filtered_df)} data points were filtered out due to missing {metric} values")
                else:
                    print(f"Warning: No valid data for {metric} in {file_path} after filtering out missing values")
            else:
                print(f"Warning: Metric {metric} not found in {file_path}")
        else:
            print(f"Warning: File not found - {file_path}")
    
    # Plot E2E latency on the right subplot
    for label, file_path in latency_files.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            metric = "e2e_latency_seconds"
            
            if metric in df.columns:
                df[metric] = pd.to_numeric(df[metric], errors='coerce')
                filtered_df = df.dropna(subset=[metric])
                
                if len(filtered_df) > 0:
                    values = sorted(filtered_df[metric].values)
                    n = len(values)
                    cumulative_prob = np.arange(1, n + 1) / n
                    
                    line = ax2.plot(values, cumulative_prob, '-', linewidth=2, label=label)[0]
                    color = line.get_color()
                    
                    # Add percentile lines
                    percentiles = [50, 90, 95, 99]
                    print(f"\nStatistics for {label} - {metric}:")
                    for p in percentiles:
                        percentile_value = np.percentile(values, p)
                        ax2.axvline(x=percentile_value, color=color, linestyle='--', alpha=0.2)
                        print(f"p{p}: {percentile_value:.2f}s")
                    
                    print(f"Mean: {filtered_df[metric].mean():.2f}s")
                    print(f"Count: {len(filtered_df)} (of {len(df)} total)")
                    
                    if len(filtered_df) < len(df):
                        print(f"Note: {len(df) - len(filtered_df)} data points were filtered out due to missing {metric} values")
                else:
                    print(f"Warning: No valid data for {metric} in {file_path} after filtering out missing values")
            else:
                print(f"Warning: Metric {metric} not found in {file_path}")
        else:
            print(f"Warning: File not found - {file_path}")
    
    # Customize the left subplot (TTFT)
    ax1.set_xlabel('TTFT (seconds)', fontsize=20)
    ax1.set_ylabel('CDF', fontsize=20)
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Customize the right subplot (E2E latency)
    ax2.set_xlabel('E2E Latency (seconds)', fontsize=20)
    ax2.set_ylabel('CDF', fontsize=20)
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Add a single legend at the top center of the figure
    fig.legend(legend_lines, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1), 
               ncol=3, fontsize=16)  # ncol=3 to make the three labels appear in a row
    
    # Add dataset label on the left side
    fig.text(0.03, 0.5, dataset_name, fontsize=20, rotation=90, va='center', ha='center')
    
    # Adjust the layout to make room for the legend and dataset label
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, left=0.1)  # Make space for the legend at the top and dataset label on the left
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the plot
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    latency_files = {
        "Baseline": "logs/hitless_upgrade/bs_60/gpu_2_4/no_serve/kickout_immediately/by_arrival_time/by_arrival_time/latencies.csv",
        "SwiftServe with FCFS": "logs/hitless_upgrade/bs_60/gpu_2_4/decode_only/kickout_immediately/by_arrival_time/by_arrival_time/latencies.csv",
        "SwiftServe":
        "logs/hitless_upgrade/bs_60/gpu_2_4/decode_only/kickout_immediately/by_arrival_time/by_prefill_status/latencies.csv",
        # "Baseline": "logs/hitless_upgrade/bs_60/trace/no_serve/kickout_immediately/by_arrival_time/by_arrival_time/latencies.csv",
        # "SwiftServe with FCFS": "logs/hitless_upgrade/bs_60/trace/decode_only/kickout_immediately/by_arrival_time/by_arrival_time/latencies.csv",
        # "SwiftServe": "logs/hitless_upgrade/bs_60/trace/prefill_only/kickout_immediately/by_arrival_time/by_prefill_status/latencies.csv",
    }
    
    # Create output directory
    output_dir = "logs/hitless_upgrade/gpu_2_4/bs_120/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate combined plot with TTFT and E2E latency side by side
    plot_combined_metrics(
        latency_files,
        output_path=f"{output_dir}/combined_latency_cdf.pdf",
        dataset_name="ShareGPT"  # You can change this to the actual dataset name
    )

if __name__ == "__main__":
    main()