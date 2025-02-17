import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_combined_cdf(latency_files: dict, output_path: str):
    """
    Plot combined CDF from multiple latency files.
    
    Args:
        latency_files: Dict mapping strategy names to csv file paths
        output_path: Path to save the combined plot
    """
    plt.figure(figsize=(10, 6))
    
    for label, file_path in latency_files.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            latencies = sorted(df['latency'].values)
            n = len(latencies)
            cumulative_prob = np.arange(1, n + 1) / n
            
            line = plt.plot(latencies, cumulative_prob, '-', linewidth=2, label=label)[0]
            color = line.get_color()
            
            # Add percentile lines and print statistics
            percentiles = [50, 90, 95, 99]
            print(f"\nStatistics for {label}:")
            for i, p in enumerate(percentiles):
                percentile_value = np.percentile(latencies, p)
                plt.axvline(x=percentile_value, color=color, linestyle='--', alpha=0.2)
                y_offset = 0.1 * (i / len(percentiles))
                plt.text(percentile_value, 0.2 + y_offset, 
                        f'{label}\np{p}={percentile_value:.2f}s',
                        rotation=90, verticalalignment='center', fontsize=8)
                print(f"p{p}: {percentile_value:.2f}s")
            
            # Print mean and count
            print(f"Mean: {df['latency'].mean():.2f}s")
            print(f"Count: {len(df)}")
        else:
            print(f"Warning: File not found - {file_path}")
    
    plt.xlabel('Latency (seconds)')
    plt.ylabel('Cumulative Probability')
    plt.title('Request Latency CDF Comparison')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the combined plot
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    latency_files = {
        "No Upgrade": "logs/figure_7/model_01-ai/Yi-Coder-1.5B_bs_32_attn_fa_vattn/no_upgrade/replica_0/no_upgrade/request_latencies.csv",
        "No Serving During Upgrade": "logs/figure_7/model_01-ai/Yi-Coder-1.5B_bs_32_attn_fa_vattn/basic_upgrade/replica_0/basic_upgrade/request_latencies.csv",
        "Overlap Prefill During Upgrade": "logs/figure_7/model_01-ai/Yi-Coder-1.5B_bs_32_attn_fa_vattn/prefill_upgrade/replica_0/no_upgrade/request_latencies.csv",
        "Overlap Decode During Upgrade": "logs/figure_7/model_01-ai/Yi-Coder-1.5B_bs_32_attn_fa_vattn/overlap_upgrade/replica_0/overlap_upgrade_new/request_latencies.csv"
    }
    
    plot_combined_cdf(
        latency_files,
        output_path="logs/figure_7/combined_latency_cdf.png"
    )

if __name__ == "__main__":
    main()