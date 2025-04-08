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
                values = sorted(df[metric].values)
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
                print(f"Mean: {df[metric].mean():.2f}s")
                print(f"Count: {len(df)}")
            else:
                print(f"Warning: Metric {metric} not found in {file_path}")
        else:
            print(f"Warning: File not found - {file_path}")
    
    # Customize x-axis label based on metric
    if metric == "e2e_latency_seconds":
        plt.xlabel('End-to-End Latency (seconds)')
    elif metric == "ttft_seconds":
        plt.xlabel('Time to First Token (seconds)')
    elif metric == "tpot_seconds":
        plt.xlabel('Time per Output Token (seconds)')
    else:
        plt.xlabel(f'{metric} (seconds)')
        
    plt.ylabel('Cumulative Probability')
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the plot
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    latency_files = {
        # "No Upgrade": "logs/multi_model_upgrade_overlap/bs_32/gpu_0_1_2_3_to_0_1_2_3/no_upgrade/request_latencies.csv",
        "Upgrade|No Overlap Serving": "logs/multi_model_upgrade_overlap/bs_32/gpu_0_1_2_3_to_0_1_2_3/no_serve/request_latencies.csv",
        # "Overlap Decode|kickout|select_by_to_finish|reschedule_by_arrival": "logs/hitless_upgrade/bs_32/decode_only/kickout_immediately/by_finish_time/by_arrival_time/latencies.csv",
        # "Overlap Decode|kickout|select_by_arrival|reschedule_by_arrival": "logs/hitless_upgrade/bs_32/decode_only/kickout_immediately/by_arrival_time/by_arrival_time/latencies.csv",
        # "Overlap Decode|kickout|select_by_arrival|reschedule_by_prefill": "logs/hitless_upgrade/bs_32/decode_only/kickout_immediately/by_arrival_time/by_prefill_status/latencies.csv",
        # "Overlap Decode|wait_then_kick|select_by_arrival|reschedule_by_arrival": "logs/hitless_upgrade/bs_32/decode_only/wait_then_kickout/by_arrival_time/by_arrival_time/latencies.csv",
        # "Overlap Prefill|kickout|select_by_arrival|reschedule_by_arrival":
        # "logs/hitless_upgrade/bs_32/prefill_only/kickout_immediately/by_arrival_time/by_arrival_time/latencies.csv",
        # "No Upgrade": "logs/hitless_upgrade/bs_32/no_upgrade/latencies.csv",
        # "Upgrade|No Overlap Serving": "logs/hitless_upgrade/bs_32/gpu_2_to_4/no_serve/latencies.csv",
        "Overlap Decode|kickout|select_by_to_finish|reschedule_by_arrival": "logs/multi_model_upgrade_overlap/bs_32/gpu_0_1_2_3_to_0_1_2_3/overlap/request_latencies.csv",
        # "Overlap Decode|kickout|select_by_arrival|reschedule_by_arrival": "logs/hitless_upgrade/bs_32/gpu_2_to_4/decode_only/kickout_immediately/by_arrival_time/by_arrival_time/latencies.csv",
        # "Overlap Decode|kickout|select_by_arrival|reschedule_by_prefill": "logs/hitless_upgrade/bs_32/gpu_2_to_4/decode_only/kickout_immediately/by_arrival_time/by_prefill_status/latencies.csv",
        # "Overlap Decode|wait_then_kick|select_by_arrival|reschedule_by_arrival": "logs/hitless_upgrade/bs_32/gpu_2_to_4/decode_only/wait_then_kickout/by_arrival_time/by_arrival_time/latencies.csv",
        # "Overlap Prefill|kickout|select_by_arrival|reschedule_by_arrival": "logs/hitless_upgrade/bs_32/gpu_2_to_4/prefill_only/kickout_immediately/by_arrival_time/by_arrival_time/latencies.csv",
    }
    
    # Create output directory
    output_dir = "logs/hitless_upgrade/bs_32/gpu_2_to_4/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate separate plots for each metric
    plot_metric_cdf(
        latency_files,
        metric="latency",
        output_path=f"{output_dir}/e2e_latency_cdf.png",
        title="End-to-End Latency CDF Comparison"
    )
    
    plot_metric_cdf(
        latency_files,
        metric="ttft",
        output_path=f"{output_dir}/ttft_cdf.png",
        title="Time to First Token CDF Comparison"
    )
    
    plot_metric_cdf(
        latency_files,
        metric="tpot",
        output_path=f"{output_dir}/tpot_cdf.png",
        title="Time per Output Token CDF Comparison"
    )

if __name__ == "__main__":
    main()