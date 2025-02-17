import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import os

class LatencyTracker:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Use fixed filename for easier comparison
        self.csv_path = os.path.join(output_dir, 'request_latencies.csv')
        
        # Initialize CSV with headers
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['request_id', 'latency', 'timestamp'])

    def log_latency(self, request_id, latency):
        """Log a single latency measurement"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([request_id, latency, timestamp])

    def plot_cdf(self, output_filename=None, label=None):
        """Generate CDF plot from logged latencies"""
        # Read the CSV file
        df = pd.read_csv(self.csv_path)
        
        # Sort latencies and calculate CDF
        latencies = sorted(df['latency'].values)
        n = len(latencies)
        cumulative_prob = np.arange(1, n + 1) / n
        
        # Create the plot if it doesn't exist
        if not plt.get_fignums():
            plt.figure(figsize=(10, 6))
            plt.grid(True, alpha=0.3)
            plt.xlabel('Latency (seconds)')
            plt.ylabel('Cumulative Probability')
            plt.title('Latency CDF')
        
        # Plot CDF with label if provided
        label = label if label else os.path.basename(os.path.dirname(self.output_dir))
        line = plt.plot(latencies, cumulative_prob, '-', linewidth=2, label=label)[0]
        color = line.get_color()
        
        # Add percentile lines
        percentiles = [50, 90, 95, 99]
        for p in percentiles:
            percentile_value = np.percentile(latencies, p)
            plt.axvline(x=percentile_value, color=color, linestyle='--', alpha=0.2)
            y_offset = 0.1 * (percentiles.index(p) / len(percentiles))
            plt.text(percentile_value, 0.2 + y_offset, f'{label}\np{p}={percentile_value:.2f}s',
                    rotation=90, verticalalignment='center', fontsize=8)
        
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save plot if filename provided
        if output_filename:
            plt.savefig(os.path.join(self.output_dir, output_filename))
            plt.close()

    def get_statistics(self):
        """Calculate and return basic statistics"""
        df = pd.read_csv(self.csv_path)
        stats = {
            'count': len(df),
            'mean': df['latency'].mean(),
            'median': df['latency'].median(),
            'p90': df['latency'].quantile(0.90),
            'p95': df['latency'].quantile(0.95),
            'p99': df['latency'].quantile(0.99),
            'min': df['latency'].min(),
            'max': df['latency'].max()
        }
        return stats

    @staticmethod
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
                
                # Add percentile lines
                percentiles = [50, 90, 95, 99]
                for i, p in enumerate(percentiles):
                    percentile_value = np.percentile(latencies, p)
                    plt.axvline(x=percentile_value, color=color, linestyle='--', alpha=0.2)
                    y_offset = 0.1 * (i / len(percentiles))
                    plt.text(percentile_value, 0.2 + y_offset, 
                            f'{label}\np{p}={percentile_value:.2f}s',
                            rotation=90, verticalalignment='center', fontsize=8)
        
        plt.xlabel('Latency (seconds)')
        plt.ylabel('Cumulative Probability')
        plt.title('Request Latency CDF Comparison')
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save the combined plot
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()