import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import os
import time
from collections import deque

class LatencyTracker:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Use fixed filename for easier comparison
        self.csv_path = os.path.join(output_dir, 'request_latencies.csv')
        self.throughput_path = os.path.join(output_dir, 'throughput_metrics.csv')
        
        # Initialize latency CSV with headers
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['request_id', 'latency', 'ttft', 'tpot', 'total_tokens', 'timestamp'])
        
        # Initialize throughput CSV with headers (for instantaneous measurements)
        with open(self.throughput_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['elapsed_time', 'tokens_generated'])
        
        # Throughput tracking
        self.start_time = time.monotonic()
        self.last_throughput_time = self.start_time
        self.last_token_count = 0
        self.total_token_count = 0
        self.active_requests_count = 0
        
        # Sliding window for smoother throughput measurements
        self.window_size = 1.0  # 1-second measurement window
        self.token_history = deque()  # Stores (timestamp, token_count) tuples
        
        # Store TTFT and TPOT separately for flexibility
        self.ttft_data = {}
        self.tpot_data = {}

    def log_ttft(self, request_id, ttft):
        """Log Time To First Token for a request"""
        self.ttft_data[request_id] = ttft

    def log_tpot(self, request_id, tpot):
        """Log Time Per Output Token for a request"""
        self.tpot_data[request_id] = tpot
    
    def log_tokens(self, new_tokens):
        """
        Log tokens generated to track instantaneous throughput
        Args:
            request_id: Unique ID for the request
            token_count: Current total token count for this request
            prev_token_count: Previous token count from last update
        """
        current_time = time.monotonic()
        
        # Update total token count
        self.total_token_count += new_tokens
        
        # Add to token history with timestamp
        self.log_token_generation(current_time, self.total_token_count)
        
        # # Measure and record instantaneous throughput
        # self._measure_throughput(current_time)
        
        return new_tokens
    
    def log_token_generation(self, current_time, total_tokens):
        """
        Log each token generation event directly to the CSV file
        """
        # Calculate elapsed time since start
        elapsed_time = current_time - self.start_time
        
        # Log the token generation event to CSV
        with open(self.throughput_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Write header if file is empty
            if f.tell() == 0:
                writer.writerow([
                    'elapsed_time', 
                    'token_count',
                ])
            
            # Write this token generation event
            writer.writerow([
                f"{elapsed_time:.6f}",
                total_tokens
            ])
        
        # Still maintain token history for other purposes if needed
        self.token_history.append((current_time, total_tokens))
    
    # def _measure_throughput(self, current_time):
    #     """
    #     Measure instantaneous throughput based on recent token generation
    #     """
    #     # Remove tokens that are outside the current window
    #     window_start = current_time - self.window_size
    #     while self.token_history and self.token_history[0][0] < window_start:
    #         self.token_history.popleft()
        
    #     # Calculate tokens in the current window
    #     tokens_in_window = sum(tokens for _, tokens in self.token_history)
        
    #     # Only record if we have some tokens and enough time has passed (avoid too frequent recording)
    #     time_since_last = current_time - self.last_throughput_time
    #     if tokens_in_window > 0 and time_since_last >= 0.1:  # Record at most 10 times per second
    #         # Calculate instantaneous throughput
    #         elapsed_time = current_time - self.start_time
    #         tokens_per_second = tokens_in_window / self.window_size
            
    #         # Record throughput
    #         with open(self.throughput_path, 'a', newline='') as f:
    #             writer = csv.writer(f)
    #             writer.writerow([
    #                 datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
    #                 f"{elapsed_time:.6f}",
    #                 tokens_in_window,
    #                 f"{tokens_per_second:.2f}",
    #                 self.active_requests_count
    #             ])
            
    #         self.last_throughput_time = current_time
    
    
    def request_started(self, request_id):
        """Track when a request starts for active request counting"""
        self.active_requests_count += 1
    
    def log_latency(self, request_id, latency, total_tokens=None):
        """Log the complete latency measurement for a request"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Get tokens for this request
        tokens = total_tokens if total_tokens is not None else 0
        
        # Get TTFT and TPOT if available
        ttft = self.ttft_data.get(request_id, 0)
        tpot = self.tpot_data.get(request_id, 0)
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                request_id, 
                latency, 
                ttft,
                tpot,
                tokens,
                timestamp
            ])
            
        # Update active request count
        self.active_requests_count = max(0, self.active_requests_count - 1)
        
        # Clean up tracking data
        if request_id in self.ttft_data:
            del self.ttft_data[request_id]
        if request_id in self.tpot_data:
            del self.tpot_data[request_id]

    def plot_cdf(self, output_filename=None, label=None):
        """Generate CDF plot from logged latencies"""
        # Read the CSV file
        df = pd.read_csv(self.csv_path)
        
        if len(df) == 0:
            return
            
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
    
    def plot_throughput(self, output_filename=None, window_size=5):
        """
        Generate throughput over time plot
        Args:
            output_filename: Name of file to save plot
            window_size: Size of rolling average window in data points
        """
        # Read the throughput CSV file
        df = pd.read_csv(self.throughput_path)
        
        if len(df) == 0:
            return
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        # Plot instantaneous throughput
        plt.scatter(df['elapsed_time'], df['tokens_per_second'], alpha=0.3, color='blue', label='Instantaneous')
        
        # Calculate rolling average if enough data points
        if len(df) > window_size:
            df['smoothed_tps'] = df['tokens_per_second'].rolling(window=window_size).mean()
            plt.plot(df['elapsed_time'], df['smoothed_tps'], linewidth=2, color='darkblue', label=f'{window_size}-point rolling avg')
        
        # Add second y-axis for active requests
        ax2 = plt.gca().twinx()
        ax2.plot(df['elapsed_time'], df['active_requests'], color='red', linestyle='--', label='Active requests')
        ax2.set_ylabel('Active Requests', color='red')
        ax2.tick_params(axis='y', colors='red')
        
        # Format the plot
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.xlabel('Elapsed Time (seconds)')
        plt.ylabel('Tokens per Second')
        plt.title('Instantaneous Throughput Over Time')
        
        # Combine legends from both axes
        lines1, labels1 = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Save plot if filename provided
        if output_filename:
            plt.savefig(os.path.join(self.output_dir, output_filename))
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

    def get_statistics(self):
        """Calculate and return basic statistics"""
        df = pd.read_csv(self.csv_path)
        throughput_df = pd.read_csv(self.throughput_path) if os.path.exists(self.throughput_path) else None
        
        stats = {
            'count': len(df),
            'mean_latency': df['latency'].mean(),
            'median_latency': df['latency'].median(),
            'p90_latency': df['latency'].quantile(0.90),
            'p95_latency': df['latency'].quantile(0.95),
            'p99_latency': df['latency'].quantile(0.99),
            'min_latency': df['latency'].min(),
            'max_latency': df['latency'].max()
        }
        
        # Add TTFT and TPOT stats if available
        if 'ttft' in df.columns and len(df) > 0:
            stats.update({
                'mean_ttft': df['ttft'].mean(),
                'median_ttft': df['ttft'].median(),
                'p90_ttft': df['ttft'].quantile(0.90)
            })
        
        if 'tpot' in df.columns and len(df) > 0:
            stats.update({
                'mean_tpot': df['tpot'].mean(),
                'median_tpot': df['tpot'].median(),
                'p90_tpot': df['tpot'].quantile(0.90)
            })
        
        # Add throughput stats if available
        if throughput_df is not None and len(throughput_df) > 0:
            stats.update({
                'mean_throughput': throughput_df['tokens_per_second'].mean(),
                'peak_throughput': throughput_df['tokens_per_second'].max(),
                'last_throughput': throughput_df['tokens_per_second'].iloc[-1],
                'peak_concurrent_requests': throughput_df['active_requests'].max()
            })
        
        return stats

    def export_latencies(self, include_throughput=True):
        """Export detailed metrics including throughput data"""
        # Generate basic latency CDF
        self.plot_cdf(output_filename='latency_cdf.png')
        
        # Generate throughput plot if enabled
        if include_throughput:
            self.plot_throughput(output_filename='throughput.png')
        
        # Export stats to a summary file
        stats = self.get_statistics()
        with open(os.path.join(self.output_dir, 'stats_summary.txt'), 'w') as f:
            f.write("Performance Metrics Summary\n")
            f.write("==========================\n\n")
            
            f.write("Latency Metrics (seconds):\n")
            f.write(f"  Total Requests: {stats['count']}\n")
            f.write(f"  Mean Latency: {stats['mean_latency']:.4f}\n")
            f.write(f"  Median Latency: {stats['median_latency']:.4f}\n")
            f.write(f"  p90 Latency: {stats['p90_latency']:.4f}\n")
            f.write(f"  p95 Latency: {stats['p95_latency']:.4f}\n")
            f.write(f"  p99 Latency: {stats['p99_latency']:.4f}\n\n")
            
            if 'mean_ttft' in stats:
                f.write("Time To First Token (TTFT) Metrics (seconds):\n")
                f.write(f"  Mean TTFT: {stats['mean_ttft']:.4f}\n")
                f.write(f"  Median TTFT: {stats['median_ttft']:.4f}\n")
                f.write(f"  p90 TTFT: {stats['p90_ttft']:.4f}\n\n")
            
            if 'mean_tpot' in stats:
                f.write("Time Per Output Token (TPOT) Metrics (seconds/token):\n")
                f.write(f"  Mean TPOT: {stats['mean_tpot']:.4f}\n")
                f.write(f"  Median TPOT: {stats['median_tpot']:.4f}\n")
                f.write(f"  p90 TPOT: {stats['p90_tpot']:.4f}\n\n")
            
            if 'mean_throughput' in stats:
                f.write("Throughput Metrics (tokens/second):\n")
                f.write(f"  Mean Throughput: {stats['mean_throughput']:.2f}\n")
                f.write(f"  Peak Throughput: {stats['peak_throughput']:.2f}\n")
                f.write(f"  Latest Throughput: {stats['last_throughput']:.2f}\n")
                f.write(f"  Peak Concurrent Requests: {stats['peak_concurrent_requests']}\n")

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
        
    @staticmethod
    def plot_combined_throughput(throughput_files: dict, output_path: str):
        """
        Plot combined throughput from multiple files.
        
        Args:
            throughput_files: Dict mapping strategy names to csv file paths
            output_path: Path to save the combined plot
        """
        plt.figure(figsize=(12, 6))
        
        for label, file_path in throughput_files.items():
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                
                if len(df) == 0:
                    continue
                    
                # Create smooth line
                window_size = min(15, max(5, len(df) // 10))  # Adaptive window size
                if len(df) > window_size:
                    df['smoothed_tps'] = df['tokens_per_second'].rolling(window=window_size).mean()
                    plt.plot(df['elapsed_time'], df['smoothed_tps'], 
                            linewidth=2, label=f'{label}')
                else:
                    plt.plot(df['elapsed_time'], df['tokens_per_second'], 
                            linewidth=2, label=f'{label}')
        
        plt.xlabel('Elapsed Time (seconds)')
        plt.ylabel('Tokens per Second')
        plt.title('Instantaneous Throughput Comparison')
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save the combined plot
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()