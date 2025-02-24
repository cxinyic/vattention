"""
Module for tracking and analyzing request latencies.
"""

import os
import logging
import statistics
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

class LatencyTracker:
    """
    Tracks and analyzes latencies for benchmark requests.
    
    This class provides methods to log latencies, compute statistics,
    and visualize latency distribution through CDF plots.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the latency tracker.
        
        Args:
            output_dir: Directory to save latency plots and statistics
        """
        self.output_dir = output_dir
        self.latencies: Dict[str, float] = {}
        os.makedirs(output_dir, exist_ok=True)
        
    def log_latency(self, request_id: str, latency: float) -> None:
        """
        Log latency for a specific request.
        
        Args:
            request_id: Unique identifier for the request
            latency: Latency value in seconds
        """
        self.latencies[request_id] = latency
        
    def get_statistics(self) -> Dict[str, float]:
        """
        Calculate statistical metrics for logged latencies.
        
        Returns:
            Dictionary containing latency statistics (min, max, mean, median, p50, p90, p95, p99)
        """
        if not self.latencies:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0
            }
            
        latency_values = list(self.latencies.values())
        
        # Calculate percentiles
        p50 = np.percentile(latency_values, 50)
        p90 = np.percentile(latency_values, 90)
        p95 = np.percentile(latency_values, 95)
        p99 = np.percentile(latency_values, 99)
        
        return {
            "min": min(latency_values),
            "max": max(latency_values),
            "mean": statistics.mean(latency_values),
            "median": statistics.median(latency_values),
            "p50": p50,
            "p90": p90,
            "p95": p95,
            "p99": p99
        }
        
    def plot_cdf(self) -> None:
        """
        Generate and save a CDF (Cumulative Distribution Function) plot of latencies.
        
        The plot is saved to the output directory specified during initialization.
        """
        if not self.latencies:
            logger.warning("No latency data available for plotting CDF")
            return
            
        latency_values = list(self.latencies.values())
        latency_values.sort()
        
        # Calculate CDF values
        y_values = np.arange(1, len(latency_values) + 1) / len(latency_values)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(latency_values, y_values)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Latency (seconds)')
        plt.ylabel('Cumulative Probability')
        plt.title('Latency CDF')
        
        # Add percentile markers
        stats = self.get_statistics()
        percentiles = [50, 90, 95, 99]
        percentile_keys = ['p50', 'p90', 'p95', 'p99']
        
        for i, p in enumerate(percentiles):
            plt.axvline(x=stats[percentile_keys[i]], color='r', linestyle='--', alpha=0.5)
            plt.text(stats[percentile_keys[i]] * 1.05, 0.1 + i * 0.05, 
                     f'P{p}={stats[percentile_keys[i]]:.2f}s', 
                     color='r')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'latency_cdf.png'))
        plt.close()
        
        logger.info(f"Latency CDF plot saved to {self.output_dir}/latency_cdf.png")
        
    def export_latencies(self, file_path: str = None) -> None:
        """
        Export latency data to a CSV file.
        
        Args:
            file_path: Optional path to save the CSV file. If None, saves to output_dir/latencies.csv
        """
        if file_path is None:
            file_path = os.path.join(self.output_dir, 'latencies.csv')
            
        with open(file_path, 'w') as f:
            f.write("request_id,latency_seconds\n")
            for req_id, latency in self.latencies.items():
                f.write(f"{req_id},{latency}\n")
                
        logger.info(f"Latency data exported to {file_path}")