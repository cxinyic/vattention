"""
Module for tracking and analyzing request latencies including TTFT and TPOT.
"""

import os
import logging
import statistics
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

class LatencyTracker:
    """
    Tracks and analyzes latencies for benchmark requests.
    
    This class provides methods to log latencies, compute statistics,
    and visualize latency distribution through CDF plots.
    
    Tracks three main metrics:
    - E2E Latency: End-to-end latency for the entire request
    - TTFT: Time To First Token
    - TPOT: Time Per Output Token (average)
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the latency tracker.
        
        Args:
            output_dir: Directory to save latency plots and statistics
        """
        self.output_dir = output_dir
        # Store individual metrics
        self.latencies: Dict[str, float] = {}  # E2E latencies
        self.ttft: Dict[str, float] = {}  # Time To First Token
        self.tpot: Dict[str, float] = {}  # Time Per Output Token
        
        os.makedirs(output_dir, exist_ok=True)
        
    def log_latency(self, request_id: str, latency: float) -> None:
        """
        Log end-to-end latency for a specific request.
        
        Args:
            request_id: Unique identifier for the request
            latency: End-to-end latency value in seconds
        """
        logger.info(f"LogTracker: Request {request_id} latency: {latency:.4f}s")
        self.latencies[request_id] = latency
    
    def log_ttft(self, request_id: str, ttft: float) -> None:
        """
        Log Time To First Token for a specific request.
        
        Args:
            request_id: Unique identifier for the request
            ttft: Time to first token in seconds
        """
        self.ttft[request_id] = ttft
    
    def log_tpot(self, request_id: str, tpot: float) -> None:
        """
        Log Time Per Output Token for a specific request.
        
        Args:
            request_id: Unique identifier for the request
            tpot: Average time per output token in seconds
        """
        self.tpot[request_id] = tpot
        
    def get_statistics(self) -> Dict[str, float]:
        """
        Calculate statistical metrics for logged end-to-end latencies.
        
        Returns:
            Dictionary containing latency statistics (min, max, mean, median, p50, p90, p95, p99)
        """
        return self._calculate_stats(self.latencies, "latency")
    
    def get_ttft_statistics(self) -> Dict[str, float]:
        """
        Calculate statistical metrics for TTFT values.
        
        Returns:
            Dictionary containing TTFT statistics (min, max, mean, median, p50, p90, p95, p99)
        """
        return self._calculate_stats(self.ttft, "ttft")
    
    def get_tpot_statistics(self) -> Dict[str, float]:
        """
        Calculate statistical metrics for TPOT values.
        
        Returns:
            Dictionary containing TPOT statistics (min, max, mean, median, p50, p90, p95, p99)
        """
        return self._calculate_stats(self.tpot, "tpot")
    
    def _calculate_stats(self, data_dict: Dict[str, float], metric_name: str) -> Dict[str, float]:
        """
        Calculate statistics for a given metric dictionary.
        
        Args:
            data_dict: Dictionary mapping request_ids to metric values
            metric_name: Name of the metric for logging
            
        Returns:
            Dictionary containing metric statistics
        """
        if not data_dict:
            logger.warning(f"No {metric_name} data available for statistics calculation")
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
            
        values = list(data_dict.values())
        
        # Calculate percentiles
        p50 = np.percentile(values, 50)
        p90 = np.percentile(values, 90)
        p95 = np.percentile(values, 95)
        p99 = np.percentile(values, 99)
        
        return {
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p50": p50,
            "p90": p90,
            "p95": p95,
            "p99": p99
        }
        
    def plot_cdf(self, metric: str = "latency") -> None:
        """
        Generate and save a CDF (Cumulative Distribution Function) plot for the specified metric.
        
        Args:
            metric: Which metric to plot ("latency", "ttft", or "tpot")
            
        The plot is saved to the output directory specified during initialization.
        """
        data_dict = None
        title = ""
        y_label = "seconds"
        
        if metric == "latency":
            data_dict = self.latencies
            title = "End-to-End Latency CDF"
        elif metric == "ttft":
            data_dict = self.ttft
            title = "Time To First Token (TTFT) CDF"
        elif metric == "tpot":
            data_dict = self.tpot
            title = "Time Per Output Token (TPOT) CDF"
            y_label = "seconds per token"
        else:
            logger.error(f"Unknown metric: {metric}")
            return
            
        if not data_dict:
            logger.warning(f"No {metric} data available for plotting CDF")
            return
            
        values = list(data_dict.values())
        values.sort()
        
        # Calculate CDF values
        y_values = np.arange(1, len(values) + 1) / len(values)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(values, y_values)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel(f'{metric.upper()} ({y_label})')
        plt.ylabel('Cumulative Probability')
        plt.title(title)
        
        # Add percentile markers
        stats = self._calculate_stats(data_dict, metric)
        percentiles = [50, 90, 95, 99]
        percentile_keys = ['p50', 'p90', 'p95', 'p99']
        
        for i, p in enumerate(percentiles):
            plt.axvline(x=stats[percentile_keys[i]], color='r', linestyle='--', alpha=0.5)
            plt.text(stats[percentile_keys[i]] * 1.05, 0.1 + i * 0.05, 
                     f'P{p}={stats[percentile_keys[i]]:.2f}s', 
                     color='r')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{metric}_cdf.png'))
        plt.close()
        
        logger.info(f"{metric.upper()} CDF plot saved to {self.output_dir}/{metric}_cdf.png")
    
    def plot_all_cdfs(self) -> None:
        """Generate and save CDF plots for all metrics (latency, TTFT, TPOT)."""
        self.plot_cdf("latency")
        self.plot_cdf("ttft")
        self.plot_cdf("tpot")
        
    def export_latencies(self, file_path: Optional[str] = None) -> None:
        """
        Export all latency data to a CSV file.
        
        Args:
            file_path: Optional path to save the CSV file. If None, saves to output_dir/latencies.csv
        """
        if file_path is None:
            file_path = os.path.join(self.output_dir, 'latencies.csv')
            
        with open(file_path, 'w') as f:
            f.write("request_id,e2e_latency_seconds,ttft_seconds,tpot_seconds\n")
            # Get all unique request IDs across all metrics
            all_request_ids = set(list(self.latencies.keys()) + 
                                 list(self.ttft.keys()) + 
                                 list(self.tpot.keys()))
            logger.info(f"Exporting latency len of self.latencies is {len(self.latencies)} len of self.ttft is {len(self.ttft)} len of self.tpot is {len(self.tpot)}")
            for req_id in all_request_ids:
                e2e_latency = self.latencies.get(req_id, "")
                ttft = self.ttft.get(req_id, "")
                tpot = self.tpot.get(req_id, "")
                f.write(f"{req_id},{e2e_latency},{ttft},{tpot}\n")
                
        logger.info(f"Latency data exported to {file_path}")
        
    def print_summary(self) -> None:
        """Print a summary of all latency statistics to the logger."""
        e2e_stats = self.get_statistics()
        ttft_stats = self.get_ttft_statistics()
        tpot_stats = self.get_tpot_statistics()
        
        logger.info("===== Latency Statistics Summary =====")
        logger.info("End-to-End Latency Statistics:")
        for key, value in e2e_stats.items():
            logger.info(f"  {key}: {value:.4f}s")
            
        logger.info("\nTime To First Token (TTFT) Statistics:")
        for key, value in ttft_stats.items():
            logger.info(f"  {key}: {value:.4f}s")
            
        logger.info("\nTime Per Output Token (TPOT) Statistics:")
        for key, value in tpot_stats.items():
            logger.info(f"  {key}: {value:.4f}s per token")
        logger.info("=====================================")