import subprocess
import time
import csv
import os
import threading
import signal
import datetime
import argparse
import numpy as np

class GPUMonitor:
    """Monitor average GPU utilization and memory usage at regular intervals"""
    
    def __init__(
        self, 
        interval: float = 0.1, 
        output_file: str = "gpu_utilization.csv"
    ):
        """
        Initialize the GPU monitor.
        
        Args:
            interval: Time between measurements in seconds (default: 0.1)
            output_file: File to write results to
        """
        self.interval = interval
        self.output_file = output_file
        self.running = False
        self.monitor_thread = None
        self.start_time = None
        
    def _get_gpu_metrics(self):
        """Get current average GPU utilization and memory usage across all GPUs"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse values
            gpu_utils = []
            mem_percentages = []
            
            for line in result.stdout.strip().split('\n'):
                try:
                    parts = [part.strip() for part in line.split(',')]
                    if len(parts) >= 3:
                        gpu_utils.append(float(parts[0]))
                        # Calculate memory usage percentage
                        mem_used = float(parts[1])
                        mem_total = float(parts[2])
                        mem_percent = (mem_used / mem_total) * 100
                        mem_percentages.append(mem_percent)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Could not parse line: {line}, Error: {e}")
                    continue
            
            # Return averages or 0 if no valid readings
            avg_gpu_util = np.mean(gpu_utils) if gpu_utils else 0.0
            avg_mem_percent = np.mean(mem_percentages) if mem_percentages else 0.0
            
            return avg_gpu_util, avg_mem_percent
            
        except subprocess.CalledProcessError as e:
            print(f"Error getting GPU info: {e}")
            return 0.0, 0.0
        except Exception as e:
            print(f"Unexpected error: {e}")
            return 0.0, 0.0
    
    def _monitor_loop(self):
        """Main monitoring loop that runs in a separate thread"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(self.output_file)), exist_ok=True)
        
        # Initialize CSV file with headers
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["avg_gpu_utilization", "avg_memory_utilization"])
        
        self.start_time = time.time()
        self.all_gpu_utils = []
        self.all_mem_utils = []
        
        while self.running:
            try:
                # Get average GPU and memory utilization
                avg_gpu_util, avg_mem_util = self._get_gpu_metrics()
                self.all_gpu_utils.append(avg_gpu_util)
                self.all_mem_utils.append(avg_mem_util)
                
                # Write to CSV
                with open(self.output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([f"{avg_gpu_util:.2f}", f"{avg_mem_util:.2f}"])
                
                # Wait for the next interval
                time.sleep(self.interval)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(self.interval)  # Continue even if there's an error
    
    def start(self):
        """Start the GPU monitoring"""
        if self.running:
            print("Monitor is already running")
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"GPU monitoring started. Saving utilization to {self.output_file} every {self.interval}s")
    
    def stop(self):
        """Stop the GPU monitoring"""
        if not self.running:
            print("Monitor is not running")
            return
            
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        # Calculate and print overall averages
        if self.all_gpu_utils:
            avg_gpu = sum(self.all_gpu_utils) / len(self.all_gpu_utils)
            avg_mem = sum(self.all_mem_utils) / len(self.all_mem_utils)
            print(f"\nOverall average GPU utilization: {avg_gpu:.2f}%")
            print(f"Overall average Memory utilization: {avg_mem:.2f}%")
        
        print(f"GPU monitoring stopped. Data saved to {self.output_file}")


def monitor_gpu_for_program(cmd: list, interval: float = 0.1, output_file: str = "gpu_utilization.csv"):
    """
    Monitor GPU while running a specified command.
    
    Args:
        cmd: Command to run as a list of strings
        interval: Monitoring interval in seconds
        output_file: File to save metrics to
    """
    monitor = GPUMonitor(interval=interval, output_file=output_file)
    
    # Setup signal handlers to ensure monitor stops on interruption
    def signal_handler(sig, frame):
        print("\nReceived interrupt signal. Stopping monitoring and program...")
        monitor.stop()
        if process and process.poll() is None:
            process.terminate()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the monitor
    monitor.start()
    
    # Run the command
    process = None
    try:
        process = subprocess.Popen(cmd)
        process.wait()
    except Exception as e:
        print(f"Error running command: {e}")
    finally:
        # Stop the monitor when the command completes
        monitor.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor GPU utilization")
    parser.add_argument("--interval", type=float, default=0.1, help="Sampling interval in seconds")
    parser.add_argument("--output", type=str, default="gpu_utilization.csv", help="Output CSV file")
    parser.add_argument("--standalone", action="store_true", help="Run as standalone monitor (without running a command)")
    args = parser.parse_args()
    
    if args.standalone:
        # Run as standalone monitor
        monitor = GPUMonitor(interval=args.interval, output_file=args.output)
        try:
            monitor.start()
            print("Press Ctrl+C to stop monitoring")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        finally:
            monitor.stop()
    else:
        print("No command specified. Run with --standalone to monitor without a command.")