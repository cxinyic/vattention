import os
import json
import threading
import logging
import importlib
from typing import Dict, Any, List, Tuple
import pathlib

from sarathi.benchmark.config import Config
from sarathi.benchmark.benchmark_launcher import BenchmarkRunnerLauncher
from sarathi.utils import get_ip

# Modify BenchmarkRunnerLauncher to skip ray initialization
from sarathi.benchmark import benchmark_launcher
# Store the original __init__ method
original_init = benchmark_launcher.BenchmarkRunnerLauncher.__init__

# Create a patched __init__ method that skips ray initialization
def patched_init(self, config, new_config=None):
    # Skip ray.init() call and directly assign attributes
    self._config = config
    self._new_config = new_config
    self._is_multi_replica = self._config.cluster_num_replicas > 1
    
    # Call rest of initialization without ray.init()
    required_blocks, pages_per_block = self.calculate_upgrade_blocks()
    self._config.upgrade_required_blocks = required_blocks
    self._config.pages_per_block = pages_per_block
    
    # Continue with normal initialization
    if self._is_multi_replica:
        self._validate_cluster_resources()
        self._runners = self._create_runners()
        self._aggregate_metric_store = self._create_aggregate_metric_store()
    else:
        # Use explicit resource mapping from config if available
        if self._config.replica_resource_mapping:
            try:
                self.replica_resource_mapping = json.loads(self._config.replica_resource_mapping)
                logging.info(f"Using replica resource mapping from config: {self.replica_resource_mapping}")
            except Exception as e:
                logging.warning(f"Failed to parse replica_resource_mapping from config: {e}")
                logging.warning("Falling back to dynamic resource mapping")
                self.replica_resource_mapping = self._get_replica_resource_mapping()
        else:
            self.replica_resource_mapping = self._get_replica_resource_mapping()
            
        assert len(self.replica_resource_mapping) == 1
        self._runner = benchmark_launcher.BenchmarkRunner(
            0, 
            self._config, 
            self.replica_resource_mapping["0"]
        )
    
    # Apply wandb config if needed
    if hasattr(benchmark_launcher, 'wandb') and benchmark_launcher.wandb.run is not None:
        benchmark_launcher.wandb.config.update(self._config.__dict__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get absolute path to the project root directory
script_dir = pathlib.Path(__file__).parent.resolve()
project_root = script_dir.parent.parent  # Go up two levels to ~/vattention

# Path to the trace file
traces_dir = os.path.join(project_root, "scripts", "artifact_asplos25", "traces")
online_trace_file = os.path.join(traces_dir, "arxiv_long_online.csv")


def create_config(
    model: str,
    batch_size: int,
    attn_backend: str,
    output_dir: str,
    gpu_ids: List[int],
    context_length: int = 4608,
    block_size: int = None,
    tp_degree: int = None,
    pp_degree: int = 1,
) -> Config:
    """
    Create benchmark configuration
    
    Args:
        model: Model name
        batch_size: Batch size
        attn_backend: Attention backend
        output_dir: Output directory
        gpu_ids: List of GPU IDs to use
        context_length: Context length
        block_size: Block size (will be inferred if None)
        tp_degree: Tensor parallel degree (defaults to len(gpu_ids))
        pp_degree: Pipeline parallel degree
    """
    if tp_degree is None:
        tp_degree = len(gpu_ids)
        
    if block_size is None:
        if attn_backend == "fa_paged":
            block_size = 256
        elif attn_backend in ["fa_vattn", "fi_vattn"]:
            block_size = 2097152  # 2MB
        else:
            block_size = 16
    
    # Create resource mapping in the format expected by Ray
    # Format: {replica_id: [(node_ip, gpu_id), ...]}
    resource_mapping = {}
    node_ip = f"node:{get_ip()}"
    resource_mapping["0"] = [(node_ip, gpu_id) for gpu_id in gpu_ids]
    
    # Convert to JSON string as expected by the Config
    resource_mapping_json = json.dumps(resource_mapping)
    logger.info(f"Created resource mapping for {model}: {resource_mapping_json}")
            
    args = {
        # model config
        "model_name": model,
        "model_max_model_len": 16384,
        "model_block_size": block_size,
        "model_attention_backend": attn_backend,
        "gpu_memory_utilization": 0.9,
        "model_tensor_parallel_degree": tp_degree,
        "model_pipeline_parallel_degree": pp_degree,
        "model_load_format": "auto",
        
        # cluster config
        "cluster_num_replicas": 1,
        
        # request generator config
        "request_generator_provider": "synthetic",
        "synthetic_request_generator_interval_provider": "static",
        "synthetic_request_generator_num_requests": batch_size * 2,
        # uniform only
        "synthetic_request_generator_length_provider": "uniform",
        "trace_request_length_generator_trace_file": online_trace_file,
        "uniform_request_length_generator_max_tokens": context_length * 2,
        "uniform_request_length_generator_min_tokens": context_length,
        "uniform_request_length_generator_prefill_to_decode_ratio": 8,
        
        # scheduler config
        "replica_scheduler_provider": "sarathi",
        "replica_scheduler_max_batch_size": batch_size,
        "sarathi_scheduler_chunk_size": 2097152,
        "vllm_scheduler_max_tokens_in_batch": 2097152,
        
        # metrics config
        "metrics_store_enable_op_level_metrics": False,
        "metrics_store_keep_individual_batch_metrics": True,
        "metrics_store_enable_cpu_op_level_metrics": False,
        "metrics_store_enable_request_outputs": False,
        "metrics_store_wandb_project": None,
        "metrics_store_wandb_group": None,
        "metrics_store_wandb_run_name": None,
        "metrics_store_wandb_sweep_id": None,
        "metrics_store_wandb_run_id": None,
        
        # output config
        "output_dir": output_dir,
        
        # other required configs
        "write_metrics": True,
        "write_chrome_trace": False,
        "enable_profiling": False,
        "seed": 42,
        "time_limit": 0,  # 0 means no time limit
        
        # Explicit GPU assignment - this is crucial
        "replica_resource_mapping": resource_mapping_json,
    }
    
    return Config(args)


class MultiModelRunner:
    """Runs multiple models concurrently with specific GPU assignments"""
    
    def __init__(self, 
                 model_configs: List[Tuple[str, List[int]]], 
                 attn_backend: str,
                 batch_size: int,
                 base_output_dir: str):
        """
        Initialize the multi-model runner
        
        Args:
            model_configs: List of tuples (model_name, gpu_ids)
                Where gpu_ids is a list of GPU IDs to use for that model
            attn_backend: Attention backend to use
            batch_size: Batch size for each model
            base_output_dir: Base output directory
        """
        self.model_configs = model_configs
        self.attn_backend = attn_backend
        self.batch_size = batch_size
        self.base_output_dir = base_output_dir
        self.launchers = []
        
        # Validate that all GPU IDs are unique across models
        all_gpus = []
        for _, gpu_ids in model_configs:
            all_gpus.extend(gpu_ids)
        
        if len(all_gpus) != len(set(all_gpus)):
            raise ValueError("Duplicate GPU IDs found across models. Each GPU can only be assigned to one model.")
            
        # Initialize Ray here once for all models
        import ray
        ray.init(ignore_reinit_error=True)
        logger.info("Ray initialized once for all models")
        
        # Patch BenchmarkRunnerLauncher.__init__ to skip ray.init()
        benchmark_launcher.BenchmarkRunnerLauncher.__init__ = patched_init
        logger.info("Patched BenchmarkRunnerLauncher.__init__ to skip ray.init()")
    
    def prepare_configs(self) -> List[Config]:
        """Prepare configs for all models"""
        configs = []
        
        for idx, (model_name, gpu_ids) in enumerate(self.model_configs):
            # Create model-specific output directory
            gpu_str = "_".join(str(gpu) for gpu in gpu_ids)
            output_dir = os.path.join(
                self.base_output_dir,
                f"multi_model",
                f"model_{idx}_{model_name.split('/')[-1]}",
                f"bs_{self.batch_size}",
                f"gpus_{gpu_str}"
            )
            os.makedirs(output_dir, exist_ok=True)
            
            # Create config with explicit GPU assignment
            config = create_config(
                model=model_name,
                batch_size=self.batch_size,
                attn_backend=self.attn_backend,
                output_dir=output_dir,
                gpu_ids=gpu_ids,
                pp_degree=1
            )
            
            configs.append(config)
        
        return configs
    
    def run_model(self, model_idx: int, config: Config):
        """Run a single model"""
        model_name = self.model_configs[model_idx][0]
        gpu_ids = self.model_configs[model_idx][1]
        
        print(f"\n=====================================================================================")
        print(f"Running Model {model_idx}: {model_name}")
        print(f"GPUs: {gpu_ids}, Batch Size: {self.batch_size}, Backend: {self.attn_backend}")
        print(f"=====================================================================================\n")
        
        # Get the resource mapping from the config
        resource_mapping_json = config.replica_resource_mapping
        resource_mapping = json.loads(resource_mapping_json)
        logger.info(f"Model {model_idx} running with resource mapping: {resource_mapping}")
        
        # Create and run benchmark
        launcher = BenchmarkRunnerLauncher(config=config)
        
        # Store launcher for reference
        self.launchers.append(launcher)
        
        # Run model
        launcher.run()
    
    def run_all_models(self):
        """Run all models in parallel"""
        # Prepare configs for all models
        configs = self.prepare_configs()
        
        # Create threads for each model
        threads = []
        for idx, config in enumerate(configs):
            thread = threading.Thread(
                target=self.run_model,
                args=(idx, config)
            )
            threads.append(thread)
        
        # Start all model threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        print("\n=====================================================================================")
        print("Multi-model execution completed!")
        print("=====================================================================================\n")


def main():
    """Main function to run multiple models concurrently with specific GPU assignments"""
    # Define models and their GPU configurations: (model_name, [gpu_ids])
    model_configs = [
        ("01-ai/Yi-Coder-1.5B", [0, 1]),      # Model A: GPUs 0, 1
        ("01-ai/Yi-Coder-1.5B", [2, 3])    # Model B: GPUs 2, 3
    ]
    
    # Configuration variables
    attn_backend = "fa_vattn"
    batch_size = 16
    base_output_dir = "logs/multi_model"
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create multi-model runner
    runner = MultiModelRunner(
        model_configs=model_configs,
        attn_backend=attn_backend,
        batch_size=batch_size,
        base_output_dir=base_output_dir
    )
    
    # Run all models
    runner.run_all_models()
    
    # Restore the original __init__ method when done
    benchmark_launcher.BenchmarkRunnerLauncher.__init__ = original_init


if __name__ == "__main__":
    main()