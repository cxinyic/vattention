import os
import json
import threading
import logging
import time
from typing import Dict, Any, List, Tuple, Optional
import pathlib

from sarathi.benchmark.config import Config
from sarathi.config import UpgradeConfig, UpgradeStrategy
from sarathi.benchmark.benchmark_launcher import BenchmarkRunnerLauncher, CoordinatedUpgradeState
from sarathi.benchmark.upgrade_utils.upgrade_state import UpgradeState
from sarathi.utils import get_ip
from ray.util.placement_group import PlacementGroup
from sarathi.benchmark.benchmark_runner import BenchmarkRunner

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    upgrade_config: Optional[UpgradeConfig] = None,
    context_length: int = 4608,
    block_size: int = None,
    tp_degree: int = None,
    pp_degree: int = 1,
    placement_group: Optional[PlacementGroup] = None

) -> Config:
    """
    Create benchmark configuration
    
    Args:
        model: Model name
        batch_size: Batch size
        attn_backend: Attention backend
        output_dir: Output directory
        gpu_ids: List of GPU IDs to use
        upgrade_config: Upgrade configuration (optional)
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
        "synthetic_request_generator_num_requests": batch_size,
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
        
        # Explicit GPU assignment
        "replica_resource_mapping": resource_mapping_json,
        "placement_group": placement_group,

        # multi-model flag
        "multi_model": True,
        
    }
    
    # Add upgrade config if provided
    if upgrade_config is not None:
        args.update({
            "upgrade_strategy": upgrade_config.strategy,
            "upgrade_time": upgrade_config.upgrade_time,
            "upgrade_engine_type": upgrade_config.engine_type,
            "upgrade_original_gpu_count": upgrade_config.original_gpu_count,
            "upgrade_drain_strategy": upgrade_config.drain_strategy,
            "upgrade_drain_timeout": upgrade_config.drain_timeout,
            "upgrade_kickout_strategy": upgrade_config.kickout_strategy,
            "upgrade_selection_policy": upgrade_config.selection_policy,
            "upgrade_serving_strategy": upgrade_config.serving_strategy,
            "upgrade_reschedule_policy": upgrade_config.reschedule_policy,
        })
    
    return Config(args)

class MultiModelUpgradeManager:
    """Manages hitless upgrades across multiple models that exchange GPU resources"""
    
    def __init__(self, 
                 multi_model_configs: List[Tuple[str, List[int], List[int]]], 
                 attn_backend: str,
                 batch_size: int,
                 base_output_dir: str,
                 upgrade_config: Optional[UpgradeConfig] = None):
        """
        Initialize the multi-model upgrade manager
        
        Args:
            multi_model_configs: List of tuples (model_name, initial_gpus, final_gpus)
            attn_backend: Attention backend to use
            batch_size: Batch size for each model
            base_output_dir: Base output directory
            upgrade_config: Upgrade configuration 
        """
        self.multi_model_configs = multi_model_configs
        self.attn_backend = attn_backend
        self.batch_size = batch_size
        self.base_output_dir = base_output_dir
        self.upgrade_config = upgrade_config
        
        # Initialize the coordination state
        self.coord_state = CoordinatedUpgradeState()
        
        # Validate configurations
        # if len(multi_model_configs) != 2:
        #     raise ValueError(f"Expected exactly 2 models, got {len(multi_model_configs)}")
        
        # Initialize Ray once for all models
        self._initialize_ray()
        
    def _initialize_ray(self):
        """Initialize Ray once for all models"""
        import ray
        logger.info("Initializing Ray for all models")
        ray.init(ignore_reinit_error=True)
        
        # Get cluster resources to determine total available GPUs
        cluster_resources = ray.cluster_resources()
        logger.info(f"Cluster resources: {cluster_resources}")
        
        # Get total number of GPUs in the cluster
        total_gpus = int(cluster_resources.get("GPU", 0))
        logger.info(f"Total GPUs in cluster: {total_gpus}")
                
        # Get the current node IP
        current_ip = get_ip()
        logger.info(f"Current node IP: {current_ip}")
        
        # Create a placement group with appropriate bundles for all GPUs in the cluster
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(total_gpus)]
        bundles[0][f"node:{current_ip}"] = 0.001  # Node constraint
        
        logger.info(f"Creating placement group with bundles: {bundles}")
        self.pg = ray.util.placement_group(bundles, strategy="PACK")
        logger.info(f"Placement group created: {self.pg}")
        
        # Wait for placement group to be ready
        logger.info("Waiting for placement group to be ready...")
        ray.get(self.pg.ready())
        logger.info("Ray initialized once for all models")
        pg_table = ray.util.placement_group_table(self.pg)
        logger.info(f"Placement group ready: {pg_table}")
    
    def prepare_configs(self) -> Dict[int, Tuple[Config, Config]]:
        """Prepare initial and new configs for all models"""
        configs = {}
        for idx, (model_name, initial_gpus, final_gpus) in enumerate(self.multi_model_configs):
            # Create model-specific output directory
            initial_gpu_str = "_".join(str(gpu) for gpu in initial_gpus)
            final_gpu_str = "_".join(str(gpu) for gpu in final_gpus)
            
            output_dir = os.path.join(
                self.base_output_dir,
                f"multi_model_upgrade",
                f"model_{idx}_{model_name.split('/')[-1]}",
                f"bs_{self.batch_size}",
                f"gpu_{initial_gpu_str}_to_{final_gpu_str}"
            )
            os.makedirs(output_dir, exist_ok=True)
            
            # Use same upgrade time for both models
            initial_config = create_config(
                model=model_name,
                batch_size=self.batch_size,
                attn_backend=self.attn_backend,
                output_dir=output_dir,
                gpu_ids=initial_gpus,
                tp_degree=len(initial_gpus),
                pp_degree=1,
                upgrade_config=self.upgrade_config,
                placement_group=self.pg
            )
            
            # Store GPU IDs directly in the config
            initial_config.gpu_ids = initial_gpus
            
            new_config = create_config(
                model=model_name,
                batch_size=self.batch_size,
                attn_backend=self.attn_backend,
                output_dir=output_dir,
                gpu_ids=final_gpus,
                tp_degree=len(final_gpus),
                pp_degree=1,
                upgrade_config=self.upgrade_config,
                placement_group=self.pg
            )
            
            # Store GPU IDs directly in the config
            new_config.gpu_ids = final_gpus
            
            configs[idx] = (initial_config, new_config)
        
        return configs
    
    def _get_replica_resource_mapping_for_config(self, config):
        """Get resource mapping for a config"""
        resource_mapping = {}
        node_ip = f"node:{get_ip()}"
        
        # Get GPU IDs from config
        gpu_ids = []
        if hasattr(config, 'gpu_ids') and config.gpu_ids is not None:
            # Use directly attached GPU IDs
            gpu_ids = config.gpu_ids
        elif hasattr(config, 'args') and 'gpu_ids' in config.args:
            # Get from args dictionary
            gpu_ids = config.args['gpu_ids']
        else:
            # If we can't find it, try to parse from the resource mapping
            try:
                if hasattr(config, 'replica_resource_mapping') and config.replica_resource_mapping:
                    mapping_dict = json.loads(config.replica_resource_mapping)
                    if "0" in mapping_dict:
                        gpu_ids = [item[1] for item in mapping_dict["0"]]
            except Exception as e:
                logger.error(f"Error parsing GPU IDs from resource mapping: {e}")
        
        if not gpu_ids:
            # Fallback - extract from multi_model_configs based on model index
            logger.warning("No GPU IDs found in config, falling back to multi_model_configs")
            for idx, (_, initial_gpus, final_gpus) in enumerate(self.multi_model_configs):
                if hasattr(config, 'output_dir') and f"model_{idx}_" in config.output_dir:
                    # This is the config for model idx
                    if "to_" in config.output_dir and any(str(gid) in config.output_dir.split("_")[-1] for gid in final_gpus):
                        gpu_ids = final_gpus
                    else:
                        gpu_ids = initial_gpus
                    logger.info(f"Using GPU IDs {gpu_ids} for model {idx} based on output path")
                    break
        
        logger.info(f"Using GPU IDs: {gpu_ids}")
        
        # Create the resource mapping
        resource_mapping["0"] = [(node_ip, gpu_id) for gpu_id in gpu_ids]
        logger.info(f"Created resource mapping: {resource_mapping}")
        return resource_mapping
    
    def run_first_phase(self):
        """Run first phase of all models until upgrade is needed using threads"""
        configs = self.prepare_configs()
        runners = {}
        results = {}
        threads = {}
        
        # Shared dict for storing results from threads with thread safety
        result_lock = threading.Lock()
        
        def run_model(idx, initial_config):
            """Thread function to run a model"""
            try:
                model_name = self.multi_model_configs[idx][0]
                initial_gpus = self.multi_model_configs[idx][1]
                logger.info(f"Thread {idx}: Starting first phase for Model {idx} ({model_name}) with GPUs {initial_gpus}")
                
                # Store GPU IDs directly in the config if not already there
                if not hasattr(initial_config, 'gpu_ids') or initial_config.gpu_ids is None:
                    initial_config.gpu_ids = initial_gpus
                    logger.info(f"Thread {idx}: Added GPU IDs {initial_gpus} directly to config for model {idx}")
                
                # Create resource mapping for the config
                resource_mapping = self._get_replica_resource_mapping_for_config(initial_config)
                
                # Debug log to verify resource mapping
                logger.info(f"Thread {idx}: Resource mapping for model {idx}: {resource_mapping}")
                
                # Create and run the benchmark
                runner = BenchmarkRunner(0, initial_config, resource_mapping["0"])
                
                # Store runner in shared dict
                with result_lock:
                    runners[idx] = runner
                
                # Run until upgrade is needed
                result = runner.run()
                
                # Store result in shared dict
                with result_lock:
                    results[idx] = result
                
                logger.info(f"Thread {idx}: First phase complete for Model {idx}, result: {result['status']}")
                
                # Verify we need to upgrade
                if result["status"] != "UPGRADE_NEEDED":
                    logger.error(f"Thread {idx}: Expected UPGRADE_NEEDED status, got {result['status']}")
                    raise ValueError(f"Expected UPGRADE_NEEDED status, got {result['status']}")
                    
            except Exception as e:
                logger.error(f"Thread {idx}: Error in model thread: {e}")
                # Store the exception to be raised in the main thread
                with result_lock:
                    results[f"error_{idx}"] = e
        
        # Start threads for each model
        for idx, (initial_config, _) in configs.items():
            thread = threading.Thread(
                target=run_model, 
                args=(idx, initial_config),
                name=f"Model-{idx}-Thread"
            )
            threads[idx] = thread
            logger.info(f"Created thread for Model {idx}")
        
        # Start all threads
        for idx, thread in threads.items():
            logger.info(f"Starting thread for Model {idx}")
            thread.start()
        
        # Wait for all threads to complete
        for idx, thread in threads.items():
            logger.info(f"Waiting for Model {idx} thread to complete...")
            thread.join()
            logger.info(f"Model {idx} thread completed")
        
        # Check for errors
        error_keys = [k for k in results.keys() if isinstance(k, str) and k.startswith("error_")]
        if error_keys:
            for error_key in error_keys:
                logger.error(f"Error in {error_key}: {results[error_key]}")
            raise RuntimeError("One or more model threads encountered errors")
        
        logger.info("First phase complete for all models")
        return runners, results
    
    def run_second_phase(self, runners, results):
        """Run second phase with new configurations using threads"""
        
        new_runners = {}
        threads = {}
        
        # Shared dict for storing results with thread safety
        result_lock = threading.Lock()
        thread_errors = {}
        
        # Shutdown Ray only once before starting new threads
        logger.info("Shutting down Ray before second phase")
        import ray
        ray.shutdown()
        self._initialize_ray()
        configs = self.prepare_configs()
        
        def run_model_second_phase(idx, new_config):
            """Thread function to run a model's second phase"""
            try:
                model_name = self.multi_model_configs[idx][0]
                final_gpus = self.multi_model_configs[idx][2]
                logger.info(f"Thread {idx}: Starting second phase for Model {idx} ({model_name}) with GPUs {final_gpus}")
                
                # Store GPU IDs directly in the config if not already there
                if not hasattr(new_config, 'gpu_ids') or new_config.gpu_ids is None:
                    new_config.gpu_ids = final_gpus
                    logger.info(f"Thread {idx}: Added GPU IDs {final_gpus} directly to config for model {idx}")
                
                # Get progress and tracker from first phase
                progress = results[idx]["progress"]
                track = results[idx]["tracker"]
                
                # Create new resource mapping based on new config
                new_resource_mapping = self._get_replica_resource_mapping_for_config(new_config)
                
                # Debug log to verify resource mapping
                logger.info(f"Thread {idx}: Resource mapping for model {idx}: {new_resource_mapping}")
                
                # Create new runner with the new resource mapping
                new_runner = BenchmarkRunner(
                    0, 
                    new_config, 
                    new_resource_mapping["0"],
                    is_new_runner=True
                )
                new_runner.load_progress(progress, track)
                
                # Store new runner in shared dict
                with result_lock:
                    new_runners[idx] = new_runner
                
                # Run the second phase
                logger.info(f"Thread {idx}: Running second phase for Model {idx}")
                result = new_runner.run()
                logger.info(f"Thread {idx}: Second phase complete for Model {idx}, result: {result}")
                
            except Exception as e:
                logger.error(f"Thread {idx}: Error in model second phase thread: {e}")
                # Store the exception
                with result_lock:
                    thread_errors[idx] = e
        
        # Start threads for each model's second phase
        for idx, (_, new_config) in configs.items():
            thread = threading.Thread(
                target=run_model_second_phase, 
                args=(idx, new_config),
                name=f"Model-{idx}-Second-Phase-Thread"
            )
            threads[idx] = thread
            logger.info(f"Created second phase thread for Model {idx}")
        
        # Start all threads
        for idx, thread in threads.items():
            logger.info(f"Starting second phase thread for Model {idx}")
            thread.start()
        
        # Wait for all threads to complete
        for idx, thread in threads.items():
            logger.info(f"Waiting for Model {idx} second phase thread to complete...")
            thread.join()
            logger.info(f"Model {idx} second phase thread completed")
        
        # Check for errors
        if thread_errors:
            for idx, error in thread_errors.items():
                logger.error(f"Error in Model {idx} second phase: {error}")
            raise RuntimeError("One or more model second phase threads encountered errors")
        
        logger.info("Second phase complete for all models")
    
    def run_coordinated_upgrade(self):
        """Run coordinated upgrade of all models"""
        logger.info("\n=====================================================================================")
        logger.info("Starting coordinated multi-model upgrade")
        logger.info("=====================================================================================\n")
        
        # Run first phase until upgrade is needed
        runners, results = self.run_first_phase()
        
        # Run second phase with new configurations
        self.run_second_phase(runners, results)
        
        logger.info("\n=====================================================================================")
        logger.info("Multi-model coordinated upgrade completed!")
        logger.info("=====================================================================================\n")

def main():
    """Main function to run coordinated upgrade of multiple models that exchange GPU resources"""
    # Define models and their GPU configurations:
    # (model_name, initial_gpus, final_gpus)
    multi_model_configs = [
        ("01-ai/Yi-Coder-1.5B", [0, 1], [0, 1, 2, 3])    # Model A: 2 GPUs -> 3 GPUs (gains GPU 1)
        # ("01-ai/Yi-Coder-1.5B", [0, 1], [0])            # Model B: 2 GPUs -> 1 GPU (releases GPU 1)
    ]
    
    # Configuration variables
    attn_backend = "fa_vattn"
    batch_size = 30
    base_output_dir = "logs/multi_model_upgrade"

    # Configure upgrade with NO_SERVE strategy for clean handoff
    upgrade_config = UpgradeConfig(
        strategy=UpgradeStrategy.Mode.UPGRADE,
        upgrade_time=30,                                     # Both models upgrade at this time
        original_gpu_count=2,                                # Starting with 2 GPUs each
        drain_strategy=UpgradeStrategy.DrainStrategy.KICKOUT_IMMEDIATELY,
        drain_timeout=0,
        kickout_strategy=UpgradeStrategy.KickoutStrategy.SELECTED_REQUESTS,
        selection_policy=UpgradeStrategy.SelectionPolicy.BY_ARRIVAL_TIME,
        serving_strategy=UpgradeStrategy.ServingStrategy.NO_SERVE,  # Stop serving during upgrade
        reschedule_policy=UpgradeStrategy.ReschedulePolicy.BY_ARRIVAL_TIME
    )
    
    
    # Create multi-model upgrade manager
    manager = MultiModelUpgradeManager(
        multi_model_configs=multi_model_configs,
        attn_backend=attn_backend,
        batch_size=batch_size,
        base_output_dir=base_output_dir,
        upgrade_config=upgrade_config  
    )
    
    # Run coordinated upgrade with concurrent execution
    manager.run_coordinated_upgrade()

if __name__ == "__main__":
    main()