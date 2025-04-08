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
        if len(multi_model_configs) != 2:
            raise ValueError(f"Expected exactly 2 models, got {len(multi_model_configs)}")
        
        self._create_ray_placement_group()
        
    
    def _create_ray_placement_group(self):
        # Initialize Ray here once for all models
        import ray
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
        # Each bundle requests GPU and CPU resources
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(total_gpus)]
        
        # Add node constraint to the first bundle to ensure it's created on the current node
        bundles[0][f"node:{current_ip}"] = 0.001
        
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
        original_upgrade_time = self.upgrade_config.upgrade_time    
        for idx, (model_name, initial_gpus, final_gpus) in enumerate(self.multi_model_configs):
            # Create model-specific output directory
            initial_gpu_str = "_".join(str(gpu) for gpu in initial_gpus)
            final_gpu_str = "_".join(str(gpu) for gpu in final_gpus)
            
            output_dir = os.path.join(
                self.base_output_dir,
                f"bs_{self.batch_size}",
                f"gpu_{initial_gpu_str}_to_{final_gpu_str}",
                f"no_upgrade"
            )
            os.makedirs(output_dir, exist_ok=True)
            
            # TODO(XY): assume that model A will expand and model B will shrink
            #TODO(XY): should add a expand/shrink parameter
            old_engine_config = self.upgrade_config
            # if idx == 0:
            #     old_engine_config.upgrade_time = 999999  # Model A will be
            #     # triggered manually
            #     # old_engine_config.upgrade_time = original_upgrade_time
            # else:
            #     old_engine_config.upgrade_time = original_upgrade_time
            initial_config = create_config(
                model=model_name,
                batch_size=self.batch_size,
                attn_backend=self.attn_backend,
                output_dir=output_dir,
                gpu_ids=initial_gpus,
                tp_degree=2,
                pp_degree=2,
                upgrade_config=old_engine_config,
                placement_group=self.pg
            )
            
            new_config = create_config(
                model=model_name,
                batch_size=self.batch_size,
                attn_backend=self.attn_backend,
                output_dir=output_dir,
                gpu_ids=final_gpus,
                tp_degree=4,
                pp_degree=1,
                upgrade_config=self.upgrade_config,
                placement_group=self.pg
            )
            
            configs[idx] = (initial_config, new_config)
        
        return configs
    
    def run_model_shrink(self, initial_config: Config, new_config: Config):
        """Run Model B with upgrade (GPUs 2,3 -> GPU 3)"""
        model_name, initial_gpus, final_gpus = self.multi_model_configs[1]  # Model B is at index 1
        
        logger.info(f"\n=====================================================================================")
        logger.info(f"Running Model B: {model_name}")
        logger.info(f"GPUs: {initial_gpus} → {final_gpus}")
        logger.info(f"=====================================================================================\n")
        
        # Create launcher
        launcher = BenchmarkRunnerLauncher(
            config=initial_config,
            new_config=new_config
        )
        
        # Set the coord_state on the launcher for use in our patched method
        launcher.coord_state = self.coord_state
        
        # Run with upgrade
        result = launcher.run_with_upgrade()
        
        # Wait a bit to ensure signal is processed by Model A
        time.sleep(1)
        
        # Signal that Model B upgrade is complete (if not already signaled by our patch)
        if not self.coord_state._gpu_2_freed.is_set():
            logger.info("Model B upgrade complete - signaling that GPU 2 is now free (fallback)")
            self.coord_state.signal_gpu_2_freed()
        return result
        

    def run_model_expand(self, initial_config: Config, new_config: Config):
        """Run Model A with upgrade (GPUs 0,1 -> GPUs 0,1,2)"""
        model_name, initial_gpus, final_gpus = self.multi_model_configs[0]  # Model A is at index 0
        
        logger.info(f"\n=====================================================================================")
        logger.info(f"Running Model A: {model_name}")
        logger.info(f"GPUs: {initial_gpus} → {final_gpus}")
        logger.info(f"=====================================================================================\n")
        
        # Create benchmark launcher

        launcher = BenchmarkRunnerLauncher(
            config=initial_config,
            new_config=new_config
        )
        launcher.coord_state = self.coord_state
        
        # Start running normally
        logger.info("Model A running normally with GPUs 2,3")
        logger.info("Model A will wait for GPU 2 to be freed by Model B before upgrading...")
        
        # Define a function to monitor and trigger the upgrade
        # def monitor_and_trigger_upgrade():
        #     # Wait for GPU 2 to be freed by Model B
        #     while not self.coord_state.wait_for_gpu_2_freed(timeout=1):
        #         # Keep waiting until GPU 2 is freed
        #         pass
            
        #     logger.info("Model A received signal that GPU 2 is available - triggering upgrade")
            
        #     # Set the upgrade time to now (0 seconds from now) to trigger the upgrade process
        #     if hasattr(launcher, "_runner") and hasattr(launcher._runner, "_llm_engine") and \
        #     hasattr(launcher._runner._llm_engine, "upgrade_config"):
                
        #         # Set the upgrade time to slightly before current time to trigger immediate upgrade
        #         launcher._runner._config.upgrade_time = 0.01
        #         logger.info(f"New upgrade time: {launcher._runner._config.upgrade_time}")
                        
        # # Start the monitoring thread
        # monitor_thread = threading.Thread(target=monitor_and_trigger_upgrade)
        # monitor_thread.daemon = True
        # monitor_thread.start()
        
        # Run the benchmark - this will be upgraded when the monitor thread triggers it
        result = launcher.run_with_upgrade()

        return result

    def run_coordinated_upgrade(self):
        """Run all models with coordinated upgrade"""
        # Prepare configs for all models
        configs = self.prepare_configs()
        
        # Create threads for each model
        
        # model_shrink_thread = threading.Thread(
        #     target=self.run_model_shrink,
        #     args=(configs[1][0], configs[1][1]),
        #     name="ModelB-Thread"
        # )

        model_expand_thread = threading.Thread(
            target=self.run_model_expand,
            args=(configs[0][0], configs[0][1]),
            name="ModelA-Thread"
        )
        
        # Start all model threads
        # logger.info("Starting Model B thread (GPUs 0,1)")
        # model_shrink_thread.start()
        
        logger.info("Starting Model A thread (GPUs 2,3)")
        model_expand_thread.start()
        
        # Wait for all threads to complete
        logger.info("Waiting for all model threads to complete...")
        # model_shrink_thread.join()
        model_expand_thread.join()
        
        logger.info("\n=====================================================================================")
        logger.info("Multi-model coordinated upgrade completed!")
        logger.info("=====================================================================================\n")

def main():
    """Main function to run coordinated upgrade of multiple models that exchange GPU resources"""
    # Define models and their GPU configurations:
    # (model_name, initial_gpus, final_gpus)
    multi_model_configs = [
        ("01-ai/Yi-Coder-1.5B", [0, 1, 2, 3], [0, 1, 2, 3]),    # Model A: 2 GPUs -> 3 GPUs
        ("01-ai/Yi-Coder-1.5B", [0, 1], [0])         # Model B: 2 GPUs -> 1 GPU
    ]
    
    # Configuration variables
    attn_backend = "fa_vattn"
    batch_size = 32
    base_output_dir = "logs/multi_model_upgrade_overlap"

    # TODO(XY): now assume that two models share the same upgrade config
    # upgrade_config = UpgradeConfig(
    #     strategy=UpgradeStrategy.Mode.UPGRADE,
    #     upgrade_time=30,
    #     # original_gpu_count=2,
    #     drain_strategy=UpgradeStrategy.DrainStrategy.KICKOUT_IMMEDIATELY,
    #     drain_timeout=0,
    #     kickout_strategy=UpgradeStrategy.KickoutStrategy.SELECTED_REQUESTS,
    #     selection_policy=UpgradeStrategy.SelectionPolicy.BY_ARRIVAL_TIME,
    #     serving_strategy=UpgradeStrategy.ServingStrategy.DECODE_ONLY,
    #     reschedule_policy=UpgradeStrategy.ReschedulePolicy.BY_ARRIVAL_TIME
    # )
    upgrade_config = UpgradeConfig(
        strategy=UpgradeStrategy.Mode.NO_UPGRADE
    )
    # upgrade_config = UpgradeConfig(
    #     strategy=UpgradeStrategy.Mode.UPGRADE,
    #     upgrade_time=30,
    #     drain_strategy=UpgradeStrategy.DrainStrategy.KICKOUT_IMMEDIATELY,
    #     drain_timeout=0,
    #     kickout_strategy=UpgradeStrategy.KickoutStrategy.SELECTED_REQUESTS,
    #     selection_policy=UpgradeStrategy.SelectionPolicy.BY_ARRIVAL_TIME,
    #     serving_strategy=UpgradeStrategy.ServingStrategy.NO_SERVE,
    #     reschedule_policy=UpgradeStrategy.ReschedulePolicy.BY_ARRIVAL_TIME
    # )
    # upgrade_config = UpgradeConfig(
    #     strategy=UpgradeStrategy.Mode.UPGRADE,
    #     upgrade_time=20,
    #     original_gpu_count=2,
    #     drain_strategy=UpgradeStrategy.DrainStrategy.KICKOUT_IMMEDIATELY,
    #     drain_timeout=0,
    #     kickout_strategy=UpgradeStrategy.KickoutStrategy.SELECTED_REQUESTS,
    #     selection_policy=UpgradeStrategy.SelectionPolicy.BY_ARRIVAL_TIME,
    #     serving_strategy=UpgradeStrategy.ServingStrategy.NO_SERVE,
    #     reschedule_policy=UpgradeStrategy.ReschedulePolicy.BY_PREFILL_STATUS
    # )
    # upgrade_config = UpgradeConfig(
    #     strategy=UpgradeStrategy.Mode.NO_UPGRADE
    # )
    
    # Create multi-model upgrade manager
    manager = MultiModelUpgradeManager(
        multi_model_configs=multi_model_configs,
        attn_backend=attn_backend,
        batch_size=batch_size,
        base_output_dir=base_output_dir,
        upgrade_config=upgrade_config  
    )
    
    # Run multi-model upgrade
    manager.run_coordinated_upgrade()

if __name__ == "__main__":
    main()