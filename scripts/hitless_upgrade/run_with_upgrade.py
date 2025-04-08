import os
import time
from typing import Dict, Any, Optional

from sarathi.benchmark.config import Config
from sarathi.config import UpgradeConfig, UpgradeStrategy
from sarathi.benchmark.benchmark_launcher import BenchmarkRunnerLauncher

import pathlib

# Get absolute path to the project root directory
# Assuming the current script is in ~/vattention/scripts/hitless_upgrade
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
    upgrade_config: UpgradeConfig,
    context_length: int = 4608,
    block_size: int = None,
    tp_degree: int = 4,
    pp_degree: int = 1,
) -> Config:
    """Create benchmark configuration"""
    if block_size is None:
        if attn_backend == "fa_paged":
            block_size = 256
        elif attn_backend in ["fa_vattn", "fi_vattn"]:
            block_size = 2097152  # 2MB
        else:
            block_size = 16
            
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
        "synthetic_request_generator_num_requests": batch_size * 2 ,
        # uniform only
        "synthetic_request_generator_length_provider": "uniform",
        "trace_request_length_generator_trace_file": online_trace_file,
        "uniform_request_length_generator_max_tokens": context_length * 2,
        "uniform_request_length_generator_min_tokens": context_length,
        "uniform_request_length_generator_prefill_to_decode_ratio": 8,

        # trace only
        # "synthetic_request_generator_length_provider": "trace",
        # "trace_request_length_generator_trace_file": online_trace_file,
        # "trace_request_length_generator_prefill_scale_factor": 1,
        # "trace_request_length_generator_decode_scale_factor": 1,
        # "trace_request_length_generator_min_tokens": 0,
        # "trace_request_length_generator_max_tokens": context_length,
        
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
        "replica_resource_mapping": None,
        
        # upgrade config
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

    }
    
    return Config(args)



def run_benchmark(
    model: str = "01-ai/Yi-6B-200k",
    attn_backend: str = "fa_vattn",
    batch_size: int = 8,
    base_output_dir: str = "logs/hitless_upgrade",
    upgrade_config: Optional[UpgradeConfig] = None,
) -> None:
    """Run benchmark with upgrade capability"""
    # Use default upgrade config if none provided
    if upgrade_config is None:
        print("Using default upgrade configuration")
        upgrade_config = UpgradeConfig(
            strategy=UpgradeStrategy.Mode.UPGRADE,
            upgrade_time=20,
            original_gpu_count=None,
            drain_strategy=UpgradeStrategy.DrainStrategy.KICKOUT_IMMEDIATELY,
            drain_timeout=5,
            kickout_strategy=UpgradeStrategy.KickoutStrategy.SELECTED_REQUESTS,
            selection_policy=UpgradeStrategy.SelectionPolicy.BY_ARRIVAL_TIME,
            serving_strategy=UpgradeStrategy.ServingStrategy.DECODE_ONLY,
            reschedule_policy=UpgradeStrategy.ReschedulePolicy.BY_ARRIVAL_TIME
        )

    # Create output directory with strategy-specific subdirectory
    if upgrade_config.strategy == UpgradeStrategy.Mode.NO_UPGRADE:
        output_dir = os.path.join(
            base_output_dir,
            f"bs_{batch_size}",
            "no_upgrade"
        )
    else:
        output_dir = os.path.join(
            base_output_dir,
            f"bs_{batch_size}",
            "gpu_2_to_4",
            upgrade_config.serving_strategy.name.lower(),
            upgrade_config.drain_strategy.name.lower(),
            upgrade_config.selection_policy.name.lower(),
            upgrade_config.reschedule_policy.name.lower()
        )
    os.makedirs(output_dir, exist_ok=True)
    
    # Create initial config with "old" engine type
    old_engine_config = UpgradeConfig(
        strategy=upgrade_config.strategy,
        upgrade_time=upgrade_config.upgrade_time,
        engine_type="old",
        original_gpu_count=upgrade_config.original_gpu_count,
        drain_strategy=upgrade_config.drain_strategy,
        drain_timeout=upgrade_config.drain_timeout,
        kickout_strategy=upgrade_config.kickout_strategy,
        selection_policy=upgrade_config.selection_policy,
        serving_strategy=upgrade_config.serving_strategy,
        reschedule_policy=upgrade_config.reschedule_policy
    )
    
    initial_config = create_config(
        model=model,
        batch_size=batch_size,
        attn_backend=attn_backend,
        output_dir=output_dir,
        tp_degree=2,
        pp_degree=1,
        upgrade_config=old_engine_config
    )
    
    # Create new config with "new" engine type
    new_engine_config = UpgradeConfig(
        strategy=upgrade_config.strategy,
        upgrade_time=upgrade_config.upgrade_time,
        engine_type="new",
        original_gpu_count=upgrade_config.original_gpu_count,
        drain_strategy=upgrade_config.drain_strategy,
        drain_timeout=upgrade_config.drain_timeout,
        kickout_strategy=upgrade_config.kickout_strategy,
        selection_policy=upgrade_config.selection_policy,
        serving_strategy=upgrade_config.serving_strategy,
        reschedule_policy=upgrade_config.reschedule_policy
    )
    
    new_config = create_config(
        model=model,
        batch_size=batch_size,
        attn_backend=attn_backend,
        output_dir=output_dir,
        tp_degree=4,
        pp_degree=1,
        upgrade_config=new_engine_config
    )
    
    print("\n=====================================================================================")
    print(f"Running Config ==> Model: {model} Batch Size: {batch_size} Attention Backend: {attn_backend}")
    print(f"Upgrade Configuration:")
    print(upgrade_config)
    print("======================================================================================\n")
    
    # Create and run benchmark
    launcher = BenchmarkRunnerLauncher(
        config=initial_config,
        new_config=new_config
    )
    
    launcher.run_with_upgrade()



def main():
    """Main function to run benchmarks with different configurations"""
    # Configuration variables
    models = ["01-ai/Yi-Coder-1.5B"]
    attn_backends = ["fa_vattn"]
    batch_sizes = [32]
    
    # Create the upgrade configuration once
    upgrade_config = UpgradeConfig(
        strategy=UpgradeStrategy.Mode.UPGRADE,
        upgrade_time=40,
        drain_strategy=UpgradeStrategy.DrainStrategy.KICKOUT_IMMEDIATELY,
        drain_timeout=0,
        kickout_strategy=UpgradeStrategy.KickoutStrategy.SELECTED_REQUESTS,
        selection_policy=UpgradeStrategy.SelectionPolicy.BY_ARRIVAL_TIME,
        serving_strategy=UpgradeStrategy.ServingStrategy.DECODE_ONLY,
        reschedule_policy=UpgradeStrategy.ReschedulePolicy.BY_ARRIVAL_TIME
    )
    # upgrade_config = UpgradeConfig(
    #     strategy=UpgradeStrategy.Mode.UPGRADE,
    #     upgrade_time=40,
    #     drain_strategy=UpgradeStrategy.DrainStrategy.WAIT_THEN_KICKOUT,
    #     drain_timeout=0,
    #     kickout_strategy=UpgradeStrategy.KickoutStrategy.SELECTED_REQUESTS,
    #     selection_policy=UpgradeStrategy.SelectionPolicy.BY_ARRIVAL_TIME,
    #     serving_strategy=UpgradeStrategy.ServingStrategy.DECODE_ONLY,
    #     reschedule_policy=UpgradeStrategy.ReschedulePolicy.BY_ARRIVAL_TIME
    # )
    # upgrade_config = UpgradeConfig(
    #     strategy=UpgradeStrategy.Mode.UPGRADE,
    #     upgrade_time=40,
    #     drain_strategy=UpgradeStrategy.DrainStrategy.KICKOUT_IMMEDIATELY,
    #     drain_timeout=0,
    #     kickout_strategy=UpgradeStrategy.KickoutStrategy.SELECTED_REQUESTS,
    #     selection_policy=UpgradeStrategy.SelectionPolicy.BY_ARRIVAL_TIME,
    #     serving_strategy=UpgradeStrategy.ServingStrategy.PREFILL_ONLY,
    #     reschedule_policy=UpgradeStrategy.ReschedulePolicy.BY_ARRIVAL_TIME
    # )
    # upgrade_config = UpgradeConfig(
    #     strategy=UpgradeStrategy.Mode.UPGRADE,
    #     upgrade_time=40,
    #     drain_strategy=UpgradeStrategy.DrainStrategy.KICKOUT_IMMEDIATELY,
    #     drain_timeout=0,
    #     kickout_strategy=UpgradeStrategy.KickoutStrategy.SELECTED_REQUESTS,
    #     selection_policy=UpgradeStrategy.SelectionPolicy.BY_ARRIVAL_TIME,
    #     serving_strategy=UpgradeStrategy.ServingStrategy.DECODE_ONLY,
    #     reschedule_policy=UpgradeStrategy.ReschedulePolicy.BY_PREFILL_STATUS
    # )
    # upgrade_config = UpgradeConfig(
    #     strategy=UpgradeStrategy.Mode.UPGRADE,
    #     upgrade_time=40,
    #     drain_strategy=UpgradeStrategy.DrainStrategy.KICKOUT_IMMEDIATELY,
    #     drain_timeout=0,
    #     kickout_strategy=UpgradeStrategy.KickoutStrategy.SELECTED_REQUESTS,
    #     selection_policy=UpgradeStrategy.SelectionPolicy.BY_ARRIVAL_TIME,
    #     serving_strategy=UpgradeStrategy.ServingStrategy.NO_SERVE,
    #     reschedule_policy=UpgradeStrategy.ReschedulePolicy.BY_ARRIVAL_TIME
    # )
    # upgrade_config = UpgradeConfig(
    #     strategy=UpgradeStrategy.Mode.UPGRADE,
    #     upgrade_time=40,
    #     original_gpu_count=2,
    #     drain_strategy=UpgradeStrategy.DrainStrategy.KICKOUT_IMMEDIATELY,
    #     drain_timeout=0,
    #     kickout_strategy=UpgradeStrategy.KickoutStrategy.SELECTED_REQUESTS,
    #     selection_policy=UpgradeStrategy.SelectionPolicy.BY_FINISH_TIME,
    #     serving_strategy=UpgradeStrategy.ServingStrategy.DECODE_ONLY,
    #     reschedule_policy=UpgradeStrategy.ReschedulePolicy.BY_ARRIVAL_TIME
    # )
    # upgrade_config = UpgradeConfig(
    #     strategy=UpgradeStrategy.Mode.NO_UPGRADE
    # )
    
    # Run experiments with all configurations
    for model in models:
        for attn_backend in attn_backends:
            for bs in batch_sizes:
                run_benchmark(
                    model=model,
                    attn_backend=attn_backend,
                    batch_size=bs,
                    base_output_dir="logs/hitless_upgrade",
                    upgrade_config=upgrade_config
                )


if __name__ == "__main__":
    main()