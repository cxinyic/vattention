import os
import time
from typing import Dict, Any

from sarathi.benchmark.config import Config
from sarathi.config import UpgradeConfig, UpgradeStrategy
from sarathi.benchmark.benchmark_runner import BenchmarkRunnerLauncher
from sarathi.benchmark.latency_tracker import LatencyTracker

def create_config(
    model: str,
    batch_size: int,
    attn_backend: str,
    output_dir: str,
    context_length: int = 4608,
    block_size: int = None,
    tp_degree: int = 4,
    pp_degree: int = 1,
    upgrade_strategy: UpgradeStrategy = UpgradeStrategy.NO_UPGRADE,
    upgrade_required_blocks: int = 20,
    upgrade_engine_type: str = "old",
    upgrade_time: float = None,
) -> Config:
    """Create benchmark configuration"""
    if block_size is None:
        if attn_backend == "fa_paged":
            block_size = 256
        elif attn_backend in ["fa_vattn", "fi_vattn"]:
            block_size = 2097152  # 2MB
        else:
            block_size = 16
    print(f"upgrade_strategy: {upgrade_strategy}")
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
        "synthetic_request_generator_length_provider": "uniform",
        "synthetic_request_generator_interval_provider": "static",
        "synthetic_request_generator_num_requests": batch_size * 2,
        "uniform_request_length_generator_max_tokens": context_length,
        "uniform_request_length_generator_min_tokens": context_length,
        "uniform_request_length_generator_prefill_to_decode_ratio": 8,
        "trace_request_length_generator_prefill_scale_factor": 1,
        "trace_request_length_generator_decode_scale_factor": 1,
        "trace_request_generator_max_tokens": context_length,
        
        # scheduler config
        "replica_scheduler_provider": "sarathi",
        "replica_scheduler_max_batch_size": batch_size,
        "sarathi_scheduler_chunk_size": 2097152,
        "vllm_scheduler_max_tokens_in_batch": 2097152,
        
        # metrics config
        "metrics_store_enable_op_level_metrics": False,
        "metrics_store_keep_individual_batch_metrics": True,
        "metrics_store_enable_cpu_op_level_metrics": False,
        "metrics_store_enable_request_outputs": True,
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
        "upgrade_strategy": upgrade_strategy,
        "upgrade_required_blocks": upgrade_required_blocks,
        "upgrade_engine_type": upgrade_engine_type,
        "upgrade_time": upgrade_time,
    }
    
    return Config(args)

def run_benchmark(
    model: str = "01-ai/Yi-6B-200k",
    attn_backend: str = "fa_vattn",
    batch_size: int = 8,
    upgrade_time: float = 10,  
    base_output_dir: str = "logs/figure_7",
    upgrade_strategy: UpgradeStrategy = UpgradeStrategy.NO_UPGRADE,
    upgrade_required_blocks: int = 20,
) -> None:
    """Run benchmark with upgrade capability"""
    
    # Create output directory with strategy-specific subdirectory
    output_dir = os.path.join(
        base_output_dir,
        f"model_{model}_bs_{batch_size}_attn_{attn_backend}",
        upgrade_strategy.name.lower()
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"In run_benchmark, upgrade_strategy: {upgrade_strategy}")
    # Create initial config with "old" engine type
    initial_config = create_config(
        model=model,
        batch_size=batch_size,
        attn_backend=attn_backend,
        output_dir=output_dir,
        tp_degree=4,
        pp_degree=1,
        upgrade_strategy=upgrade_strategy,
        upgrade_required_blocks=upgrade_required_blocks,
        upgrade_engine_type="old",
        upgrade_time=upgrade_time
    )
    
    # Create new config with "new" engine type
    new_config = create_config(
        model=model,
        batch_size=batch_size,
        attn_backend=attn_backend,
        output_dir=output_dir,
        tp_degree=4,
        pp_degree=1,
        upgrade_strategy=upgrade_strategy,
        upgrade_required_blocks=upgrade_required_blocks,
        upgrade_engine_type="new",
        upgrade_time=upgrade_time
    )
    
    print("\n=====================================================================================")
    print(f"Running Config ==> Model: {model} Batch Size: {batch_size} Attention Backend: {attn_backend}")
    print(f"Upgrade Strategy: {upgrade_strategy.name}")
    if upgrade_strategy != UpgradeStrategy.NO_UPGRADE:
        print(f"Upgrade Time: {upgrade_time} seconds")
        print(f"Required Blocks: {upgrade_required_blocks}")
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
    
    # Run experiments with all upgrade strategies
    for model in models:
        for attn_backend in attn_backends:
            for bs in batch_sizes:
                # Run with overlap upgrade

                # run_benchmark(
                #     model=model,
                #     attn_backend=attn_backend,
                #     batch_size=bs,
                #     upgrade_time=None,
                #     base_output_dir="logs/figure_7",
                #     upgrade_strategy=UpgradeStrategy.NO_UPGRADE
                # )
                
                # Run with basic upgrade (no overlap)
                # run_benchmark(
                #     model=model,
                #     attn_backend=attn_backend,
                #     batch_size=bs,
                #     upgrade_time=40,
                #     base_output_dir="logs/figure_7",
                #     upgrade_strategy=UpgradeStrategy.BASIC_UPGRADE,
                #     upgrade_required_blocks=20
                # )
                
                # # Run with overlap upgrade
                run_benchmark(
                    model=model,
                    attn_backend=attn_backend,
                    batch_size=bs,
                    upgrade_time=40,
                    base_output_dir="logs/figure_7",
                    upgrade_strategy=UpgradeStrategy.DECODE_UPGRADE,
                    upgrade_required_blocks=20
                )

if __name__ == "__main__":
    main()