import os
import time
from typing import Dict, Any

from sarathi.benchmark.config import Config
from sarathi.benchmark.benchmark_runner import BenchmarkRunnerLauncher

def create_config(
    model: str,
    batch_size: int,
    attn_backend: str,
    output_dir: str,
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
        "cluster_num_replicas": 1,  # Added this
        
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
    }
    
    return Config(args)

def run_benchmark(
    model: str = "01-ai/Yi-6B-200k",
    attn_backend: str = "fa_vattn",
    batch_size: int = 8,
    upgrade_time: float = 10,  
    base_output_dir: str = "logs/figure_7",
) -> None:
    """Run benchmark with upgrade capability"""
    
    # Create output directory
    output_dir = os.path.join(
        base_output_dir,
        f"model_{model}_bs_{batch_size}_attn_{attn_backend}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Create initial and new configs (same configuration for now)
    initial_config = create_config(
        model=model,
        batch_size=batch_size,
        attn_backend=attn_backend,
        output_dir=output_dir,
        tp_degree=4,
        pp_degree=1
    )
    
    # For now, using same config for upgrade
    new_config = create_config(
        model=model,
        batch_size=batch_size,
        attn_backend=attn_backend,
        output_dir=output_dir,
        tp_degree=4,
        pp_degree=1
    )
    
    print("\n=====================================================================================")
    print(f"Running Config ==> Model: {model} Batch Size: {batch_size} Attention Backend: {attn_backend}")
    print(f"Upgrade Time: {upgrade_time} seconds")
    print("======================================================================================\n")
    
    # Create and run benchmark
    launcher = BenchmarkRunnerLauncher(
        config=initial_config,
        upgrade_time=upgrade_time,
        new_config=new_config
    )
    
    launcher.run_with_upgrade()

def main():
    """Main function to run benchmarks"""
    # Configuration variables
    # models = ["01-ai/Yi-6B-200k"]  # Can be expanded to ["yi-6b", "llama-3-8b", "yi-34b"]
    models = ["01-ai/Yi-Coder-1.5B"]
    attn_backends = ["fa_vattn"]  # Can be expanded to ["fa_paged", "fi_paged", "fa_vattn"]
    batch_sizes = [32]  # Can be expanded to [1, 2, 4, 8, 12, 16, 32]
    
    # Run experiments
    for model in models:
        for attn_backend in attn_backends:
            for bs in batch_sizes:
                run_benchmark(
                    model=model,
                    attn_backend=attn_backend,
                    batch_size=bs,
                    upgrade_time=20,  
                    base_output_dir="logs/figure_7"
                )

if __name__ == "__main__":
    main()