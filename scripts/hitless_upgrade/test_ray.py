import ray
import logging
from ray.util import get_current_placement_group

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Target GPUs
TARGET_GPUS = [2, 3]

@ray.remote(num_gpus=1)
def task_on_gpu():
    # Get GPU IDs assigned to this task
    gpu_ids = ray.get_gpu_ids()
    
    # Get Ray's node ID
    node_id = ray.get_runtime_context().get_node_id()
    
    # Try to get CUDA info
    cuda_info = {}
    try:
        import torch
        if torch.cuda.is_available():
            cuda_info = {
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0)
            }
    except ImportError:
        pass
    
    return {
        "gpu_ids": gpu_ids,
        "node_id": node_id,
        "cuda_info": cuda_info
    }

def main():
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Get cluster resources
    cluster_resources = ray.cluster_resources()
    logger.info(f"Cluster resources: {cluster_resources}")
    
    # Get current node IP
    node_ip = ray.util.get_node_ip_address()
    logger.info(f"Current node IP: {node_ip}")
    
    # Create a simple placement group
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(4)]
    
    # Use your exact node IP from the logs
    exact_node_ip = "158.130.4.20"  # Use the IP from your logs
    bundles[0][f"node:{exact_node_ip}"] = 0.001
    
    logger.info(f"Creating placement group with bundles: {bundles}")
    pg = ray.util.placement_group(bundles, strategy="PACK")
    
    # Wait for placement group to be ready
    logger.info("Waiting for placement group to be ready...")
    ray.get(pg.ready())
    
    # Check placement group
    pg_table = ray.util.placement_group_table(pg)
    logger.info(f"Placement group: {pg_table}")
    
    # Run tasks on the placement group
    results = []
    for i in range(len(TARGET_GPUS)):
        result = task_on_gpu.options(
            scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=TARGET_GPUS[i]
            )
        ).remote()
        results.append(result)
    

    TARGET_GPUS_new = [0, 1]
    for i in range(len(TARGET_GPUS_new)):
        result = task_on_gpu.options(
            scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=TARGET_GPUS_new[i]
            )
        ).remote()
        results.append(result)
    
    
    # Get results
    task_results = ray.get(results)
    for i, result in enumerate(task_results):
        logger.info(f"Task {i} result: {result}")
    
    # Clean up
    ray.shutdown()

if __name__ == "__main__":
    main()