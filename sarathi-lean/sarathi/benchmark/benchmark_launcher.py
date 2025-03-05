"""
BenchmarkRunnerLauncher for coordinating benchmark execution across multiple replicas.
"""

import json
import logging
import threading
import time
from typing import Dict, Any

import ray
import wandb
import torch

from sarathi import LLMEngine
from sarathi.benchmark.config import Config
from sarathi.benchmark.benchmark_runner import BenchmarkRunner
from sarathi.benchmark.custom_types import ReplicaResourceMapping
from sarathi.benchmark.upgrade_utils.block_calculation import calculate_required_blocks
from sarathi.benchmark.upgrade_utils.gpu_memory_info import log_memory_usage
from sarathi.benchmark.upgrade_utils.upgrade_state import UpgradeState
from sarathi.metrics.metrics_store import MetricsStore
from sarathi.utils import get_ip

from sarathi.config import (
    MetricsConfig,
    UpgradeStrategy,
)

logger = logging.getLogger(__name__)

class BenchmarkRunnerLauncher:
    """
    Launcher for coordinating benchmark runners across multiple replicas.
    
    This class is responsible for initializing and managing BenchmarkRunner
    instances, handling resource allocation, and coordinating the upgrade process.
    """
    
    def __init__(
        self, 
        config: Config, 
        new_config: Config = None,
    ) -> None:
        """
        Initialize the benchmark launcher.
        
        Args:
            config: Main benchmark configuration
            new_config: Configuration for the upgraded model (optional)
        """
        self._config = config
        self._new_config = new_config
        self._is_multi_replica = self._config.cluster_num_replicas > 1

        ray.init(ignore_reinit_error=True)
        required_blocks, pages_per_block = self.calculate_upgrade_blocks()
        self._config.upgrade_required_blocks = required_blocks
        self._config.pages_per_block = pages_per_block
        logger.info(f"Required blocks for upgrade: {required_blocks}")
        
        # Initialize based on multi-replica setting
        if self._is_multi_replica:
            self._validate_cluster_resources()
            self._runners = self._create_runners()
            self._aggregate_metric_store = self._create_aggregate_metric_store()
        else:
            self.replica_resource_mapping = self._get_replica_resource_mapping()
            assert len(self.replica_resource_mapping) == 1
            self._runner = BenchmarkRunner(
                0, 
                self._config, 
                self.replica_resource_mapping["0"]
            )

        if wandb.run is not None:
            wandb.config.update(self._config.__dict__)
    
    def calculate_upgrade_blocks(self) -> tuple:
        """
        Calculate required blocks for model upgrade using engine configs.
        
        Returns:
            Tuple of (required_blocks, pages_per_block)
        """
        if not self._new_config:
            # If no new config provided, return default values
            return 0, 0

        # Get engine configs from current config
        current_configs = LLMEngine.get_engine_configs(
            model=self._config.model_name,
            tokenizer=self._config.model_name,
            tensor_parallel_size=self._config.model_tensor_parallel_degree,
            pipeline_parallel_size=self._config.model_pipeline_parallel_degree,
            dtype="float16",
            trust_remote_code=True
        )
        
        # Get engine configs from new config
        new_configs = LLMEngine.get_engine_configs(
            model=self._new_config.model_name,
            tokenizer=self._new_config.model_name,
            tensor_parallel_size=self._new_config.model_tensor_parallel_degree,
            pipeline_parallel_size=self._new_config.model_pipeline_parallel_degree,
            dtype="float16",
            trust_remote_code=True
        )

        # Extract model and parallel configs
        model_config = current_configs[0]  # First element is ModelConfig
        parallel_config = current_configs[2]  # Third element is ParallelConfig
        new_parallel_config = new_configs[2]  # Third element from new configs
        
        required_blocks, pages_per_block = calculate_required_blocks(
            model_config=model_config,
            parallel_config=parallel_config,
            new_parallel_config=new_parallel_config,
            attention_backend=self._config.model_attention_backend
        )
        
        logger.info(f"Calculated required blocks for upgrade: {required_blocks}")
        return required_blocks, pages_per_block
    
    def _validate_cluster_resources(self):
        """Validate that sufficient GPU resources are available in the cluster."""
        num_replicas = self._config.cluster_num_replicas
        tp_degree = self._config.model_tensor_parallel_degree
        pp_degree = self._config.model_pipeline_parallel_degree
        num_gpus_required = num_replicas * tp_degree * pp_degree

        available_resources = ray.available_resources()

        assert (
            available_resources["GPU"] >= num_gpus_required
        ), f"Insufficient GPUs. Required: {num_gpus_required}, Available: {available_resources['GPU']}"
        
    def _create_runners(self):
        """
        Create Ray actors for each benchmark runner.
        
        Returns:
            List of BenchmarkRunner Ray actors
        """
        assert (
            self._config.model_tensor_parallel_degree > 1
            or self._config.model_pipeline_parallel_degree > 1
        )

        replica_resource_mapping = self._get_replica_resource_mapping()
        runner_class = ray.remote(num_cpus=1)(BenchmarkRunner)
        runners = []

        for replica_id in range(self._config.cluster_num_replicas):
            runners.append(
                runner_class.options(
                    resources={
                        replica_resource_mapping[str(replica_id)][0][0]: 0.01,
                    },
                ).remote(
                    replica_id, 
                    self._config, 
                    replica_resource_mapping[str(replica_id)],
                )
            )

        return runners
    
    def _create_aggregate_metric_store(self):
        """
        Create an aggregate metrics store for multi-replica setups.
        
        Returns:
            MetricsStore instance for aggregating metrics
        """
        metric_config = MetricsConfig(
            replica_id=0,  # dummy replica id
            write_metrics=self._config.write_metrics,
            output_dir=self._config.output_dir,
            wandb_project=self._config.metrics_store_wandb_project,
            wandb_group=self._config.metrics_store_wandb_group,
            wandb_run_name=self._config.metrics_store_wandb_run_name,
            enable_op_level_metrics=self._config.metrics_store_enable_op_level_metrics,
            enable_cpu_op_level_metrics=self._config.metrics_store_enable_cpu_op_level_metrics,
            enable_chrome_trace=self._config.write_chrome_trace,
            enable_request_outputs=self._config.metrics_store_enable_request_outputs,
            keep_individual_batch_metrics=self._config.metrics_store_keep_individual_batch_metrics,
        )
        metrics_store = MetricsStore(metric_config)
        metrics_store.mark_initial_memory_profiling_done()

        return metrics_store
        
    def _get_replica_resource_mapping(self) -> ReplicaResourceMapping:
        """
        Get the mapping of replicas to resources.
        
        Returns:
            Dictionary mapping replica IDs to resource lists
        """
        if self._config.replica_resource_mapping:
            replica_resource_mapping = json.loads(self._config.replica_resource_mapping)
            logger.info(f"Replica resource mapping: {replica_resource_mapping}")
            return replica_resource_mapping

        cluster_resources_keys = list(ray.available_resources().keys())
        num_gpus = ray.available_resources()["GPU"]
        logger.info(f"Cluster resources num_gpus: {num_gpus}")
        ip_addresses = [
            x
            for x in cluster_resources_keys
            if x.startswith("node:") and x != "node:__internal_head__"
        ]

        runner_ip = f"node:{get_ip()}"
        logger.info(f"Runner IP: {runner_ip}, ip_addresses: {ip_addresses}")
        ip_addresses.remove(runner_ip)
        ip_addresses.insert(0, runner_ip)

        num_nodes = len(ip_addresses)
        assert num_nodes > 0, "No nodes found in the cluster"
        assert num_gpus > 0, "No GPUs found in the cluster"
        assert (
            num_gpus % num_nodes == 0
        ), f"Number of GPUs ({num_gpus}) is not a multiple of number of nodes ({num_nodes})"
        num_gpus_per_node = int(num_gpus // num_nodes)
        num_replicas = self._config.cluster_num_replicas
        num_gpus_per_replica = (
            self._config.model_tensor_parallel_degree
            * self._config.model_pipeline_parallel_degree
        )

        assert (
            num_gpus >= num_replicas * num_gpus_per_replica
        ), f"Insufficient GPUs. Required: {num_replicas * num_gpus_per_replica}, Available: {num_gpus}"

        replica_resource_mapping = {}

        available_gpus = []
        for ip_address in ip_addresses:
            for gpu_id in (range(num_gpus_per_node)):
                available_gpus.append((ip_address, gpu_id))
        logger.info(f"Available GPUs: {available_gpus}")
        
        for replica_id in range(num_replicas):
            replica_resource_mapping[str(replica_id)] = []
            for _ in range(num_gpus_per_replica):
                replica_resource_mapping[str(replica_id)].append(available_gpus.pop(0))

        logger.info(f"Replica resource mapping: {replica_resource_mapping}")

        return replica_resource_mapping


    def _run_with_overlap_upgrade(self):
        """Run single replica upgrade with overlap serving."""
        upgrade_state = UpgradeState()
        new_runner = None
        saved_progress = None

        def init_new_runner():
            nonlocal new_runner
            # Calculate new resource mapping based on new config
            new_resource_mapping = self._get_replica_resource_mapping_for_config(self._new_config)
            
            new_runner = BenchmarkRunner(
                0, 
                self._new_config, 
                new_resource_mapping["0"],
                is_new_runner=True
            )
            upgrade_state.set_weights_loaded()
            # logger.info(f"New runner initialized with resources: {new_resource_mapping['0']}")

        # Step 1: Set up upgrade state and start original runner
        self._runner.set_upgrade_state(upgrade_state)
        result = self._runner.run()

        if isinstance(result, dict) and result["status"] == "UPGRADE_NEEDED":
            # Start weight loading in background
            init_thread = threading.Thread(target=init_new_runner)
            init_thread.start()
            
            # Continue serving with reduced capacity during weight loading
            result = self._runner.run_during_overlap()
            
            if isinstance(result, dict) and result["status"] == "READY_FOR_HANDOVER":
                old_workers = self._runner._llm_engine.workers if hasattr(self._runner._llm_engine, 'workers') else []

                if self._runner.is_pipeline_engine:
                    logger.info("Stopping pipeline engine execution loops")
                    self._runner._llm_engine.stop_execution_loops()
                self._runner._llm_engine.cleanup()
                del self._runner

                saved_progress = result["progress"]
                # Wait for weight loading to complete
                init_thread.join()

                if old_workers:
                    logger.info(f"Cleaning up {len(old_workers)} old Ray workers")
                    for worker in old_workers:
                        try:
                            ray.kill(worker)
                        except Exception as e:
                            logger.error(f"Error cleaning up old worker: {e}")

                # Force CUDA cleanup
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()
                log_memory_usage("AFTER FORCE CLEANUP")

                if new_runner is not None:
                    new_runner._llm_engine.init_rest()
                    track = result["tracker"]
                    new_runner.load_progress(saved_progress, track)
                    final_result = new_runner.run()
                    logger.info(f"New runner completed with status: {final_result}")
                else:
                    logger.error("New runner initialization failed")
        wandb.finish()
    
    def _run_without_overlap_upgrade(self):
        """Run single replica upgrade without overlap serving."""
        result = self._runner.run()
        
        if isinstance(result, dict) and result["status"] == "UPGRADE_NEEDED":
            progress = result["progress"]
            track = result["tracker"]
            
            # Clean up Ray resources
            ray.shutdown()
            ray.init(ignore_reinit_error=True)
            
            # Calculate new resource mapping based on new config
            new_resource_mapping = self._get_replica_resource_mapping_for_config(self._new_config)
            
            # Create new runner with the new resource mapping
            new_runner = BenchmarkRunner(
                0, 
                self._new_config, 
                new_resource_mapping["0"],
                is_new_runner=True
            )
            new_runner.load_progress(progress, track)
            
            # Run second phase
            new_runner.run()
        wandb.finish()
	
    def _run_normal(self):
        """Run without any upgrade."""
        result = self._runner.run()
        wandb.finish()
        return result
    
    def run(self):
        self._run_normal()
    
    def run_with_upgrade(self):
        """Run benchmark with configurable upgrade strategy."""
        if not self._is_multi_replica:
            if self._config.upgrade_serving_strategy == UpgradeStrategy.ServingStrategy.DECODE_ONLY or self._config.upgrade_serving_strategy == UpgradeStrategy.ServingStrategy.PREFILL_ONLY:
                return self._run_with_overlap_upgrade()
            elif self._config.upgrade_serving_strategy == UpgradeStrategy.ServingStrategy.NO_SERVE:
                return self._run_without_overlap_upgrade()
            else:
                return self._run_normal()  # Just run normally for NO_UPGRADE
        else:
            # Handle multi-replica upgrade
            logger.warning("Multi-replica upgrade not yet implemented")
            # TODO: Implement multi-replica upgrade support
            pass
    
    def _get_replica_resource_mapping_for_config(self, config: Config) -> ReplicaResourceMapping:
        """
        Get resource mapping for a specific config, with support for upgrade scenarios.
        
        Args:
            config: The config to calculate resources for
            
        Returns:
            Dictionary mapping replica IDs to resource lists
        """
        #TODO(XY): assume there is only one replica and it uses exactly pp*tp GPUs
        # Use provided mapping if available
        if config.replica_resource_mapping:
            replica_resource_mapping = json.loads(config.replica_resource_mapping)
            logger.info(f"Using provided replica resource mapping: {replica_resource_mapping}")
            return replica_resource_mapping

        # Get cluster resources
        cluster_resources = ray.available_resources()
        cluster_resources_keys = list(cluster_resources.keys())
        num_gpus = cluster_resources.get("GPU", 0)
        logger.info(f"Cluster resources num_gpus: {num_gpus}")
        
        # Get node IPs
        ip_addresses = [
            x
            for x in cluster_resources_keys
            if x.startswith("node:") and x != "node:__internal_head__"
        ]

        runner_ip = f"node:{get_ip()}"
        logger.info(f"Runner IP: {runner_ip}, ip_addresses: {ip_addresses}")
        
        # Ensure the runner IP is first in the list
        if runner_ip in ip_addresses:
            ip_addresses.remove(runner_ip)
        ip_addresses.insert(0, runner_ip)

        num_nodes = len(ip_addresses)
        assert num_nodes > 0, "No nodes found in the cluster"
        assert num_gpus > 0, "No GPUs found in the cluster"
        
        # Calculate GPUs needed per replica
        num_replicas = config.cluster_num_replicas
        num_gpus_per_replica = (
            config.model_tensor_parallel_degree
            * config.model_pipeline_parallel_degree
        )
        
        # Use num_gpus_per_replica directly for physical GPUs count
        # This works because you know you have at least 4 GPUs available
        total_physical_gpus_needed = num_gpus_per_replica * num_replicas
                        
        # Create available GPU list - create as many as we need
        available_gpus = []
        for ip_address in ip_addresses:
            # Distribute GPUs evenly across nodes
            gpus_per_node = total_physical_gpus_needed // num_nodes
            if ip_address == ip_addresses[0]:  # First node gets any remainder
                gpus_per_node += total_physical_gpus_needed % num_nodes
                
            for gpu_id in range(gpus_per_node):
                available_gpus.append((ip_address, gpu_id))
        
        logger.info(f"Available GPUs for config {config.model_name}: {available_gpus}")
        
        # Assign GPUs to replicas
        replica_resource_mapping = {}
        
        for replica_id in range(num_replicas):
            replica_resource_mapping[str(replica_id)] = []
            for _ in range(num_gpus_per_replica):
                if available_gpus:
                    node_ip, gpu_id = available_gpus.pop(0)
                    # Store with the calculated GPU fraction
                    replica_resource_mapping[str(replica_id)].append((node_ip, gpu_id))
                else:
                    raise ValueError(f"Not enough GPUs available for replica {replica_id}")

        logger.info(f"Calculated replica resource mapping for {config.model_name}: {replica_resource_mapping}")
        return replica_resource_mapping