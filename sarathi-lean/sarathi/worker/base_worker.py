"""A GPU worker class."""

import os
import time
from typing import Optional, Tuple, List, Any

import torch
import torch.distributed

from sarathi.config import (
    BaseSchedulerConfig,
    CacheConfig,
    MetricsConfig,
    ModelConfig,
    ParallelConfig,
)
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import SamplerOutputs, Sequence
from sarathi.core.sequence_manager.worker_sequence_manager import WorkerSequenceManager
from sarathi.logger import init_logger
from sarathi.metrics.metrics_store import MetricsStore
from sarathi.model_executor import set_random_seed
from sarathi.model_executor.attention import set_attention_backend
from sarathi.model_executor.model_runner import ModelRunner
from sarathi.model_executor.parallel_utils.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_tensor_model_parallel_rank,
    initialize_model_parallel,
)
from sarathi.utils.threading_utils import synchronized
from sarathi.worker.cache_engine import get_cache_engine
from sarathi.worker.cache_engine import get_cache_mem_alloc_backend
import ray
from datetime import datetime
import vattention


logger = init_logger(__name__)


class BaseWorker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: BaseSchedulerConfig,
        cache_config: CacheConfig,
        metrics_config: MetricsConfig,
        local_rank: int,
        rank: Optional[int] = None,
        distributed_init_method: Optional[str] = None,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        # this is partially initialized cache config, ie. it doesn't have
        # information about the number of blocks, it will get updated after profiling
        self.cache_config = cache_config
        self.metrics_config = metrics_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.device = rank
        # Uninitialized cache engine. Will be initialized by
        # self.init_cache_engine(self.cache_config)
        self.cache_engine = None
        self.gpu_cache = None
        # Sequence manager also needs number of blocks for initialization
        self.seq_manager = None

        set_attention_backend(model_config.attention_backend)
        
        self._verify_parallel_config()
        self.metrics_store = MetricsStore(metrics_config)

        self._cpu_kvcache_manager = None
        self.cpu_cache_connected = False

    def _verify_parallel_config(self) -> None:
        assert self.parallel_config.pipeline_parallel_size == 1
    
    def connect_cpu_cache(self, cpu_cache_info) -> None:
        """Connect this worker to the global CPU cache"""
        try:
            vattention.connect_to_existing_cpu_memory(
                cpu_cache_info["address"],
                cpu_cache_info["metadata"]
            )
            self.cpu_cache_connected = True
            logger.info(f"Worker successfully connected to CPU cache at address: {cpu_cache_info['address']}")
        except Exception as e:
            logger.error(f"Failed to connect to CPU cache: {e}")
            self.cpu_cache_connected = False

    @torch.inference_mode()
    @synchronized
    def init_model(self):
        # torch.distributed.all_reduce does not free the input tensor until
        # the synchronization point. This causes the memory usage to grow
        # as the number of all_reduce calls increases. This env var disables
        # this behavior.
        # Related issue:
        # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

        # This env var set by Ray causes exceptions with graph building.
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
        ray_resources = ray.get_runtime_context().get_assigned_resources()
        logger.info(f"Worker {self.rank}  - Ray assigned resources: {ray_resources}")
        self.device = torch.device(f"cuda:0")
            
        # Try setting the device
        logger.info(f"Worker {self.rank} - Attempting to set device to: {self.rank}")
        torch.cuda.synchronize()
        torch.cuda.set_device(0)

        # Initialize the distributed environment.
        _init_distributed_environment(
            self.parallel_config, self.rank, self.distributed_init_method
        )

        self.tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        self.pipeline_model_parallel_rank = get_pipeline_model_parallel_rank()

        self.is_tensor_parallel_rank_zero = self.tensor_model_parallel_rank == 0
        self.is_first_pipeline_stage = self.pipeline_model_parallel_rank == 0
        self.is_last_pipeline_stage = (
            self.pipeline_model_parallel_rank
            == self.parallel_config.pipeline_parallel_size - 1
        )

        logger.info(
            f"Initializing worker {self.rank} on device {self.device}, "
            f"tensor parallel rank {self.tensor_model_parallel_rank} "
            f"and pipeline parallel rank {self.pipeline_model_parallel_rank}."
        )

        # Initialize the model.
        set_random_seed(self.model_config.seed)
        self.model_runner = ModelRunner(
            self.model_config,
            self.parallel_config,
            self.scheduler_config,
            self.cache_config,
            self.device,
            self.rank,
        )
        logger.info(f"Model initialized on worker {self.rank}.")

    @torch.inference_mode()
    @synchronized
    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        torch.cuda.set_device(self.device)

        self.cache_config = cache_config

        mem_alloc_backend = get_cache_mem_alloc_backend(self.model_config.attention_backend)

        self.cache_engine = get_cache_engine(self.model_config.attention_backend)(
            self.cache_config, self.model_config, self.parallel_config, mem_alloc_backend
        )
        self.gpu_cache = self.cache_engine.gpu_cache

        self.seq_manager = WorkerSequenceManager(
            self.cache_config,
            self.scheduler_config,
            self.model_config,
            self.rank,
        )
        logger.info(f"XY: Cache engine initialized on worker {self.rank}.")
        
        # return self.cache_engine
    def get_free_blocks(self) -> int:
        return self.cache_engine.num_free_blocks()
    
    def preempt_requests(self, preempted_seq: List) -> None:
        self.cache_engine.preempt_requests(preempted_seq)

    @synchronized
    def add_seq(self, seq: Sequence) -> None:
        self.seq_manager.add_seq(seq)

    @synchronized
    def get_model_parallel_ranks(self) -> Tuple[int, int]:
        return self.tensor_model_parallel_rank, self.pipeline_model_parallel_rank

    def on_step_completed(
        self, scheduler_outputs: SchedulerOutputs, sampler_outputs: SamplerOutputs
    ) -> None:
        self.seq_manager.on_step_completed(scheduler_outputs, sampler_outputs)
        

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_outputs: SchedulerOutputs,
        preempted_seq: Optional[List] = None,
    ) -> Optional[SamplerOutputs]:
        
        batch_stage_start_time = time.monotonic()
        self.seq_manager.block_manager.set_free_blocks(self.cache_engine.num_free_blocks()) 
        _, seq_metadata_list = self.seq_manager.on_schedule(scheduler_outputs)
        if preempted_seq:
            logger.info(f"Worker {self.rank} - Preempting seq_ids {[seq.seq_id for seq in preempted_seq]}")
            self.preempt_requests(preempted_seq)

        self.cache_engine.step(seq_metadata_list)

        sampler_outputs = self.model_runner.run(
            seq_metadata_list,
            self.gpu_cache,
        )
        
        # This will not be executed in the pp mode
        self.on_step_completed(scheduler_outputs, sampler_outputs)  
        
        # Only execute cache completion in non-pp case
        if self.parallel_config.pipeline_parallel_size == 1:
            self.cache_engine.on_step_completion(seq_metadata_list)
        for seq_metadata in seq_metadata_list:
            if seq_metadata.seq.is_finished():
                self.seq_manager._free_seq(seq_metadata.seq.seq_id)

        batch_stage_end_time = time.monotonic()

        self.metrics_store.on_batch_stage_end(
            seq_metadata_list,
            scheduler_outputs,
            self.tensor_model_parallel_rank,
            self.pipeline_model_parallel_rank,
            batch_stage_start_time,
            batch_stage_end_time,
        )

        return sampler_outputs #, self.cache_engine.num_free_blocks()

    @synchronized
    def get_metrics_store(self) -> MetricsStore:
        return self.metrics_store

    @synchronized
    def mark_initial_memory_profiling_done(self):
        self.metrics_store.mark_initial_memory_profiling_done()

    @synchronized
    def reset_metrics(self) -> None:
        self.metrics_store.reset()

    @synchronized
    def start_profiling(self) -> None:
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        )
        self.profiler.__enter__()

    @synchronized
    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
    ) -> Tuple[int, int]:
        return self.model_runner.profile_num_available_blocks(
            block_size, gpu_memory_utilization
        )

    @synchronized
    def stop_profiling(self) -> None:
        self.profiler.__exit__(None, None, None)
        self.profiler.export_chrome_trace(
            f"{self.metrics_config.output_dir}/profiler_trace_rank_{self.rank}.json"
        )

    @synchronized
    def cleanup(self) -> None:
        self.cache_engine.cleanup_kvcache()
    
    @synchronized
    def release_sequences_kv(self, sequences_PA: List[Sequence], sequences_VA: List[Sequence]) -> None:
        for sequence in sequences_PA:
            self.cache_engine.free_request(sequence.seq_id, force_release=True)
        for sequence in sequences_VA:
            self.cache_engine.free_request(sequence.seq_id, force_release=False)

    @synchronized
    def release_empty_kv(self, nr_physical_blocks: int) -> None:
        self.cache_engine.free_physical_blocks(nr_physical_blocks)
    
    
def _init_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
) -> None:
    """Initialize the distributed environment."""
    if torch.distributed.is_initialized():
        torch_world_size = torch.distributed.get_world_size()
        if torch_world_size != parallel_config.world_size:
            raise RuntimeError(
                "torch.distributed is already initialized but the torch world "
                "size does not match parallel_config.world_size "
                f"({torch_world_size} vs. {parallel_config.world_size})."
            )
    elif not distributed_init_method:
        raise ValueError(
            "distributed_init_method must be set if torch.distributed "
            "is not already initialized"
        )
    else:
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=parallel_config.world_size,
            rank=rank,
            init_method=distributed_init_method,
        )

    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    initialize_model_parallel(
        parallel_config.tensor_parallel_size, parallel_config.pipeline_parallel_size
    )

    

