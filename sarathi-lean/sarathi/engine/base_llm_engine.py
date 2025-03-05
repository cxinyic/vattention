import copy
import math
import time
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

from sarathi.config import (
    BaseSchedulerConfig,
    CacheConfig,
    MetricsConfig,
    ModelConfig,
    ParallelConfig,
)
from sarathi.core.datatypes.request_output import RequestOutput
from sarathi.core.datatypes.sampling_params import SamplingParams
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import SamplerOutputs, Sequence, SequenceMetadata
from sarathi.core.scheduler.scheduler_registry import SchedulerRegistry
from sarathi.core.sequence_manager.engine_sequence_manager import EngineSequenceManager
from sarathi.engine.ray_utils import RayWorker, initialize_cluster, ray
from sarathi.logger import init_logger
from sarathi.metrics.constants import CpuOperationMetrics
from sarathi.metrics.cpu_timer import CpuTimer
from sarathi.metrics.metrics_store import MetricsStore
from sarathi.transformers_utils.tokenizer import get_tokenizer
from sarathi.utils import Counter, get_ip, get_random_port, unset_cuda_visible_devices
from sarathi.model_executor.attention import AttentionBackend
from sarathi.core.block_space_manager.vattention_block_space_manager import (
    vAttentionBlockSpaceManager
)
from sarathi.config import UpgradeConfig, UpgradeStrategy
import torch

logger = init_logger(__name__)

_MAX_WORKER_CONCURRENCY = 3

ModelParallelRank = Tuple[int, int]

ENGINE_GPU_ALLOCATION = {
    "old": 0.51,
    "new": 0.49
}

class BaseLLMEngine:
    """An LLM engine that receives requests and generates texts.

    This is the main class for the Sarathi engine. It receives requests
    from clients and generates texts from the LLM. It includes a tokenizer, a
    language model (possibly distributed across multiple GPUs), and GPU memory
    space allocated for intermediate states (aka KV cache). This class utilizes
    iteration-level scheduling and efficient memory management to maximize the
    serving throughput.

    NOTE: The config arguments are derived from the `EngineArgs` class. For the
    comprehensive list of arguments, see `EngineArgs`.

    Args:
        model_config: The configuration related to the LLM model.
        cache_config: The configuration related to the KV cache memory
            management.
        parallel_config: The configuration related to distributed execution.
        scheduler_config: The configuration related to the request scheduler.
        metrics_config: The configuration related to metrics store.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: BaseSchedulerConfig,
        metrics_config: MetricsConfig,
        upgrade_config: UpgradeConfig,  # Changed from upgrade_engine_type to upgrade_config
    ) -> None:
        
        logger.info(
            "Initializing an LLM engine with config: "
            f"model={model_config.model!r}, "
            f"tokenizer={model_config.tokenizer!r}, "
            f"tokenizer_mode={model_config.tokenizer_mode}, "
            f"revision={model_config.revision}, "
            f"trust_remote_code={model_config.trust_remote_code}, "
            f"dtype={model_config.dtype}, "
            f"download_dir={model_config.download_dir!r}, "
            f"load_format={model_config.load_format}, "
            f"tensor_parallel_size={parallel_config.tensor_parallel_size}, "
            f"pipeline_parallel_size={parallel_config.pipeline_parallel_size}, "
            f"seed={model_config.seed}, "
            f"attention_backend={model_config.attention_backend})"
        )
        # TODO(woosuk): Print more configs in debug mode.

        self.model_config = model_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.metrics_config = metrics_config
        self.upgrade_config = upgrade_config  # Store the entire upgrade config
        self._verify_args()

        # Extract engine_type from upgrade_config
        self.upgrade_engine_type = upgrade_config.engine_type
        logger.info(f"Engine upgrade type: {self.upgrade_engine_type}, upgrade strategy: {self.upgrade_config.strategy}")
        # For the basic upgrade strategy, the new engine is the same with the old engine
        if self.upgrade_config.serving_strategy == UpgradeStrategy.ServingStrategy.NO_SERVE and self.upgrade_engine_type == "new":
            self.upgrade_engine_type = "old"
        assert self.upgrade_engine_type in ["old", "new"], (
            f"Engine upgrade type must be 'old' or 'new', got {self.upgrade_engine_type}"
        )

        # only used for upgrade mode
        self.preempted_sequences = []

        self.tokenizer = get_tokenizer(
            model_config.tokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
            trust_remote_code=model_config.trust_remote_code,
            revision=model_config.revision,
        )

        self.seq_manager = EngineSequenceManager(self.tokenizer)
        self.seq_counter = Counter()

        self.metrics_store = MetricsStore(metrics_config)

        self.worker_map: Dict[ModelParallelRank, int] = {}

        # Initialize the cluster.
        initialize_cluster()

        # Create the parallel GPU workers.
        self._init_workers_ray()
        
        # Profile the memory usage and initialize the cache.
        # self._init_cache()
        # # Initialize the worker map.
        # self._init_worker_map()

        # self.mark_initial_memory_profiling_done()

        # # Create the scheduler.
        # self.scheduler = SchedulerRegistry.get(
        #     scheduler_config.type, scheduler_config, cache_config
        # )
        # self.scheduler.set_block_manager(model_config)
        

        # self._scheduler_timer = CpuTimer(CpuOperationMetrics.SCHEDULE)
        # self._process_model_outputs_timer = CpuTimer(
        #     CpuOperationMetrics.PROCESS_MODEL_OUTPUTS
        # )
    
    def init_rest(self):
        # self._init_workers_ray()
        # Profile the memory usage and initialize the cache.
        self._init_cache()
        # Initialize the worker map.
        self._init_worker_map()

        self.mark_initial_memory_profiling_done()

        # Create the scheduler.
        self.scheduler = SchedulerRegistry.get(
            self.scheduler_config.type, self.scheduler_config, self.cache_config
        )
        self.scheduler.set_block_manager(self.model_config)
        

        self._scheduler_timer = CpuTimer(CpuOperationMetrics.SCHEDULE)
        self._process_model_outputs_timer = CpuTimer(
            CpuOperationMetrics.PROCESS_MODEL_OUTPUTS
        )
    def _validate_parallel_config(self) -> None:
        assert self.parallel_config.pipeline_parallel_size == 1

    def _get_worker_impl(self):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from sarathi.worker.base_worker import (
            BaseWorker,  # pylint: disable=import-outside-toplevel
        )

        return BaseWorker

    def _init_workers_ray(self, **ray_remote_kwargs):
        replica_resource_mapping = self.parallel_config.replica_resource_mapping
        logger.info(
            f"Starting workers with resource mapping: {replica_resource_mapping}"
        )

        self.workers: List[RayWorker] = []

        unset_cuda_visible_devices()

        # Get GPU allocation based on engine type
        gpu_allocation = ENGINE_GPU_ALLOCATION[self.upgrade_engine_type]
        logger.info(f"Initializing engine with GPU allocation: {gpu_allocation}")


        driver_ip = None
        for rank, (node_ip, _) in enumerate(replica_resource_mapping):
            if self.upgrade_engine_type == "new":
                if self.upgrade_config.is_gpu_expansion:
                    if rank >= self.upgrade_config.original_gpu_count:
                        logger.info(f"Rank {rank} is a new GPU, setting GPU allocation to 0.51")
                        gpu_allocation = 0.51
            worker_class = ray.remote(
                # num_cpus=1,
                num_gpus=gpu_allocation, # we don't use ray for managing GPUs
                **ray_remote_kwargs,
            )(RayWorker)

            if node_ip:
                worker_class = worker_class.options(
                    max_concurrency=_MAX_WORKER_CONCURRENCY,
                    resources={
                        node_ip: 0.01,
                    },
                )
            else:
                worker_class = worker_class.options(
                    max_concurrency=_MAX_WORKER_CONCURRENCY,
                )

            if rank == 0:
                if node_ip:
                    # remove node: prefix
                    driver_ip = node_ip.split(":")[1]
                else:
                    driver_ip = get_ip()

            worker = worker_class.remote(self.model_config.trust_remote_code)

            self.workers.append(worker)

        # TODO(amey): Use a more robust method to initialize the workers.
        # In case port is already in use, this will fail.
        distributed_init_method = f"tcp://{driver_ip}:{get_random_port()}"

        logger.info(
            f"Initializing workers with distributed init method: {distributed_init_method}"
        )

        # Initialize torch distributed process group for the workers.
        model_config = copy.deepcopy(self.model_config)
        parallel_config = copy.deepcopy(self.parallel_config)
        scheduler_config = copy.deepcopy(self.scheduler_config)
        cache_config = copy.deepcopy(self.cache_config)
        metrics_config = self.metrics_store.get_config_for_worker()
        
        worker_impl = self._get_worker_impl()

        for rank, worker in enumerate(self.workers):
            local_rank = replica_resource_mapping[rank][1]
            promise = worker.init_worker.remote(
                lambda rank=rank, local_rank=local_rank: worker_impl(
                    model_config,
                    parallel_config,
                    scheduler_config,
                    cache_config,
                    metrics_config,
                    local_rank,
                    rank,
                    distributed_init_method,
                )
            )
            ray.get(promise)

        self._run_workers(
            "init_model",
            get_all_outputs=True,
        )

    def _verify_args(self) -> None:
        self._validate_parallel_config()
        self.model_config.verify_with_parallel_config(self.parallel_config)

    def _init_cache(self) -> None:
        """Profiles the memory usage and initializes the KV cache."""
        # Get the maximum number of blocks that can be allocated on GPU.
        output_all = self._run_workers(
            "profile_num_available_blocks",
            get_all_outputs=True,
            block_size=self.cache_config.block_size,
            gpu_memory_utilization=self.cache_config.gpu_memory_utilization,
        )
        
        # exit(0)
        num_gpu_blocks_across_workers, physical_memory_all = map(list, zip(*output_all))

        # Since we use a shared centralized controller, we take the minimum
        # number of blocks across all workers to make sure all the memory
        # operators can be applied to all workers.
        num_gpu_blocks = min(num_gpu_blocks_across_workers)
        physical_memory = min(physical_memory_all)

        # FIXME(woosuk): Change to debug log.
        logger.info(f"# GPU blocks: {num_gpu_blocks}")

        if num_gpu_blocks <= 0:
            raise ValueError(
                "No available memory for the cache blocks. "
                "Try increasing `gpu_memory_utilization` when "
                "initializing the engine."
            )
        max_blocks_per_request = math.ceil(
            self.model_config.max_model_len / self.cache_config.block_size
        )
        if num_gpu_blocks < max_blocks_per_request:
            raise ValueError(
                f"Not enough available memory to schedule a request will maximum allowed length {self.model_config.max_model_len}. "
                f"Need {max_blocks_per_request}, available {num_gpu_blocks} gpu blocks. "
                f"Try decreasing `max_batch_size`, `max_model_len`."
            )
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.memory_for_gpu = physical_memory
        # Initialize the cache.
        self._run_workers(
            "init_cache_engine", cache_config=self.cache_config, get_all_outputs=True
        )
        # self.scheduler.block_manager.set_cache_engine(outputs[0])   

    def _init_worker_map(self) -> None:
        model_parallel_ranks = self._run_workers(
            "get_model_parallel_ranks",
            get_all_outputs=True,
        )

        self.worker_map = {mp_rank: i for i, mp_rank in enumerate(model_parallel_ranks)}

    def _on_step_completed(
        self,
        scheduler_outputs: SchedulerOutputs,
        ignored_seqs: List[SequenceMetadata],
        seq_metadata_list: List[SequenceMetadata],
        sampler_outputs: Optional[SamplerOutputs],
        start_time: float,
    ) -> List[RequestOutput]:
        with self._process_model_outputs_timer:
            self.seq_manager.on_step_completed(
                scheduler_outputs,
                sampler_outputs,
            )
            self.scheduler.on_step_completed()

        end_time = time.perf_counter()

        self.metrics_store.on_batch_end(
            seq_metadata_list=seq_metadata_list,
            scheduler_outputs=scheduler_outputs,
            batch_start_time=start_time,
            batch_end_time=end_time,
        )
        all_request_outputs = self.seq_manager.generate_request_outputs(
            ignored_seqs, seq_metadata_list
        )
        return all_request_outputs

    def add_request(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        seq_id: Optional[Union[str, int]] = None,
    ) -> None:
        """Add a request to the engine's request pool.

        The request is added to the request pool and will be processed by the
        scheduler as `engine.step()` is called. The exact scheduling policy is
        determined by the scheduler.

        Args:
            seq_id: The unique ID of the request.
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters for text generation.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            arrival_time: The arrival time of the request. If None, we use
                the current time.
        """
        if arrival_time is None:
            arrival_time = time.monotonic()

        if prompt_token_ids is None:
            assert prompt is not None
            prompt_token_ids = self.tokenizer.encode(prompt)

        # Create the sequences.
        block_size = self.cache_config.block_size
        eos_token_id = self.tokenizer.eos_token_id
        if seq_id is None:
            seq_id = next(self.seq_counter)
        seq = Sequence(
            seq_id,
            prompt,
            prompt_token_ids,
            block_size,
            eos_token_id,
            arrival_time,
            sampling_params,
        )
        # Add the sequence to the scheduler.
        self.seq_manager.add_seq(seq)
        self._run_workers(
            "add_seq",
            seq=seq,
        )
        self.scheduler.add_seq(seq)
        self.metrics_store.on_request_arrival(seq)

    def get_model_config(self) -> ModelConfig:
        """Gets the model configuration."""
        return self.model_config

    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return self.scheduler.get_num_unfinished_seqs()

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return self.scheduler.has_unfinished_seqs()

    def step(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration.
        Then, it executes the model and updates the scheduler with the model outputs.
        Finally, it decodes the sequences and returns the newly generated results.
        """
        outputs = self._run_workers("get_free_blocks" ,get_all_outputs=True)
        if type(self.scheduler.block_manager)==vAttentionBlockSpaceManager:
            if len(self.scheduler.block_manager.preemption_queue)>0:
                preemption_queue = self.scheduler.block_manager.preemption_queue
                self.scheduler.block_manager.preemption_queue = []
            else:
                preemption_queue = []
        else:
            preemption_queue = []
        self.scheduler.block_manager.set_free_blocks(min(outputs))
        start_time = time.perf_counter()
        with self._scheduler_timer:
            scheduler_outputs = self.scheduler.schedule()
        if scheduler_outputs.is_empty():
            return []

        ignored_seqs, seq_metadata_list = self.seq_manager.on_schedule(
            scheduler_outputs
        )

        sampler_outputs = self._run_workers(
            "execute_model",
            scheduler_outputs=scheduler_outputs,
            preempted_seq=preemption_queue,
        )
        # self.scheduler.block_manager.reset_free_blocks()
        # sampler_outputs, num_free_blocks = zip(*sampler_outputs)
        # self.scheduler.block_manager.set_free_blocks(min(num_free_blocks))
        return self._on_step_completed(
            scheduler_outputs,
            ignored_seqs,
            seq_metadata_list,
            sampler_outputs,
            start_time,
        )

    def _run_workers(
        self,
        method: str,
        *args,
        get_all_outputs: bool = False,
        ignore_output: bool = False,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        all_outputs = []
        for worker in self.workers:
            executor = partial(worker.execute_method.remote, method)

            output = executor(*args, **kwargs)
            all_outputs.append(output)

        if ignore_output:
            return

        all_outputs = ray.get(all_outputs)

        if get_all_outputs:
            return all_outputs

        # Make sure all workers have the same results.
        output = all_outputs[0]
        for other_output in all_outputs[1:]:
            assert output == other_output
        return output

    def _run_worker(
        self,
        model_parallel_rank: ModelParallelRank,
        method: str,
        *args,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        worker = self.workers[self.worker_map[model_parallel_rank]]
        executor = partial(worker.execute_method.remote, method)

        output = executor(*args, **kwargs)

        while True:
            try:
                output = ray.get(output, timeout=0)
                break
            except ray.exceptions.GetTimeoutError:
                time.sleep(0.005)
                continue

        return output

    def plot_metrics(self) -> None:
        self.metrics_store.plot()

    def pull_worker_metrics(self) -> None:
        worker_metrics = self._run_workers(
            "get_metrics_store",
            get_all_outputs=True,
        )
        for worker_metric in worker_metrics:
            self.metrics_store.merge(worker_metric)

    def mark_initial_memory_profiling_done(self):
        self.metrics_store.mark_initial_memory_profiling_done()
        self._run_workers("mark_initial_memory_profiling_done", get_all_outputs=True)

    def reset_metrics(self) -> None:
        self.scheduler.reset_state()
        self.metrics_store.reset()
        self._run_workers("reset_metrics", get_all_outputs=True)

    def start_profiling(self) -> None:
        self._run_workers("start_profiling")

    def stop_profiling(self) -> None:
        self._run_workers("stop_profiling")

    def get_metric_store(self) -> MetricsStore:
        return self.metrics_store

    def cleanup(self) -> None:
        self._run_workers("cleanup")
    
    def prepare_for_upgrade(self):
        """
        Prepare for upgrade by selecting and preempting sequences.
        """
        required_blocks = self.upgrade_config.required_blocks

        if self.upgrade_engine_type != "old":
            logger.warning(f"prepare_for_upgrade called on non-old engine")
            return

        logger.info(f"Preparing for upgrade, need to free {required_blocks} blocks")

        # 1. Tell block manager to select sequences to preempt
        if type(self.scheduler.block_manager) == vAttentionBlockSpaceManager:
            # The only difference between decode and prefill is the preemption_mode
            preemption_mode = "partial" if self.upgrade_config.serving_strategy == UpgradeStrategy.ServingStrategy.DECODE_ONLY else "full"
            free_blocks_to_use, sequences_for_physical_free, sequences_for_virtual_free = self.scheduler.select_preemption_sequences(required_blocks, preemption_mode, self.upgrade_config.selection_policy)
            
            logger.info(f"Using {free_blocks_to_use} free blocks and preempting {len(sequences_for_physical_free)} sequences")
        else:
            logger.error("Incorrect block manager type for upgrade")
            return
        
        # Handle free blocks if available
        if free_blocks_to_use > 0:
            nr_physical_blocks = free_blocks_to_use * self.upgrade_config.pages_per_block
            logger.info(f"Releasing {nr_physical_blocks} physical blocks from free pool")
            self._run_workers(
                "release_empty_kv",
                nr_physical_blocks=int(nr_physical_blocks),
                get_all_outputs=True
            )
        
        # Handle preempted sequences if available
        if sequences_for_physical_free or sequences_for_virtual_free:
            # For decode upgrade, sequences_for_virtual_free is empty (initialized as [] in the original)
            sequences_va = sequences_for_virtual_free if self.upgrade_config.serving_strategy == UpgradeStrategy.ServingStrategy.PREFILL_ONLY else []
            
            logger.info(f"Releasing KV memory for {len(sequences_for_physical_free)} physical and {len(sequences_va)} virtual sequences")
            self._run_workers(
                "release_sequences_kv",
                sequences_PA=sequences_for_physical_free,
                sequences_VA=sequences_va,
                get_all_outputs=True
            )
            logger.info("Unmapped sequences from vattention")