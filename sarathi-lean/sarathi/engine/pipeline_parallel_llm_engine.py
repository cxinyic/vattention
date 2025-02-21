import time
from dataclasses import dataclass
from queue import Queue
from threading import Event, Thread
from typing import List

from sarathi.config import (
    BaseSchedulerConfig,
    CacheConfig,
    MetricsConfig,
    ModelConfig,
    ParallelConfig,
)
from sarathi.core.datatypes.request_output import RequestOutput
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import SequenceMetadata
from sarathi.engine.base_llm_engine import BaseLLMEngine
from sarathi.logger import init_logger
from sarathi.utils.threading_utils import exit_on_error
from sarathi.config import UpgradeConfig, UpgradeStrategy
from sarathi.core.block_space_manager.vattention_block_space_manager import (
    vAttentionBlockSpaceManager,
)

logger = init_logger(__name__)

SCHEDULER_LOOP_DELAY = 0.01


@dataclass
class ScheduleStageOutputs:
    ignored_seqs: List[SequenceMetadata]
    seq_metadata_list: List[SequenceMetadata]
    scheduler_outputs: SchedulerOutputs
    start_time: float


class PipelineParallelLLMEngine(BaseLLMEngine):
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
        super().__init__(
            model_config,
            cache_config,
            parallel_config,
            scheduler_config,
            metrics_config,
            upgrade_config,
        )
        # Create the request queue.
        self.has_started_execution_loops = False
        self.scheduler_output_queue = Queue()
        self.output_queue = Queue()
        self.schedule_event = Event()
        self.microbatch_watch_event = Event()
        self.schedule_thread = Thread(target=self._schedule_loop, daemon=True)
        self.microbatch_watch_thread = Thread(
            target=self._microbatch_watch_loop, daemon=True
        )
        self.output_thread = Thread(target=self._output_loop, daemon=True)
        self.scheduler_timer_thread = Thread(
            target=self._scheduler_timer_loop, daemon=True
        )
        # For upgrade mode only
        self.stop_scheduling = False
        self.inflight_batches_count = 0  # Track number of batches in pipeline
        self.should_stop = False

    def _validate_parallel_config(self) -> None:
        assert self.parallel_config.pipeline_parallel_size > 1

    def start_execution_loops(self) -> None:
        """Starts the execution loop."""
        self.has_started_execution_loops = True
        self.schedule_event.set()
        self.schedule_thread.start()
        self.output_thread.start()
        self.scheduler_timer_thread.start()
        self.microbatch_watch_thread.start()

    @exit_on_error
    def _scheduler_timer_loop(self) -> None:
        while True:
            time.sleep(SCHEDULER_LOOP_DELAY)
            self.schedule_event.set()
            if self.should_stop:
                break

    def _get_worker_impl(self):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from sarathi.worker.pipeline_parallel_worker import PipelineParallelWorker

        return PipelineParallelWorker

    @exit_on_error
    def _schedule_loop(self) -> None:
        while True:
            self.schedule_event.wait()
            self.schedule_event.clear()

            if self.should_stop:
                break
            if self.stop_scheduling:
                continue

            start_time = time.perf_counter()
            outputs = self._run_workers("get_free_blocks", get_all_outputs=True)
            if type(self.scheduler.block_manager) == vAttentionBlockSpaceManager:
                if len(self.scheduler.block_manager.preemption_queue) > 0:
                    preemption_queue = self.scheduler.block_manager.preemption_queue
                    self.scheduler.block_manager.preemption_queue = []
                else:
                    preemption_queue = []
            else:
                preemption_queue = []
            self.scheduler.block_manager.set_free_blocks(min(outputs))
            # logger.info(f"Free blocks: {outputs}")
            scheduler_outputs = self.scheduler.schedule()

            if scheduler_outputs.has_no_output():
                continue

            ignored_seqs, seq_metadata_list = self.seq_manager.on_schedule(
                scheduler_outputs
            )

            self.scheduler_output_queue.put(
                ScheduleStageOutputs(
                    ignored_seqs,
                    seq_metadata_list,
                    scheduler_outputs,
                    start_time,
                )
            )

            if not scheduler_outputs.is_empty():
                self.inflight_batches_count += 1  # New batch entering pipeline
                self.microbatch_watch_event.set()
                self._run_workers(
                    "enqueue",
                    scheduler_outputs=scheduler_outputs,
                    preempted_seq=preemption_queue,
                    ignore_output=True,
                )

            end_time = time.perf_counter()
            self.metrics_store.on_schedule(seq_metadata_list, start_time, end_time)

    @exit_on_error
    def _microbatch_watch_loop(self) -> None:
        while True:
            self.microbatch_watch_event.wait()
            self.microbatch_watch_event.clear()
            if self.should_stop:
                break

            self._run_worker(
                (0, 0),  # rank zero
                "get_output",
            )
            # logger.info("Microbatch watch event triggered")
            self.schedule_event.set()

    @exit_on_error
    def _output_loop(self) -> None:
        while True:
            if self.should_stop:
                break
            scheduler_stage_output = self.scheduler_output_queue.get()

            sampler_outputs = self._run_worker(
                (
                    0,
                    self.parallel_config.pipeline_parallel_size - 1,
                ),  # TP rank zero for last pipeline stage
                "get_output",
            )

            # this needs to be optimized
            self._run_workers(
                "on_sampling_completed",
                scheduler_outputs=scheduler_stage_output.scheduler_outputs,
                sampler_outputs=sampler_outputs,
                seq_metadata_list=scheduler_stage_output.seq_metadata_list,
            )
            all_request_outputs = self._on_step_completed(
                scheduler_stage_output.scheduler_outputs,
                scheduler_stage_output.ignored_seqs,
                scheduler_stage_output.seq_metadata_list,
                sampler_outputs,
                scheduler_stage_output.start_time,
            )
            self.schedule_event.set()
            self.inflight_batches_count -= 1  # Batch completed pipeline
            self.output_queue.put(all_request_outputs)

    def step(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.

        This function performs one decoding iteration of the engine.
        This version does everything asynchronously and returns the results
        """
        if not self.has_started_execution_loops:
            self.start_execution_loops()
        return self.output_queue.get()

    def has_inflight_batches(self) -> bool:
        """Check if there are any batches still in the pipeline"""
        return self.inflight_batches_count > 0

    def signal_stop_scheduling(self) -> None:
        """Signal the scheduler to stop scheduling new batches"""
        self.stop_scheduling = True

    def signal_start_scheduling(self) -> None:
        """Signal the scheduler to start scheduling new batches"""
        self.stop_scheduling = False

    def stop_execution_loops(self) -> None:
        """Stop all execution loops gracefully"""
        logger.info("Stopping execution loops")
        self.should_stop = True
        self.stop_scheduling = True
        while self.has_inflight_batches():
            time.sleep(0.1)
        self._run_workers("cleanup_pp_worker",)
        self.schedule_event.set()
        self.microbatch_watch_event.set()

        # Wait for queues to be processed
        try:
            while not self.scheduler_output_queue.empty():
                self.scheduler_output_queue.get_nowait()
            while not self.output_queue.empty():
                self.output_queue.get_nowait()
        except Exception as e:
            logger.error(f"Error clearing queues: {e}")

        logger.info("Execution loops stopped")
    
