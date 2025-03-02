"""A GPU worker class."""

from queue import Queue
from threading import Thread
from typing import Optional, Tuple

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
from sarathi.core.datatypes.sequence import SamplerOutputs
from sarathi.logger import init_logger
from sarathi.utils.threading_utils import exit_on_error, synchronized
from sarathi.worker.base_worker import BaseWorker
from typing import List, Tuple, Union
from sarathi.core.datatypes.sequence import SequenceMetadata
from datetime import datetime
import time


logger = init_logger(__name__)


class PipelineParallelWorker(BaseWorker):
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
        super().__init__(
            model_config,
            parallel_config,
            scheduler_config,
            cache_config,
            metrics_config,
            local_rank,
            rank,
            distributed_init_method,
        )
        self.execution_queue = Queue()
        self.output_queue = Queue()
        self.execution_thread = Thread(target=self._execution_loop, daemon=True)
        self.should_stop = False

    def _verify_parallel_config(self) -> None:
        assert self.parallel_config.pipeline_parallel_size > 1

    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        super().init_cache_engine(cache_config)
        self.execution_thread.start()

    def enqueue(self, scheduler_outputs, preempted_seq=None):
        """Modified enqueue method to handle preempted sequences."""
        work_item = {
            "scheduler_outputs": scheduler_outputs,
            "preempted_seq": preempted_seq or []
        }
        self.execution_queue.put(work_item)

    def on_step_completed(
        self, scheduler_outputs: SchedulerOutputs, sampler_outputs: SamplerOutputs
    ) -> None:
        # in pipeline parallel case, each stage won't have sampler output
        # so we need to do the book keeping update later
        pass

    @synchronized
    def on_sampling_completed(
        self, scheduler_outputs: SchedulerOutputs, sampler_outputs: SamplerOutputs, seq_metadata_list: List[SequenceMetadata]
    ) -> None:
        self.seq_manager.on_step_completed(scheduler_outputs, sampler_outputs)
        
        for seq_metadata in seq_metadata_list:
            seq = self.seq_manager.seq_map[seq_metadata.seq.seq_id]  
            # must check from the seq_map of seq_manager
            if seq.is_finished():
                self.cache_engine.free_request(seq.seq_id)

    @exit_on_error
    def _execution_loop(self) -> None:
        torch.cuda.set_device(self.device)

        while True:
            if self.should_stop:
                break
            # Get both scheduler outputs and any preempted sequences
            work_item = self.execution_queue.get()
            
            # Check for stop signal
            if work_item is None:
                break
                
            scheduler_outputs = work_item["scheduler_outputs"]
            preempted_seq = work_item.get("preempted_seq", [])
            is_preempted_seq_empty = len(preempted_seq) == 0
            output = self.execute_model(scheduler_outputs, preempted_seq)

            if not self.is_tensor_parallel_rank_zero:
                continue
            if not is_preempted_seq_empty and self.is_last_pipeline_stage:
                output_package = {
                    "output": output,
                    "preemption_completed": True
                }
                self.output_queue.put(output_package)
            else:
                if self.is_first_pipeline_stage or self.is_last_pipeline_stage:
                    output_package = {
                        "output": output,
                        "preemption_completed": False
                    }
                    self.output_queue.put(output_package)
        
        logger.info(f"Execution loop stopped on rank {self.local_rank}")

    def get_output(self) -> Optional[SamplerOutputs]:
        return self.output_queue.get()

    @synchronized
    def get_model_parallel_ranks(self) -> Tuple[int, int]:
        return self.tensor_model_parallel_rank, self.pipeline_model_parallel_rank

    @synchronized
    def cleanup_pp_worker(self) -> None:
        """Cleanup worker resources including execution thread and queues"""
        logger.info(f"Cleaning up worker on rank {self.local_rank}")
        
        # Signal thread to stop
        self.should_stop = True  
        self.execution_queue.put(None)
        # Clear the queues
        try:
            while not self.execution_queue.empty():
                self.execution_queue.get_nowait()
            while not self.output_queue.empty():
                self.output_queue.get_nowait()
        except Exception as e:
            logger.error(f"Error clearing queues: {e}")
        
        # Put a stop signal in the queue
        self.execution_queue.put(None)
        
        # Wait for execution thread to finish
        if self.execution_thread.is_alive():
            self.execution_thread.join(timeout=5.0)
            if self.execution_thread.is_alive():
                logger.warning(f"Execution thread on rank {self.local_rank} did not stop gracefully")
        
        logger.info(f"Worker cleanup completed on rank {self.local_rank}")