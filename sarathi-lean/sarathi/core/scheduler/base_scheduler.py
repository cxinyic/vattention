import time
from abc import ABC, abstractmethod
from typing import List, Tuple
from threading import Lock


from sarathi.config import BaseSchedulerConfig, CacheConfig
from sarathi.core.block_space_manager.block_space_manager_registry import (
    BlockSpaceManagerRegistry,
)
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import Sequence, SequenceStatus
from sarathi.core.policy import PolicyFactory
from sarathi.logger import init_logger
from sarathi.metrics.metrics_store import MetricsStore
from sarathi.model_executor.attention import AttentionBackend
from sarathi.core.block_space_manager.vattention_block_space_manager import vAttentionBlockSpaceManager

logger = init_logger(__name__)


class BaseScheduler(ABC):

    def __init__(
        self,
        scheduler_config: BaseSchedulerConfig,
        cache_config: CacheConfig,
    ) -> None:
        self.metrics_store = MetricsStore()
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config

        # we maintain this just for logging purposes
        self._iteration_id = -1

        # Instantiate the scheduling policy.
        self.policy = PolicyFactory.get_policy(policy_name="fcfs")
        # Create the block space manager.
        # self.block_manager = BlockSpaceManagerRegistry.get(
        #     scheduler_config.type,
        #     cache_config.block_size,
        #     cache_config.num_gpu_blocks,
        #     scheduler_config.max_model_len,
        # ) if is_vLLM_backend() else vAttentionBlockSpaceManager

        # number of running batches should be less than or equal to the number of pipeline stages
        self.num_running_batches = 0

        # TODO(zhuohan): Use deque instead of list for better performance.
        # Sequence groups in the WAITING state.
        self.waiting: List[Sequence] = []
        # Sequence groups in the RUNNING state.
        self.running: List[Sequence] = []

        self._during_upgrade = False
        self._during_draining = False
        self._is_pipeline_parallel = False
        self.sequence_lists_lock = Lock()
        self.pp_blocking_queue = {}

    def set_block_manager(self, model_config):
        attn_cfg = model_config.attention_backend
        self.attention_backend = attn_cfg
        if AttentionBackend.is_vATTN(attn_cfg):
            self.block_manager = vAttentionBlockSpaceManager(
                # model_config.hf_config.num_hidden_layers
                self.cache_config.block_size,
                self.cache_config.num_gpu_blocks,
                self.scheduler_config.max_model_len,
            )
        else:
            self.block_manager = BlockSpaceManagerRegistry.get(
                self.scheduler_config.type,
                self.cache_config.block_size,
                self.cache_config.num_gpu_blocks,
                self.scheduler_config.max_model_len,
            )  

    def reset_state(self) -> None:
        self._iteration_id = -1

    def add_seq(self, seq: Sequence) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq)
        # logger.info(f"self.waiting {len(self.waiting)}")

    def has_unfinished_seqs(self) -> bool:
        return self.waiting or self.running

    def get_num_unfinished_seqs(self) -> int:
        return len(self.waiting) + len(self.running)
    
    def set_upgrade(self) -> None:
        self._during_upgrade = True
    
    def set_drain(self) -> None:
        self._during_draining = True
    
    def has_enough_blocks(self, required_blocks: int) -> bool:
        current_free_blocks = self.block_manager.get_num_free_gpu_blocks()
        if current_free_blocks >= required_blocks:
            return True
        return False

    @abstractmethod
    def _schedule(self) -> SchedulerOutputs:
        pass

    def schedule(self) -> SchedulerOutputs:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running and self.waiting.
        self._iteration_id += 1

        if self.num_running_batches >= self.scheduler_config.num_pipeline_stages:
            return SchedulerOutputs(
                self._iteration_id,
                ignored_seq_ids=[],
                preempted_seq_ids=[],
                scheduled_seq_metadata_list=[],
            )
        
        scheduler_outputs = None
        scheduler_outputs = self._schedule()

    
        if not scheduler_outputs.is_empty():
            self.num_running_batches += 1

        return scheduler_outputs

    def remove_finished_seqs(self) -> None:
        self.running = [seq for seq in self.running if not seq.is_finished()]

    def free_finished_seqs(self) -> None:
        for seq in self.running:
            if seq.is_finished():
                self._free_seq(seq)

    def on_step_completed(self) -> None:
        self.free_finished_seqs()
        self.remove_finished_seqs()
        self.num_running_batches -= 1
    
    def pp_on_step_completed(self,
        scheduler_outputs: SchedulerOutputs) -> None:
        if not self._is_pipeline_parallel:
            return
        self.sequence_lists_lock.acquire()
        try:
            for scheduled_seq_metadata in scheduler_outputs.scheduled_seq_metadata_list:
                seq_id = scheduled_seq_metadata.seq_id
                if seq_id in self.pp_blocking_queue:
                    self.running.append(self.pp_blocking_queue[seq_id])
                    del self.pp_blocking_queue[seq_id]
        finally:
            self.sequence_lists_lock.release()

    def _allocate(self, seq: Sequence) -> None:
        self.block_manager.allocate(seq)

    def _free_seq(self, seq: Sequence) -> None:
        self.block_manager.free(seq)

    def _append_slot(
        self,
        seq: Sequence,
    ) -> None:
        assert seq.is_executing()
        self.block_manager.append_slot(seq)

    def _preempt(
        self,
        seq: Sequence,
    ) -> None:
        assert seq.is_executing()
        self._free_seq(seq)
        if type(self.block_manager) == vAttentionBlockSpaceManager:
            self.block_manager.preemption_queue.append(seq)
        self.waiting.insert(0, seq)

    def _check_request_prompt_length(self, seq: Sequence) -> bool:
        if seq.get_len() > self.scheduler_config.max_model_len:
            logger.warning(
                f"Input prompt ({seq.get_len()} tokens) is too long"
                f" and exceeds limit of {seq.sampling_params.max_tokens}"
            )
            seq.set_status(SequenceStatus.FINISHED_IGNORED)
            self.waiting.pop(0)
            return False

        return True
    

