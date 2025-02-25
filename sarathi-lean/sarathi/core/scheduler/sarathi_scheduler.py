import time
from typing import List

import numpy as np

from sarathi.config import CacheConfig, SarathiSchedulerConfig
from sarathi.core.block_space_manager.sarathi_block_space_manager import (
    SarathiBlockSpaceManager,
)
from sarathi.core.block_space_manager.vattention_block_space_manager import (
    vAttentionBlockSpaceManager
)
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import Sequence, SequenceScheduleMetadata
from sarathi.core.scheduler.base_scheduler import BaseScheduler
from sarathi.logger import init_logger
from sarathi.model_executor.attention import is_vattention_backend
from sarathi.config import UpgradeStrategy

logger = init_logger(__name__)


class SarathiScheduler(BaseScheduler):

    def __init__(
        self,
        scheduler_config: SarathiSchedulerConfig,
        cache_config: CacheConfig,
    ) -> None:
        super().__init__(scheduler_config, cache_config)

        self.prompt_limit = self.scheduler_config.max_model_len
        self.chunk_size = self.scheduler_config.chunk_size
        self.enable_dynamic_chunking_schedule = (
            self.scheduler_config.enable_dynamic_chunking_schedule
        )
        # next four params apply only when using dynamic schedule
        self.low_chunk_size = self.scheduler_config.low_chunk_size
        self.high_chunk_size = self.scheduler_config.high_chunk_size
        self.chunk_schedule_max_tokens = self.scheduler_config.chunk_schedule_max_tokens
        self.chunk_schedule_stages = self.scheduler_config.chunk_schedule_stages
        self.enable_rolling_prefills = False

        if self.enable_dynamic_chunking_schedule:
            assert self.chunk_schedule_stages > 0
            assert self.chunk_schedule_max_tokens > 0
            assert self.low_chunk_size % 32 == 0
            assert self.high_chunk_size % 32 == 0
            self._chunk_sizes = self._compute_chunk_size_schedule()
            self._tokens_per_stage = int(
                np.ceil(self.chunk_schedule_max_tokens / self.chunk_schedule_stages)
            )
        
        # upgrade mode only
        self.preemption_strategy = None  # 'partial' or 'full'
        self.upgrade_preempted_seq_ids = set()
        self.upgrade_required_blocks = 0

    def _compute_chunk_size_schedule(self):
        # create num_steps equally spaced chunk sizes between low_chunk_size and high_chunk_size
        chunk_sizes = np.linspace(
            self.low_chunk_size,
            self.high_chunk_size,
            self.chunk_schedule_stages,
            dtype=np.int32,
        )[::-1]
        # align each chunk size to the nearest multiple of 32 or self.low_chunk_size
        round_of_chunk_sizes = min(32, self.low_chunk_size)
        chunk_sizes = (
            np.round(chunk_sizes / round_of_chunk_sizes) * round_of_chunk_sizes
        )
        chunk_sizes = chunk_sizes.astype(np.int64).tolist()

        return chunk_sizes

    def get_block_space_manager_class(self):
        return vAttentionBlockSpaceManager if is_vattention_backend() else SarathiBlockSpaceManager 
        # return SarathiBlockSpaceManager

    def _get_seq_next_num_prefill_tokens(
        self, seq: Sequence,
        batch_contains_prefill: bool,
        num_batched_tokens: int
    ) -> int:
        assert not seq.is_finished()

        if self.enable_dynamic_chunking_schedule:
            request_stage_idx = int(
                np.ceil(seq.get_num_prompt_tokens_processed() // self._tokens_per_stage)
            )
            assert request_stage_idx < len(self._chunk_sizes)
            chunk_size = self._chunk_sizes[request_stage_idx]
        else:
            chunk_size = self.chunk_size

        next_num_tokens = min(
            seq.get_prompt_len() - seq.get_num_prompt_tokens_processed(),
            chunk_size - num_batched_tokens,
        )
        
        if not batch_contains_prefill:
            return next_num_tokens

        if self.enable_rolling_prefills and num_batched_tokens < chunk_size:
            # we can have multiple prefills per batch
            # but the total number of tokens should not exceed
            # the max batch size
            return next_num_tokens
        else:
            # we will only allow one prefill per batch
            return 0

    def _schedule(self) -> SchedulerOutputs:
        # Fix the current time.
        now = time.monotonic()

        running: List[Sequence] = []
        ignored_seq_ids: List[int] = []
        preempted_seq_ids: List[int] = []
        scheduled_seq_metadata_list: List[SequenceScheduleMetadata] = []

        num_batched_tokens: int = 0
        batch_contains_prefill: bool = False
        # logger.info(f"len of self.running: {len(self.running)}, waiting: {len(self.waiting)}")
        # TODO(XY): think about the upgrade scheduler
        if self._during_upgrade:
            # Filter out upgrade-preempted sequences from running queue
            filtered_running = []
            for seq in self.running:
                if seq.seq_id in self.upgrade_preempted_seq_ids:
                    preempted_seq_ids.append(seq.seq_id)
                    continue
                filtered_running.append(seq)
            self.running = filtered_running
            # if len(filtered_running) > 0:
            #     logger.info(f"upgrade before: filtered_running sequences length: {len(filtered_running)}")

            # if type(self.block_manager) == vAttentionBlockSpaceManager:
            #     original_free_blocks = self.block_manager.free_blocks
            #     self.block_manager.set_free_blocks(original_free_blocks - self.upgrade_required_blocks)
        
        if type(self.block_manager) == vAttentionBlockSpaceManager:
            self.block_manager.clear_promised_blocks()
        ######################################################################
        # Phase 1: Add existing running sequence groups to the batch.
        # There are two cases:
        # 1. The sequence group has incomplete prefill. The routine
        # remains identical to the one in sarathi scheduler for such sequences.
        # 2. The sequence group has completed prefill. In this case, we need to
        # check for memory availability for the next chunk of decode tokens, and preempt
        # some sequence groups if necessary. Note that, the preempted sequence groups
        # might belong to either of the two categories.
        ######################################################################

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        self.running = self.policy.sort_by_priority(now, self.running)

        # in first pass process all the requests with prefill completed
        # this allows us to accurately account for the number of decode tokens
        running_prefills: List[Sequence] = []

        while self.running:
            seq = self.running.pop(0)

            if not seq.is_paused():
                running.append(seq)
                continue

            if not seq.prompt_processing_finished:
                running_prefills.append(seq)
                continue

            while not self.block_manager.can_append_slot():
                # print(f" [Sarathi] [{type(self.block_manager)}] : Cannot append seq {seq.seq_id} with {seq.get_len()} tokens")
                # if type(self.block_manager) == vAttentionBlockSpaceManager:
                #     print(f" [Sarathi] [{type(self.block_manager)}] : free blocks {self.block_manager.free_blocks - self.block_manager.promised_blocks} required blocks {self.block_manager.get_num_blocks(seq)}")
                # elif type(self.block_manager) == SarathiBlockSpaceManager:
                #     print(f" [Sarathi] [{type(self.block_manager)}] : free blocks {self.block_manager.get_num_free_gpu_blocks()} required blocks {self.block_manager.get_num_initial_blocks(seq)}")
                # logger.info("Cannot append slot")
                if self.running:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq = self.running.pop(-1)
                    # logger.info(f"preempting seq {victim_seq.seq_id}")
                    self._preempt(victim_seq)
                    preempted_seq_ids.append(victim_seq.seq_id)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq)
                    preempted_seq_ids.append(seq.seq_id)
                    break
            else:
                # Append new slots to the sequence group.
                # print(f" [Sarathi] [{type(self.block_manager)}] : Can append seq {seq.seq_id} with {seq.get_len()} tokens")
                # if type(self.block_manager) == vAttentionBlockSpaceManager:
                #     print(f" [Sarathi] [{type(self.block_manager)}] : free blocks {self.block_manager.free_blocks - self.block_manager.promised_blocks} required blocks {self.block_manager.get_num_blocks(seq)}")
                # elif type(self.block_manager) == SarathiBlockSpaceManager:
                #     print(f" [Sarathi] [{type(self.block_manager)}] : free blocks {self.block_manager.get_num_free_gpu_blocks()} required blocks {self.block_manager.get_num_initial_blocks(seq)}")

                self._append_slot(seq)
                running.append(seq)
                num_batched_tokens += 1
                scheduled_seq_metadata_list.append(
                    SequenceScheduleMetadata.from_sequence(seq)
                )

        # now add the requests with prefill incomplete
        # the memory for all these prefills has already been allocated
        # so we should be able to run all of them
        for seq in running_prefills:
            assert not seq.prompt_processing_finished

            next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(
                seq, batch_contains_prefill, num_batched_tokens
            )

            # as long as the request could fit in the batch previously
            # it should be able to fit in the batch now
            # so in non-pipeline case this condition should always be false
            # however, in pipeline case, the grouping of requests can change
            # between different microbatches, so this is not guaranteed to be always true
            if next_num_prefill_tokens == 0:
                running.append(seq)
                continue
            
            batch_contains_prefill = True
            num_batched_tokens += next_num_prefill_tokens
            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(
                    seq, prompt_chunk_len=next_num_prefill_tokens
                )
            )
            running.append(seq)

        ######################################################################
        # Phase 2: Add waiting (new) sequence groups to the batch.
        # This routine is nearly-identical to the one in sarathi scheduler
        ######################################################################
        # Optimization: We do not sort the waiting queue since the preempted
        # sequence groups are added to the front and the new sequence groups
        # are added to the back.
        while self.waiting:
            # draining mode will not allow new requests to be scheduled
            if self._during_draining:
                break
            seq = self.waiting[0]

            # This is required to handle benchmarking where we set request arrival time ahead of time
            if seq.arrival_time > now:
                break

            if not self._check_request_prompt_length(seq):
                ignored_seq_ids.append(seq.seq_id)
                continue

            # If the sequence group cannot be allocated, stop.
            # print("[SarahtiScheduler] Allocating sequence group", seq.seq_id, " with prompt len ", seq.get_prompt_len())
            if not self.block_manager.can_allocate(seq):
                # logger.info(f"Cannot allocate for waiting seq {seq.seq_id}")
                # this is different from vllm scheduler
                # even if we cannot allocate this sequence group
                # there might be other sequence groups that can be allocated
                # if type(self.block_manager) == vAttentionBlockSpaceManager:
                #     print(f" [Sarathi] [{type(self.block_manager)}] : free blocks {self.block_manager.free_blocks}, promised: {self.block_manager.promised_blocks}, actual free blocks: {self.block_manager.free_blocks}, required blocks {self.block_manager.get_num_blocks(seq)}")
                # elif type(self.block_manager) == SarathiBlockSpaceManager:
                #     print(f" [Sarathi] [{type(self.block_manager)}] : free blocks {self.block_manager.get_num_free_gpu_blocks()} required blocks {self.block_manager.get_num_initial_blocks(seq)}")
                # print(f" [Sarathi] [{type(self.block_manager)}] : Cannot allocate seq {seq.seq_id} with {seq.get_len()} tokens")
                
                break
            # else:
                # print(f" [Sarathi] [{type(self.block_manager)}] : Can allocate seq {seq.seq_id} with {seq.get_len()} tokens")
                # if type(self.block_manager) == vAttentionBlockSpaceManager:
                #     print(f" [Sarathi] [{type(self.block_manager)}] : free blocks {self.block_manager.free_blocks - self.block_manager.promised_blocks} required blocks {self.block_manager.get_num_blocks(seq)}")
                # elif type(self.block_manager) == SarathiBlockSpaceManager:
                #     print(f" [Sarathi] [{type(self.block_manager)}] : free blocks {self.block_manager.get_num_free_gpu_blocks()} required blocks {self.block_manager.get_num_initial_blocks(seq)}")

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            if len(running) >= self.scheduler_config.max_num_seqs:
                break

            # check if we can fit the prefill in the batch
            next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(
                seq, batch_contains_prefill, num_batched_tokens
            )

            if next_num_prefill_tokens == 0:
                break

            seq = self.waiting.pop(0)
            self._allocate(seq)
            batch_contains_prefill = True
            num_batched_tokens += next_num_prefill_tokens
            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(
                    seq, prompt_chunk_len=next_num_prefill_tokens
                )
            )
            running.append(seq)

        # make sure that prefills are at the start of the batch, so that we don't violate assumptions
        # made in the original vllm codebase
        self.running = running
        # logger.info(f"len of self.running: {len(self.running)}, waiting: {len(self.waiting)}")
        # for seq in self.running:
        #     logger.info(f"running sequence: {seq.seq_id}")
        return SchedulerOutputs(
            id=self._iteration_id,
            ignored_seq_ids=ignored_seq_ids,
            preempted_seq_ids=preempted_seq_ids,
            scheduled_seq_metadata_list=scheduled_seq_metadata_list,
        )

    def _schedule_upgrade(self) -> SchedulerOutputs:
        # Fix the current time.
        now = time.monotonic()
        # logger.info(f"blocks num is {self.block_manager.get_num_free_gpu_blocks()}")  

        running: List[Sequence] = []
        ignored_seq_ids: List[int] = []
        preempted_seq_ids: List[int] = []
        scheduled_seq_metadata_list: List[SequenceScheduleMetadata] = []

        num_batched_tokens: int = 0
        batch_contains_prefill: bool = False
        # Filter out upgrade-preempted sequences from running queue
        filtered_running = []
        for seq in self.running:
            if seq.seq_id in self.upgrade_preempted_seq_ids:
                preempted_seq_ids.append(seq.seq_id)
                continue
            filtered_running.append(seq)
        self.running = filtered_running
        logger.info(f"upgrade before: filtered_running sequences length: {len(filtered_running)}")

        if type(self.block_manager) == vAttentionBlockSpaceManager:
            original_free_blocks = self.block_manager.free_blocks
            self.block_manager.set_free_blocks(original_free_blocks - self.upgrade_required_blocks)
            self.block_manager.clear_promised_blocks()
        ######################################################################
        # Phase 1: Add existing running sequence groups to the batch.
        # There are two cases:
        # 1. The sequence group has incomplete prefill. The routine
        # remains identical to the one in sarathi scheduler for such sequences.
        # 2. The sequence group has completed prefill. In this case, we need to
        # check for memory availability for the next chunk of decode tokens, and preempt
        # some sequence groups if necessary. Note that, the preempted sequence groups
        # might belong to either of the two categories.
        ######################################################################

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        self.running = self.policy.sort_by_priority(now, self.running)

        # in first pass process all the requests with prefill completed
        # this allows us to accurately account for the number of decode tokens
        running_prefills: List[Sequence] = []

        while self.running:
            seq = self.running.pop(0)

            if not seq.is_paused():
                running.append(seq)
                continue

            if not seq.prompt_processing_finished:
                running_prefills.append(seq)
                continue

            while not self.block_manager.can_append_slot():
                # print(f" [Sarathi] [{type(self.block_manager)}] : Cannot append seq {seq.seq_id} with {seq.get_len()} tokens")
                # if type(self.block_manager) == vAttentionBlockSpaceManager:
                #     print(f" [Sarathi] [{type(self.block_manager)}] : free blocks {self.block_manager.free_blocks - self.block_manager.promised_blocks} required blocks {self.block_manager.get_num_blocks(seq)}")
                # elif type(self.block_manager) == SarathiBlockSpaceManager:
                #     print(f" [Sarathi] [{type(self.block_manager)}] : free blocks {self.block_manager.get_num_free_gpu_blocks()} required blocks {self.block_manager.get_num_initial_blocks(seq)}")
                logger.info("Cannot append slot")
                if self.running:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq = self.running.pop(-1)
                    self._preempt(victim_seq)
                    preempted_seq_ids.append(victim_seq.seq_id)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq)
                    preempted_seq_ids.append(seq.seq_id)
                    break
            else:
                # Append new slots to the sequence group.
                # print(f" [Sarathi] [{type(self.block_manager)}] : Can append seq {seq.seq_id} with {seq.get_len()} tokens")
                # if type(self.block_manager) == vAttentionBlockSpaceManager:
                #     print(f" [Sarathi] [{type(self.block_manager)}] : free blocks {self.block_manager.free_blocks - self.block_manager.promised_blocks} required blocks {self.block_manager.get_num_blocks(seq)}")
                # elif type(self.block_manager) == SarathiBlockSpaceManager:
                #     print(f" [Sarathi] [{type(self.block_manager)}] : free blocks {self.block_manager.get_num_free_gpu_blocks()} required blocks {self.block_manager.get_num_initial_blocks(seq)}")

                self._append_slot(seq)
                running.append(seq)
                num_batched_tokens += 1
                scheduled_seq_metadata_list.append(
                    SequenceScheduleMetadata.from_sequence(seq)
                )

        # now add the requests with prefill incomplete
        # the memory for all these prefills has already been allocated
        # so we should be able to run all of them
        for seq in running_prefills:
            assert not seq.prompt_processing_finished

            next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(
                seq, batch_contains_prefill, num_batched_tokens
            )

            # as long as the request could fit in the batch previously
            # it should be able to fit in the batch now
            # so in non-pipeline case this condition should always be false
            # however, in pipeline case, the grouping of requests can change
            # between different microbatches, so this is not guaranteed to be always true
            if next_num_prefill_tokens == 0:
                running.append(seq)
                continue
            
            batch_contains_prefill = True
            num_batched_tokens += next_num_prefill_tokens
            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(
                    seq, prompt_chunk_len=next_num_prefill_tokens
                )
            )
            running.append(seq)

        ######################################################################
        # Phase 2: Add waiting (new) sequence groups to the batch.
        # This routine is nearly-identical to the one in sarathi scheduler
        ######################################################################
        # Optimization: We do not sort the waiting queue since the preempted
        # sequence groups are added to the front and the new sequence groups
        # are added to the back.
        while self.waiting:
            seq = self.waiting[0]
            # logger.info(f"upgrade: waiting sequence: {seq.seq_id}")
            if seq.seq_id in self.upgrade_preempted_seq_ids:
                self.waiting.pop(0)
                logger.info(f"step 1")
                preempted_seq_ids.append(seq.seq_id)
                continue

            # This is required to handle benchmarking where we set request arrival time ahead of time
            if seq.arrival_time > now:
                logger.info(f"step 2")
                break

            if not self._check_request_prompt_length(seq):
                logger.info(f"step 3")
                ignored_seq_ids.append(seq.seq_id)
                continue

            # If the sequence group cannot be allocated, stop.
            # print("[SarahtiScheduler] Allocating sequence group", seq.seq_id, " with prompt len ", seq.get_prompt_len())
            if not self.block_manager.can_allocate(seq):
                logger.info(f"step 4")
                # this is different from vllm scheduler
                # even if we cannot allocate this sequence group
                # there might be other sequence groups that can be allocated
                # if type(self.block_manager) == vAttentionBlockSpaceManager:
                #     print(f" [Sarathi] [{type(self.block_manager)}] : free blocks {self.block_manager.free_blocks}, promised: {self.block_manager.promised_blocks}, actual free blocks: {self.block_manager.free_blocks}, required blocks {self.block_manager.get_num_blocks(seq)}")
                # elif type(self.block_manager) == SarathiBlockSpaceManager:
                #     print(f" [Sarathi] [{type(self.block_manager)}] : free blocks {self.block_manager.get_num_free_gpu_blocks()} required blocks {self.block_manager.get_num_initial_blocks(seq)}")
                # print(f" [Sarathi] [{type(self.block_manager)}] : Cannot allocate seq {seq.seq_id} with {seq.get_len()} tokens")
                
                break
            # else:
                # print(f" [Sarathi] [{type(self.block_manager)}] : Can allocate seq {seq.seq_id} with {seq.get_len()} tokens")
                # if type(self.block_manager) == vAttentionBlockSpaceManager:
                #     print(f" [Sarathi] [{type(self.block_manager)}] : free blocks {self.block_manager.free_blocks - self.block_manager.promised_blocks} required blocks {self.block_manager.get_num_blocks(seq)}")
                # elif type(self.block_manager) == SarathiBlockSpaceManager:
                #     print(f" [Sarathi] [{type(self.block_manager)}] : free blocks {self.block_manager.get_num_free_gpu_blocks()} required blocks {self.block_manager.get_num_initial_blocks(seq)}")

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            if len(running) >= self.scheduler_config.max_num_seqs:
                logger.info(f"step 5")
                break

            # check if we can fit the prefill in the batch
            next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(
                seq, batch_contains_prefill, num_batched_tokens
            )
            logger.info(f"step 6")

            if next_num_prefill_tokens == 0:
                logger.info(f"step 7")
                break

            seq = self.waiting.pop(0)
            self._allocate(seq)
            batch_contains_prefill = True
            num_batched_tokens += next_num_prefill_tokens
            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(
                    seq, prompt_chunk_len=next_num_prefill_tokens
                )
            )
            running.append(seq)

        # make sure that prefills are at the start of the batch, so that we don't violate assumptions
        # made in the original vllm codebase
        self.running = running
        logger.info(f"upgrade: len of self.running: {len(self.running)}")
        # for seq in self.running:
        #     logger.info(f"upgrade: running sequence: {seq.seq_id}")
        return SchedulerOutputs(
            id=self._iteration_id,
            ignored_seq_ids=ignored_seq_ids,
            preempted_seq_ids=preempted_seq_ids,
            scheduled_seq_metadata_list=scheduled_seq_metadata_list,
        )
    
    # def select_preemption_sequences(self, required_blocks: int, strategy: str = 'partial') -> List[Sequence]:
        """Select sequences for preemption based on strategy.
        
        Args:
            required_blocks (int): Number of blocks needed to be freed
            strategy (str): Either 'partial' or 'full'
                - 'partial': Preempt minimum sequences needed, keep others running
                - 'full': Preempt all sequences, but only free physical memory for enough sequences to meet required_blocks
        
        Returns:
            Tuple[List[Sequence], List[Sequence]]: For partial strategy: (sequences_to_preempt, [])
                                                  For full strategy: (sequences_for_physical_free, sequences_for_virtual_free)
        """
        assert strategy in ['partial', 'full'], f"Invalid strategy: {strategy}"
        self.preemption_strategy = strategy
        self.upgrade_required_blocks = required_blocks
        
        if required_blocks <= 0:
            return [], []
        
        
        current_free_blocks = self.block_manager.get_num_free_gpu_blocks()
        logger.info(f"Free blocks number is {current_free_blocks}")
        if current_free_blocks >= required_blocks:
            logger.info(f"Enough free blocks, no need to preempt")
            return [], []

        freed_blocks = 0
        sequences_for_physical_free = []
        sequences_for_virtual_free = []
        
        if strategy == 'partial':
            # Sort by arrival time for partial strategy - preempt newer sequences first
            running_sequences = sorted(
                self.running,
                key=lambda seq: seq.arrival_time,
                reverse=True
            )
            # TODO(XY): consider this buffer size. hardcode for now
            required_blocks += 1  # Buffer for partial strategy

            # For partial strategy, just free enough blocks
            for seq in running_sequences:
                if type(self.block_manager) == vAttentionBlockSpaceManager:
                    seq_blocks = self.block_manager.get_num_blocks(seq)
                else:
                    logger.error("Incorrect block manager type for upgrade")
                    return [], []

                freed_blocks += seq_blocks
                sequences_for_physical_free.append(seq)
                logger.info(f"Selected sequence {seq.seq_id} for preemption, frees {seq_blocks} blocks")

                if freed_blocks >= required_blocks:
                    break

        else:  # full strategy
            # Sort by prompt_processing_finished first to prioritize post-prefill sequences
            running_sequences = sorted(
                self.running,
                key=lambda seq: (seq.prompt_processing_finished, seq.arrival_time),
                reverse=True
            )

            # Keep adding to physical_free until we meet required_blocks
            for seq in running_sequences:
                if freed_blocks < required_blocks:
                    # Need more blocks, add to physical free
                    if type(self.block_manager) == vAttentionBlockSpaceManager:
                        seq_blocks = self.block_manager.get_num_blocks(seq)
                        freed_blocks += seq_blocks
                    sequences_for_physical_free.append(seq)
                    logger.info(f"Selected sequence {seq.seq_id} for physical memory free, frees {seq_blocks} blocks")
                else:
                    # Have enough blocks, rest go to virtual free
                    sequences_for_virtual_free.append(seq)
                    logger.info(f"Selected sequence {seq.seq_id} for virtual memory free")

        logger.info(f"Strategy: {strategy}, Total blocks freed: {freed_blocks}, Required: {required_blocks}")
        logger.info(f"Physical free sequences: {len(sequences_for_physical_free)}, Virtual free sequences: {len(sequences_for_virtual_free)}")

        # Update preempted sequence IDs for all sequences
        self.upgrade_preempted_seq_ids.update(seq.seq_id for seq in sequences_for_physical_free)
        self.upgrade_preempted_seq_ids.update(seq.seq_id for seq in sequences_for_virtual_free)
        
        # Free both from the block manager
        for seq in sequences_for_physical_free:
            self.block_manager.free(seq)
        for seq in sequences_for_virtual_free:
            self.block_manager.free(seq)
              
        # For full strategy, clear the running queue
        # Update running queue based on strategy
        if strategy == 'full':
            self.running = []
                  
        return sequences_for_physical_free, sequences_for_virtual_free
    
    def select_preemption_sequences(self, required_blocks: int, strategy: str = 'partial', 
                            selection_policy: UpgradeStrategy.SelectionPolicy = UpgradeStrategy.SelectionPolicy.BY_ARRIVAL_TIME) -> List[Sequence]:
        """Select sequences for preemption based on strategy and policy.
        
        Args:
            required_blocks (int): Number of blocks needed to be freed
            strategy (str): Either 'partial' or 'full'
                - 'partial': Preempt minimum sequences needed, keep others running
                - 'full': Preempt all sequences, but only free physical memory for enough sequences to meet required_blocks
            selection_policy (SelectionPolicy): Policy to determine which sequences to preempt first
                - BY_ARRIVAL_TIME: Select newer sequences first (based on arrival time)
                - BY_FINISH_TIME: Select sequences with the longest estimated remaining time
                - BY_KV_CACHE_SIZE: Select sequences using the most KV cache memory
        
        Returns:
            Tuple[List[Sequence], List[Sequence]]: For partial strategy: (sequences_to_preempt, [])
                                                  For full strategy: (sequences_for_physical_free, sequences_for_virtual_free)
        """
        assert strategy in ['partial', 'full'], f"Invalid strategy: {strategy}"
        self.preemption_strategy = strategy
        self.upgrade_required_blocks = required_blocks
        
        if required_blocks <= 0:
            return [], []
        
        current_free_blocks = self.block_manager.get_num_free_gpu_blocks()
        logger.info(f"Free blocks number is {current_free_blocks}")
        if current_free_blocks >= required_blocks:
            logger.info(f"Enough free blocks, no need to preempt")
            return [], []

        freed_blocks = 0
        sequences_for_physical_free = []
        sequences_for_virtual_free = []
        
        # Define sorting key function based on selection policy
        def get_sort_key(seq):
            if selection_policy == UpgradeStrategy.SelectionPolicy.BY_ARRIVAL_TIME:
                # Newer sequences first (higher arrival time)
                return seq.arrival_time
            elif selection_policy == UpgradeStrategy.SelectionPolicy.BY_FINISH_TIME:
                return seq.num_tokens_to_finish()
            elif selection_policy == UpgradeStrategy.SelectionPolicy.BY_KV_CACHE_SIZE:
                if type(self.block_manager) == vAttentionBlockSpaceManager:
                    return self.block_manager.get_num_blocks(seq)
                return 0  # Default fallback if block manager is not vAttentionBlockSpaceManager
        
        # Use the same sorting for both strategies - sort based only on selection policy
        running_sequences = sorted(
            self.running,
            key=get_sort_key,
            reverse=True  # Reverse to get the highest values first
        )
        
        if strategy == 'partial':
            # Buffer for partial strategy
            required_blocks += 1

            # For partial strategy, just free enough blocks
            for seq in running_sequences:
                if type(self.block_manager) == vAttentionBlockSpaceManager:
                    seq_blocks = self.block_manager.get_num_blocks(seq)
                else:
                    logger.error("Incorrect block manager type for upgrade")
                    return [], []

                freed_blocks += seq_blocks
                sequences_for_physical_free.append(seq)
                logger.info(f"Selected sequence {seq.seq_id} for preemption (policy: {selection_policy.name}), frees {seq_blocks} blocks")

                if freed_blocks >= required_blocks:
                    break

        else:  # full strategy
            # Keep adding to physical_free until we meet required_blocks
            for seq in running_sequences:
                if freed_blocks < required_blocks:
                    # Need more blocks, add to physical free
                    if type(self.block_manager) == vAttentionBlockSpaceManager:
                        seq_blocks = self.block_manager.get_num_blocks(seq)
                        freed_blocks += seq_blocks
                    sequences_for_physical_free.append(seq)
                    logger.info(f"Selected sequence {seq.seq_id} for physical memory free (policy: {selection_policy.name}), frees {seq_blocks} blocks")
                else:
                    # Have enough blocks, rest go to virtual free
                    sequences_for_virtual_free.append(seq)
                    logger.info(f"Selected sequence {seq.seq_id} for virtual memory free (policy: {selection_policy.name})")

        logger.info(f"Strategy: {strategy}, Policy: {selection_policy.name}, Total blocks freed: {freed_blocks}, Required: {required_blocks}")
        logger.info(f"Physical free sequences: {len(sequences_for_physical_free)}, Virtual free sequences: {len(sequences_for_virtual_free)}")

        # Update preempted sequence IDs for all sequences
        self.upgrade_preempted_seq_ids.update(seq.seq_id for seq in sequences_for_physical_free)
        self.upgrade_preempted_seq_ids.update(seq.seq_id for seq in sequences_for_virtual_free)
        
        # Free both from the block manager
        for seq in sequences_for_physical_free:
            self.block_manager.free(seq)
        for seq in sequences_for_virtual_free:
            self.block_manager.free(seq)
              
        # For full strategy, clear the running queue
        if strategy == 'full':
            self.running = []
                  
        return sequences_for_physical_free, sequences_for_virtual_free