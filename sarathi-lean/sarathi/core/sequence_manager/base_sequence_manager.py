from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from sarathi.core.datatypes.request_output import RequestOutput
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import (
    SamplerOutput,
    SamplerOutputs,
    Sequence,
    SequenceMetadata,
    SequenceScheduleMetadata,
)
from sarathi.core.datatypes.sequence_status import SequenceStatus
from sarathi.utils.threading_utils import synchronized

from sarathi.logger import init_logger
logger = init_logger(__name__)

class BaseSequenceManager(ABC):

    def __init__(self):
        self.seq_map = {}

    @synchronized
    def add_seq(self, seq: Sequence) -> None:
        assert seq.seq_id not in self.seq_map
        self.seq_map[seq.seq_id] = seq

    def _free_seq(self, seq_id: int) -> None:
        assert seq_id in self.seq_map
        del self.seq_map[seq_id]

    def _preempt_seq(self, seq_id: int) -> None:
        assert seq_id in self.seq_map
        seq = self.seq_map[seq_id]
        assert seq.is_executing()
        seq.reset_for_recompute()

    def _pause_seq(self, seq_id: int) -> None:
        assert seq_id in self.seq_map
        seq = self.seq_map[seq_id]
        assert seq.is_running(), f"seq_id: {seq_id}, status: {seq.get_status()}"
        seq.set_status(SequenceStatus.PAUSED)

    def _resume_seq(self, seq_id: int) -> None:
        assert seq_id in self.seq_map
        seq = self.seq_map[seq_id]
        assert seq.is_waiting() or seq.is_paused()
        seq.set_status(SequenceStatus.RUNNING)

    def _on_seq_scheduled(self, seq_sched_metadata: SequenceScheduleMetadata) -> None:
        assert seq_sched_metadata.seq_id in self.seq_map
        self._resume_seq(seq_sched_metadata.seq_id)

    @abstractmethod
    def _get_block_table(self, seq: Sequence) -> List[int]:
        pass

    @synchronized
    def on_schedule(
        self,
        scheduler_outputs: SchedulerOutputs,
    ) -> Tuple[List[Sequence], List[SequenceMetadata]]:
        ignored_seqs: List[Sequence] = []
        for seq_id in scheduler_outputs.ignored_seq_ids:
            assert seq_id in self.seq_map
            seq = self.seq_map[seq_id]
            ignored_seqs.append(seq)
            self._free_seq(seq_id)

        for seq_id in scheduler_outputs.preempted_seq_ids:
            self._preempt_seq(seq_id)

        seq_metadata_list: List[SequenceMetadata] = []
        for seq_sched_metadata in scheduler_outputs.scheduled_seq_metadata_list:
            self._on_seq_scheduled(seq_sched_metadata)
            seq = self.seq_map[seq_sched_metadata.seq_id]
            seq_metadata_list.append(
                SequenceMetadata(
                    seq,
                    self._get_block_table(seq),
                    seq_sched_metadata.num_prompt_tokens,
                )
            )

        return ignored_seqs, seq_metadata_list

    @abstractmethod
    def _on_append_token(self, seq: Sequence) -> None:
        pass

    def _process_seq_output(
        self, seq_id: int, sample: SamplerOutput, prompt_chunk_len: int
    ) -> None:
        assert seq_id in self.seq_map
        seq = self.seq_map[seq_id]
        # at this point, the seq should be in paused state
        assert not seq.is_finished()

        if not seq.prompt_processing_finished:
            seq.update_prompt_tokens_processed(prompt_chunk_len)
            return

        seq.append_token_id(sample.output_token)
        self._on_append_token(seq)
        # this function will update the seq status
        # to finished if the stop condition is met
        seq.check_stop()
        # Move this later for pp mode 
        # if seq.is_finished():
        #     self._free_seq(seq.seq_id)

    @synchronized
    def on_step_completed(
        self,
        scheduler_outputs: SchedulerOutputs,
        sampler_outputs: Optional[SamplerOutputs],
    ) -> None:
        for scheduled_seq_metadata, sampler_output in zip(
            scheduler_outputs.scheduled_seq_metadata_list, sampler_outputs
        ):
            seq = self.seq_map[scheduled_seq_metadata.seq_id]
            if seq.is_waiting():
                # seq is preempted
                # this can happen with pipeline parallel -- if the system
                # runs out of memory, it will preempt the last arrived request
                # this request might still be executing when the next stage scheduling
                # triggers the preemption
                continue
            self._pause_seq(scheduled_seq_metadata.seq_id)
            # logger.info(f"XY: Processing seq_id: {scheduled_seq_metadata.seq_id}, len: {scheduled_seq_metadata.prompt_chunk_len}")  
            self._process_seq_output(
                scheduled_seq_metadata.seq_id,
                sampler_output,
                scheduled_seq_metadata.prompt_chunk_len,
            )

    def generate_request_outputs(
        self,
        ignored_seqs: List[Sequence],
        seq_metadata_list: List[SequenceMetadata],
    ) -> List[RequestOutput]:
        all_seqs = ignored_seqs + [x.seq for x in seq_metadata_list]
        return [RequestOutput.from_seq(seq) for seq in all_seqs]
