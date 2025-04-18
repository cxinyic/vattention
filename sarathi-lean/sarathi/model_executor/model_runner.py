from typing import List, Optional, Tuple

import torch
import torch.distributed

from sarathi.config import (
    BaseSchedulerConfig,
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerType,
)
from sarathi.core.datatypes.sampling_params import SamplingParams
from sarathi.core.datatypes.sequence import Sequence, SequenceMetadata
from sarathi.logger import init_logger
from sarathi.metrics.constants import CpuOperationMetrics, OperationMetrics
from sarathi.metrics.cpu_timer import CpuTimer
from sarathi.metrics.cuda_timer import CudaTimer
from sarathi.model_executor import get_model, set_random_seed
from sarathi.model_executor.attention import get_attention_wrapper
from sarathi.model_executor.layers.sampler import Sampler
from sarathi.model_executor.utils import pad_to_alignment
from sarathi.utils import get_gpu_memory
from sarathi.worker.cache_engine import get_cache_engine
from sarathi.model_executor.attention import AttentionBackend
logger = init_logger(__name__)

USE_UVM = False
class ModelRunner:

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: BaseSchedulerConfig,
        cache_config: CacheConfig,
        device: torch.device,
        rank: int,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device = device
        self.rank = rank

        self.model = get_model(self.model_config)
        get_attention_wrapper().init(
            self.model_config,
            self.parallel_config,
            cache_config.block_size,
            self.device,
        )

        self.sampler: Optional[Sampler] = None
        if self.model.lm_head:
            self.sampler = Sampler(
                self.model.lm_head.weight, self.model.config.vocab_size
            )

        self._prepare_inputs_e2e_timer = CpuTimer(
            CpuOperationMetrics.PREPARE_INPUTS_E2E, rank=self.rank
        )
        self._sampler_e2e_timer = CpuTimer(
            CpuOperationMetrics.SAMPLER_E2E, rank=self.rank
        )
        self._model_execution_e2e_timer = CpuTimer(
            CpuOperationMetrics.MODEL_EXECUTION_E2E, rank=self.rank
        )

    def _prepare_inputs(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_tokens: List[int] = []
        input_positions: List[int] = []
        # need to know prompt chunk sizes for each prompt sequence for sampler
        current_prompt_chunk_lens: List[int] = []

        for seq_metadata in seq_metadata_list:
            if not seq_metadata.is_prompt:
                continue

            prompt_chunk_len = seq_metadata.prompt_chunk_len
            current_prompt_chunk_tokens = (
                seq_metadata.seq.get_next_prompt_chunk_token_ids(prompt_chunk_len)
            )
            # TODO(XY): the same sync bug, just ignore now
            current_prompt_chunk_len = len(current_prompt_chunk_tokens)
            current_prompt_chunk_lens.append(current_prompt_chunk_len)
            processed_prompt_len = seq_metadata.seq.get_num_prompt_tokens_processed()

            current_total_len = processed_prompt_len + current_prompt_chunk_len

            input_tokens.extend(current_prompt_chunk_tokens)
            input_positions.extend(range(processed_prompt_len, current_total_len))

        for seq_metadata in seq_metadata_list:
            if seq_metadata.is_prompt:
                continue

            generation_token = seq_metadata.seq.get_last_token_id()
            input_tokens.append(generation_token)

            context_len = seq_metadata.seq.get_len()
            position = context_len - 1
            input_positions.append(position)
        # Optimization: Pad the input length to be a multiple of 8.
        # This is required for utilizing the Tensor Cores in NVIDIA GPUs.
        input_tokens = pad_to_alignment(input_tokens, multiple_of=8)
        input_positions = pad_to_alignment(input_positions, multiple_of=8)

        # Convert to tensors.
        tokens_tensor = torch.tensor(input_tokens, dtype=torch.long, device=self.device)
        positions_tensor = torch.tensor(
            input_positions, dtype=torch.long, device=self.device
        )

        return tokens_tensor, positions_tensor

    @torch.inference_mode()
    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
    ) -> Tuple[int, int]:
        torch.cuda.set_device(self.device)

        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Enable top-k sampling to reflect the accurate memory usage.
        vocab_size = self.model.config.vocab_size
        sampling_params = SamplingParams(top_p=0.99, top_k=vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs

        seq_metadata_list: List[SequenceMetadata] = []

        if (
            self.scheduler_config.type == SchedulerType.SARATHI
            or self.scheduler_config.type == SchedulerType.SIMPLE_CHUNKING
        ):
            # Profile memory usage with a single `chunk_size` chunk
            # which is the last chunk in the longest supported sequence.
            chunk_size = self.scheduler_config.chunk_size
            seq_len = self.model_config.get_max_model_len()
            chunk_size = min(chunk_size, seq_len)
            
            seq = Sequence(
                seq_id=0,
                prompt=None,
                prompt_token_ids=[0] * seq_len,
                block_size=block_size,
                eos_token_id=1,
                arrival_time=None,
                sampling_params=sampling_params,
            )
            
            seq_metadata = SequenceMetadata(
                seq=seq,
                block_table=None,
                prompt_chunk_len=chunk_size,
            )
            seq_metadata_list.append(seq_metadata)
            
        else:
            # Profile memory usage with max_num_sequences sequences and the total
            # number of tokens equal to max_num_batched_tokens.
            for seq_id in range(max_num_seqs):
                seq_len = max_num_batched_tokens // max_num_seqs + (
                    seq_id < max_num_batched_tokens % max_num_seqs
                )

                seq = Sequence(
                    seq_id=seq_id,
                    prompt=None,
                    prompt_token_ids=[0] * seq_len,
                    block_size=block_size,
                    eos_token_id=1,
                    arrival_time=None,
                    sampling_params=sampling_params,
                )
                seq_metadata = SequenceMetadata(
                    seq=seq,
                    block_table=None,
                    prompt_chunk_len=seq_len,
                )
                seq_metadata_list.append(seq_metadata)

        input_tokens, input_positions = self._prepare_inputs(seq_metadata_list)
        get_attention_wrapper().begin_forward(seq_metadata_list)

        if AttentionBackend.is_vATTN(self.model_config.attention_backend):
            get_attention_wrapper().is_profiling_iteration = True
        # Execute the model.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        self.model(
            hidden_states=input_tokens,
            positions=input_positions,
            kv_caches=[None] * num_layers,
        )

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        total_gpu_memory = get_gpu_memory()
        # print(f"peak_memory: {peak_memory}, total_gpu_memory: {total_gpu_memory}")
        physical_memory = int(total_gpu_memory * gpu_memory_utilization - peak_memory)
        cache_block_size = get_cache_engine(self.model_config.attention_backend).get_cache_block_size(
            block_size, self.model_config, self.parallel_config
        )
        num_gpu_blocks = int(
            physical_memory // cache_block_size
        )
        num_gpu_blocks = max(num_gpu_blocks, 0)
        torch.cuda.empty_cache()

        get_attention_wrapper().end_forward()

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)
        
        return num_gpu_blocks, physical_memory

    def run(
        self,
        seq_metadata_list: List[SequenceMetadata],
        gpu_cache: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Prepare input tensors.
        with self._prepare_inputs_e2e_timer:
            input_tokens, input_positions = self._prepare_inputs(seq_metadata_list)

        get_attention_wrapper().begin_forward(seq_metadata_list)
        
            
        with self._model_execution_e2e_timer:
            # Execute the model.
            try:
                output = self.model(
                    hidden_states=input_tokens,
                    positions=input_positions,
                    kv_caches=gpu_cache,
                )
            except RuntimeError as e:
                logger.error(
                    f"RuntimeError: {e} for seq_metadata_list: {seq_metadata_list}"
                )
                raise e

        with self._sampler_e2e_timer:
            if self.sampler is not None:
                output = self.sampler(output, seq_metadata_list)

        get_attention_wrapper().end_forward()

        return output
