import json
import logging
import os
import time
import threading
from typing import Tuple, Set

import ray
import wandb
import math
from tqdm import tqdm

from sarathi import LLMEngine, SamplingParams
from sarathi.benchmark.config import Config
from sarathi.benchmark.entities import Request
from sarathi.benchmark.request_generator import RequestGeneratorRegistry
from sarathi.benchmark.custom_types import ReplicaResourceMapping, ResourceMapping
from sarathi.benchmark.utils.random import set_seeds
from sarathi.engine.base_llm_engine import BaseLLMEngine
from sarathi.core.datatypes.request_output import RequestOutput
from sarathi.metrics.metrics_store import MetricsStore
from sarathi.utils import get_ip
from sarathi.benchmark.latency_tracker import LatencyTracker
from sarathi.engine.pipeline_parallel_llm_engine import PipelineParallelLLMEngine

from sarathi.config import (
    MetricsConfig,
    ModelConfig,
    ParallelConfig,
    UpgradeStrategy,
)

logger = logging.getLogger(__name__)
import subprocess
import os
import xml.etree.ElementTree as ET
import time
import torch
import threading
import queue

def get_gpu_memory_info():
    """Get GPU memory usage directly from nvidia-smi"""
    try:
        cmd = ['nvidia-smi', '-q', '-x']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return None
            
        root = ET.fromstring(result.stdout)
        gpu_info = []
        
        for gpu in root.findall("gpu"):
            memory = gpu.find("fb_memory_usage")
            gpu_dict = {
                'id': gpu.find("minor_number").text,
                'total': memory.find("total").text.replace('MiB', '').strip(),
                'used': memory.find("used").text.replace('MiB', '').strip(),
                'free': memory.find("free").text.replace('MiB', '').strip(),
                'processes': []
            }
            
            processes = gpu.find("processes")
            if processes is not None:
                for process in processes.findall("process_info"):
                    pid = process.find("pid").text
                    used_memory = process.find("used_memory").text
                    gpu_dict['processes'].append({
                        'pid': pid,
                        'used_memory': used_memory.replace('MiB', '').strip()
                    })
                    
            gpu_info.append(gpu_dict)
            
        return gpu_info
    except Exception as e:
        logger.error(f"Error getting GPU info: {str(e)}")
        return None

def log_memory_usage(tag=""):
    """Log both PyTorch and nvidia-smi memory stats"""
    try:
        # PyTorch memory stats
        allocated_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB
        reserved_memory = torch.cuda.memory_reserved() / (1024 * 1024)    # Convert to MB
        max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024) # Convert to MB
        
        logger.info(f"=== Memory Usage {tag} ===")
        logger.info(f"PyTorch Memory - Allocated: {allocated_memory:.2f}MB, Reserved: {reserved_memory:.2f}MB, Max Allocated: {max_allocated:.2f}MB")
        
        # nvidia-smi stats
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            for gpu in gpu_info:
                logger.info(f"GPU {gpu['id']} (nvidia-smi) - Used: {gpu['used']}MB, Free: {gpu['free']}MB, Total: {gpu['total']}MB")
                if gpu['processes']:
                    process_info = [f"PID {p['pid']}: {p['used_memory']}MB" for p in gpu['processes']]
                    logger.info(f"Active processes: {', '.join(process_info)}")
    except Exception as e:
        logger.error(f"Error logging memory usage: {str(e)}")



def calculate_model_params_per_gpu(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
) -> int:
    """
    Calculate the number of parameters per GPU based on model and parallel configs.
    
    Args:
        model_config: ModelConfig instance containing model architecture details
        parallel_config: ParallelConfig instance containing parallelization strategy
    
    Returns:
        Number of parameters per GPU
    """
    # Get basic model dimensions
    hidden_size = model_config.get_hidden_size()
    total_num_layers = model_config.get_total_num_layers()
    num_q_heads = model_config.get_num_q_heads(parallel_config)
    num_kv_heads = model_config.get_num_kv_heads(parallel_config)
    head_size = model_config.get_head_size()
    
    # Calculate attention parameters per layer
    q_params = hidden_size * num_q_heads * head_size
    k_params = hidden_size * num_kv_heads * head_size
    v_params = hidden_size * num_kv_heads * head_size
    o_params = num_q_heads * head_size * hidden_size
    
    attn_params_per_layer = q_params + k_params + v_params + o_params
    
    # Calculate feedforward parameters per layer
    # Typically 4x hidden size for intermediate layer
    ff_params_per_layer = 4 * hidden_size * hidden_size + hidden_size * hidden_size
    
    # Total parameters per layer
    params_per_layer = attn_params_per_layer + ff_params_per_layer
    
    # Account for embedding layers and final layer norm
    embedding_params = model_config.get_hidden_size() * model_config.get_max_model_len()
    final_norm_params = hidden_size
    
    # Total parameters
    total_params = (params_per_layer * total_num_layers + 
                   embedding_params + final_norm_params)
    
    # Divide by parallel sizes to get params per GPU
    params_per_gpu = total_params // (
        parallel_config.pipeline_parallel_size * 
        parallel_config.tensor_parallel_size
    )
    
    return params_per_gpu

def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()

def calculate_block_size(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    attention_backend: str,
) -> int:
    """
    Calculate the block size for KV cache based on model configuration and attention backend.
    
    Args:
        model_config: Model configuration containing model architecture details
        parallel_config: Parallel configuration containing tp/pp sizes
        attention_backend: Attention backend type
        
    Returns:
        Total block size in bytes
    """
    PAGE_SIZE = 2 * 1024 * 1024  # 2MB in bytes
    
    # First calculate the initial block size from page size
    num_kv_heads_per_worker = (
        model_config.hf_config.num_key_value_heads // 
        parallel_config.tensor_parallel_size
    )
    
    head_dim = (
        model_config.hf_config.hidden_size // 
        model_config.hf_config.num_attention_heads
    )
    
    # Initial block size calculation
    block_size = PAGE_SIZE // num_kv_heads_per_worker  # Divide by KV heads per worker
    block_size = block_size // head_dim  # Divide by head dimension
    
    # If using megacache, further divide by layers per worker
    if "megacache" in attention_backend.lower():
        layers_per_worker = (
            model_config.hf_config.num_hidden_layers // 
            parallel_config.pipeline_parallel_size
        )
        block_size = block_size // layers_per_worker
    
    # Divide by element size
    dtype_size = _get_dtype_size(model_config.dtype)
    block_size = block_size // dtype_size
    
    # Now calculate the total cache block size
    head_size = model_config.get_head_size()
    num_heads = model_config.get_num_kv_heads(parallel_config)
    num_layers = model_config.get_num_layers(parallel_config)
    
    # Calculate sizes for key and value caches
    key_cache_block = block_size * num_heads * head_size
    value_cache_block = key_cache_block
    
    # Calculate total size including both key and value caches across all layers
    total = num_layers * (key_cache_block + value_cache_block)
    
    # Final size in bytes
    return dtype_size * total

def calculate_required_blocks(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    new_parallel_config: ParallelConfig,
    attention_backend: str
) -> int:
    """
    Calculate required blocks for model upgrade based on GPU memory and model parameters.
    
    Args:
        model_config: ModelConfig instance
        parallel_config: Current ParallelConfig
        new_parallel_config: New ParallelConfig for the upgrade
        attention_backend: Attention backend type
        
    Returns:
        Number of blocks required for the upgrade
    """

    # Calculate the block size
    current_block_size = calculate_block_size(
        model_config=model_config,
        parallel_config=parallel_config,
        attention_backend=attention_backend
    )
    
    # Calculate parameters per GPU for new config
    new_params_per_gpu = calculate_model_params_per_gpu(
        model_config, new_parallel_config
    )
    
    # Each parameter typically requires 4 bytes (float32)
    # or 2 bytes (float16/bfloat16) depending on dtype
    bytes_per_param = 2 if model_config.dtype in ["float16", "bfloat16"] else 4
    new_model_memory_per_gpu = new_params_per_gpu * bytes_per_param
    
    # Not Add safety margin (10% for optimizer states, gradients, etc.)
    safety_margin = 1
    required_memory = new_model_memory_per_gpu * safety_margin
    
    # Calculate number of blocks needed
    required_blocks = math.ceil(required_memory / current_block_size)
    logger.info("Memory Calculation Details:")
    logger.info(f"New model weights per GPU: {new_params_per_gpu:,} parameters")
    logger.info(f"Memory per parameter: {bytes_per_param} bytes")
    logger.info(f"Raw model memory needed: {new_model_memory_per_gpu / (1024*1024):,.2f} MB")
    logger.info(f"With {safety_margin}x safety margin: {required_memory / (1024*1024):,.2f} MB")
    logger.info("\nBlock Size Details:")
    logger.info(f"Current block size: {current_block_size / (1024*1024):,.2f} MB")
    logger.info(f"Required blocks for upgrade: {required_blocks:,}")
    
    return required_blocks

class UpgradeState:
    """Shared state for coordinating overlap serving during upgrade"""
    def __init__(self):
        self.preemption_complete = False
        self.weights_loaded = False
        self._lock = threading.Lock()
        
    def set_preemption_complete(self):
        with self._lock:
            self.preemption_complete = True
            
    def set_weights_loaded(self):
        with self._lock:
            self.weights_loaded = True
            
    def is_preemption_complete(self):
        with self._lock:
            return self.preemption_complete
            
    def is_weights_loaded(self):
        with self._lock:
            return self.weights_loaded

class BenchmarkRunner:
    def __init__(
        self,
        replica_id: int,
        config: Config,
        replica_resource_mapping: ResourceMapping = [],
        is_new_runner: bool = False
    ) -> None:
        self._replica_id = replica_id
        self._config = config
        self._num_replicas = self._config.cluster_num_replicas
        self._is_new_runner = is_new_runner
        self.upgrade_state = None

        # Track request states
        self._request_states = {}  # Store all request states
        self._finished_requests = {}  # Store completed requests
        self._pending_requests = {}  # Store in-progress requests

        self._time_limit = self._config.time_limit
        if not self._time_limit:
            self._time_limit = float("inf")

        # Determine output directory based on upgrade strategy
        base_output_dir = f"{self._config.output_dir}/replica_{replica_id}"
        if self._config.upgrade_strategy == UpgradeStrategy.DECODE_UPGRADE:
            base_output_dir = f"{base_output_dir}/decode_upgrade"
            if self._is_new_runner:
                base_output_dir = f"{base_output_dir}_new"
        elif self._config.upgrade_strategy == UpgradeStrategy.PREFILL_UPGRADE:
            base_output_dir = f"{base_output_dir}/prefill_upgrade"
            if self._is_new_runner:
                base_output_dir = f"{base_output_dir}_new"
        elif self._config.upgrade_strategy == UpgradeStrategy.BASIC_UPGRADE:
            base_output_dir = f"{base_output_dir}/basic_upgrade"
        else:
            base_output_dir = f"{base_output_dir}/no_upgrade"
        
        output_dir = base_output_dir
        logger.info(f"Output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # Initialize requests based on runner type
        self._requests = None
        if not self._is_new_runner:
            set_seeds(config.seed)
            request_generator = RequestGeneratorRegistry.get_from_str(
                self._config.request_generator_provider, self._config
            )
            self._requests = request_generator.generate()
            self._requests = self._requests[self._replica_id :: self._num_replicas]
            logger.info(f"Replica {self._replica_id} has {len(self._requests)} requests")

        # Configure wandb settings
        if self._num_replicas == 1:
            wandb_project = self._config.metrics_store_wandb_project
            wandb_group = self._config.metrics_store_wandb_group
            wandb_run_name = self._config.metrics_store_wandb_run_name
        else:
            wandb_project = None
            wandb_group = None
            wandb_run_name = None

        # Configure scheduler settings
        chunk_size = None
        if self._config.replica_scheduler_provider == "sarathi":
            chunk_size = self._config.sarathi_scheduler_chunk_size
        elif self._config.replica_scheduler_provider == "simple_chunking":
            chunk_size = self._config.simple_chunking_scheduler_chunk_size

        # Set engine type for upgrade
        upgrade_engine_type = "new" if self._is_new_runner else "old"

        # Initialize LLM engine
        self._llm_engine = LLMEngine.from_engine_args(
            replica_id=replica_id,
            replica_resource_mapping=replica_resource_mapping,
            output_dir=output_dir,
            model=self._config.model_name,
            tokenizer=self._config.model_name,
            tensor_parallel_size=self._config.model_tensor_parallel_degree,
            pipeline_parallel_size=self._config.model_pipeline_parallel_degree,
            attention_backend=self._config.model_attention_backend,
            seed=self._config.seed,
            dtype="float16",
            load_format=self._config.model_load_format,
            gpu_memory_utilization=self._config.gpu_memory_utilization,
            max_model_len=self._config.model_max_model_len,
            block_size=self._config.model_block_size,
            scheduler_type=self._config.replica_scheduler_provider,
            max_num_seqs=self._config.replica_scheduler_max_batch_size,
            chunk_size=chunk_size,
            enable_dynamic_chunking_schedule=self._config.sarathi_scheduler_enable_dynamic_chunking_schedule,
            low_chunk_size=self._config.sarathi_scheduler_low_chunk_size,
            high_chunk_size=self._config.sarathi_scheduler_high_chunk_size,
            chunk_schedule_max_tokens=self._config.sarathi_scheduler_chunk_schedule_max_tokens,
            chunk_schedule_stages=self._config.sarathi_scheduler_chunk_schedule_stages,
            max_num_batched_tokens=self._config.vllm_scheduler_max_tokens_in_batch,
            write_metrics=self._config.write_metrics,
            enable_chrome_trace=self._config.write_chrome_trace,
            wandb_project=wandb_project,
            wandb_group=wandb_group,
            wandb_run_name=wandb_run_name,
            wandb_sweep_id=self._config.metrics_store_wandb_sweep_id,
            wandb_run_id=self._config.metrics_store_wandb_run_id,
            enable_op_level_metrics=self._config.metrics_store_enable_op_level_metrics,
            enable_cpu_op_level_metrics=self._config.metrics_store_enable_cpu_op_level_metrics,
            enable_request_outputs=self._config.metrics_store_enable_request_outputs,
            keep_individual_batch_metrics=self._config.metrics_store_keep_individual_batch_metrics,
            trust_remote_code=True,
            time = self._config.upgrade_time,
            strategy = self._config.upgrade_strategy,
            required_blocks = self._config.upgrade_required_blocks,
            engine_type=upgrade_engine_type,
        )

        if not self._is_new_runner or self._config.upgrade_strategy == UpgradeStrategy.BASIC_UPGRADE:
            self._llm_engine.init_rest()
        
        self._latency_tracker = LatencyTracker(output_dir)
        self.is_pipeline_engine = isinstance(self._llm_engine, PipelineParallelLLMEngine)

    def set_upgrade_state(self, upgrade_state: UpgradeState) -> None:
        self.upgrade_state = upgrade_state

    def _get_input_params(self, request: Request, first_request_time: float) -> dict:
        sampling_params = SamplingParams(
            ignore_eos=True,
            max_tokens=request.num_decode_tokens,
            temperature=0,
            top_p=1.0,
        )
        prompt_token_ids = [1] * request.num_prefill_tokens

        return {
            "prompt": None,
            "prompt_token_ids": prompt_token_ids,
            "sampling_params": sampling_params,
            "arrival_time": first_request_time + request.arrived_at,
        }
    
    def f(self) -> None:
        # warmup the engine
        self._llm_engine.add_request(
            **self._get_input_params(self._requests[0], time.monotonic())
        )

        is_completed = False
        while not is_completed:
            step_outputs = self._llm_engine.step()
            is_completed = step_outputs[0].finished

        self._llm_engine.reset_metrics()
    
    def _add_requests(self) -> None:
        """Add all requests and initialize their states"""
        index = 0
        first_request_time = time.monotonic()
        while index < len(self._requests):
            request = self._requests[index]
            input_params = self._get_input_params(request, first_request_time)
            
            # Add request to engine
            self._llm_engine.add_request(**input_params)
            
            # Initialize request state
            # Convert index to string for consistent seq_id type
            seq_id = str(index)
            self._request_states[seq_id] = {
                'seq_id': seq_id,
                'prompt': input_params['prompt'],
                'prompt_token_ids': input_params['prompt_token_ids'],
                'current_text': "",
                'current_token_ids': [],
                'finished': False,
                'original_request': request,
                'start_time': time.monotonic()  # Add start time
            }
            # Add to pending queue
            self._pending_requests[seq_id] = self._request_states[seq_id]
            
            index += 1

    def _update_request_state(self, output: RequestOutput) -> None:
        """Update internal state tracking for a request"""
        current_time = time.monotonic()
        seq_id = str(output.seq_id)  # Convert to string for consistent key type
        
        if seq_id not in self._request_states:
            # If somehow we don't have state for this request, initialize it
            self._request_states[seq_id] = {
                'seq_id': seq_id,
                'prompt': output.prompt,
                'prompt_token_ids': output.prompt_token_ids,
                'current_text': "",
                'current_token_ids': [],
                'finished': False,
                'original_request': self._requests[int(seq_id)],
                'start_time': time.monotonic()
            }
        
        self._request_states[seq_id] = {
            'seq_id': seq_id,
            'prompt': output.prompt,
            'prompt_token_ids': output.prompt_token_ids,
            'current_text': output.text,
            'current_token_ids': output.token_ids,
            'finished': output.finished,
            'finish_reason': output.finish_reason,
            'original_request': self._requests[int(seq_id)],
            'start_time': self._request_states[seq_id]['start_time']  # Preserve start time
        }


        if output.finished:
            self._request_states[seq_id]['end_time'] = current_time
            latency = current_time - self._request_states[seq_id]['start_time']
            self._request_states[seq_id]['latency'] = latency
            self._finished_requests[seq_id] = self._request_states[seq_id]
            if seq_id in self._pending_requests:
                del self._pending_requests[seq_id]
            
            # Log the latency
            self._latency_tracker.log_latency(seq_id, latency)
            
            # Log the latency
            logger.info(f"Request {seq_id} completed with latency: {self._request_states[seq_id]['latency']:.4f} seconds, with reason: {output.finish_reason}")
        else:
            self._pending_requests[seq_id] = self._request_states[seq_id]   
            # logger.info(f"Request {seq_id} updated with output tokens: {len(self._pending_requests[seq_id]['current_token_ids'])}, prompt len is {len(self._pending_requests[seq_id]['prompt_token_ids'])}")
    
    def prepare_for_decode_upgrade(self) -> dict:
        """Prepare for upgrade by preempting sequences"""
        required_blocks = self._config.upgrade_required_blocks
        self._llm_engine.prepare_for_decode_upgrade(required_blocks)
        return {"status": "PREEMPTION_COMPLETE"}
    
    def prepare_for_prefill_upgrade(self) -> dict:
        """Prepare for upgrade by preempting sequences"""
        required_blocks = self._config.upgrade_required_blocks
        self._llm_engine.prepare_for_prefill_upgrade(required_blocks)
        return {"status": "PREEMPTION_COMPLETE"}

    def run_during_upgrade(self) -> dict:
        """Continue running with reduced capacity during upgrade"""
        self._llm_engine.scheduler.set_upgrade()
        
        num_processed_requests = 0
        num_steps = 0
        pbar = tqdm(
            total=len(self._requests),
            desc=f"Replica {self._replica_id} running during upgrade",
        )
        is_pipeline_engine = isinstance(self._llm_engine, PipelineParallelLLMEngine)
        if is_pipeline_engine:
            logger.info("Starting pipeline engine execution loops")
            self._llm_engine.signal_start_scheduling()
        while not self.upgrade_state.is_weights_loaded():
            step_outputs = self._llm_engine.step()
            num_steps += 1

            for output in step_outputs:
                self._update_request_state(output)
                if output.finished:
                    num_processed_requests += 1
                    pbar.update(1)
        logger.info("Stop serving during upgrade")
        if is_pipeline_engine:
            self._llm_engine.signal_stop_scheduling()
            logger.info("Stopping pipeline engine execution loops")
            # while self._llm_engine.has_inflight_batches():
            #     logger.info(f"Waiting for inflight batches to complete")
            #     time.sleep(0.01)  # Small sleep to prevent busy waiting
        pbar.close()
        progress = self.save_progress()
        return {"status": "READY_FOR_HANDOVER", "progress": progress}

    def _run_normal(self) -> str:
        """Original run logic until upgrade needed or completion"""
        if self._config.enable_profiling:
            self._llm_engine.start_profiling()

        num_processed_requests = 0
        num_steps = 0
        pbar = tqdm(
            total=len(self._requests),
            desc=f"Replica {self._replica_id} processed requests",
        )
        start_time = time.monotonic()

        # Only needed for pipeline engine
        is_pipeline_engine = isinstance(self._llm_engine, PipelineParallelLLMEngine)

        while num_processed_requests < len(self._requests):
            elapsed_time = time.monotonic() - start_time
            if not self._is_new_runner and self._config.upgrade_time and elapsed_time > self._config.upgrade_time:
                if is_pipeline_engine:
                    # First time hitting upgrade time - signal stop
                    logger.info(f"Replica {self._replica_id} reached upgrade time after {elapsed_time:.2f} seconds")
                    self._llm_engine.signal_stop_scheduling()
                    logger.info(f"Replica {self._replica_id} signaled for upgrade after {elapsed_time:.2f} seconds")
                    while self._llm_engine.has_inflight_batches():
                        logger.info(f"Waiting for inflight batches to complete")
                        time.sleep(0.01)  # Small sleep to prevent busy waiting
                
                # For non-pipeline engine, stop immediately
                logger.info(f"Replica {self._replica_id} stopping for upgrade after {elapsed_time:.2f} seconds")
                return "UPGRADE_NEEDED"

            step_outputs = self._llm_engine.step()
            num_steps += 1

            for output in step_outputs:
                self._update_request_state(output)
                if output.finished:
                    num_processed_requests += 1
                    pbar.update(1)

        end_time = time.monotonic()
        pbar.close()
        
        logger.info(
            f"Replica {self._replica_id} exiting after processing {num_processed_requests} requests "
            f"({num_steps} iterations), Total time taken: {end_time - start_time:.2f} seconds"
        )
        if is_pipeline_engine:
            logger.info("Stopping pipeline engine execution loops")
            self._llm_engine.stop_execution_loops()

        if self._config.enable_profiling:
            self._llm_engine.stop_profiling()

        return "COMPLETED"
    
    def save_progress(self) -> dict:            
        progress = {
            'finished_requests': {},
            'pending_requests': {},
        }
        
        # Save finished requests with their latencies
        for seq_id, state in self._finished_requests.items():
            progress['finished_requests'][seq_id] = {
                'prompt': state['prompt'],
                'prompt_token_ids': state['prompt_token_ids'],
                'generated_text': state['current_text'],
                'generated_token_ids': state['current_token_ids'],
                'latency': state.get('latency', None)
            }
        
        # Save pending requests with their start times
        for seq_id, state in self._pending_requests.items():
            logger.info(f"Saving progress for pending request {seq_id}, prompt_token_ids is {len(state['prompt_token_ids'])}, generated_token_ids_so_far is {len(state['current_token_ids'])}")
            progress['pending_requests'][seq_id] = {
                'prompt': state['prompt'],
                'prompt_token_ids': state['prompt_token_ids'],
                'generated_text_so_far': state['current_text'],
                'generated_token_ids_so_far': state['current_token_ids'],
                'original_request': state['original_request'],
                'start_time': state['start_time']  # Include start time for continuing latency tracking
            }
        
        return progress
    
    def load_progress(self, progress: dict) -> None:
        """Load saved progress to resume requests"""
        if progress is None:
            return
                
        self._finished_requests = {}
        self._pending_requests = {}
        self._requests = []  # Initialize requests list
        
        # Restore finished requests state with their latencies
        for seq_id, state in progress['finished_requests'].items():
            self._finished_requests[seq_id] = {
                'seq_id': seq_id,
                'prompt': state['prompt'],
                'prompt_token_ids': state['prompt_token_ids'],
                'current_text': state['generated_text'],
                'current_token_ids': state['generated_token_ids'],
                'finished': True,
                'latency': state['latency']  # Keep original latency
            }
            # Add to requests list to maintain correct total count
            self._requests.append(None)  # Placeholder for finished request
        
        # For pending (executed but unfinished) requests, modify prompt to include generated tokens
        for seq_id, state in progress['pending_requests'].items():
            request = state['original_request']
            original_start_time = state['start_time']  # Get original start time
            # Add request to the requests list
            self._requests.append(request)
            
            logger.info(f"Resuming request {seq_id} with generated tokens {len(state['generated_token_ids_so_far'])}")
            if len(state['generated_token_ids_so_far']) > 0:  # If request was partially executed
                # Combine original prompt tokens with generated tokens as new prompt
                new_prompt_token_ids = state['prompt_token_ids'] + state['generated_token_ids_so_far']
                sampling_params = SamplingParams(
                    ignore_eos=True,
                    max_tokens=request.num_decode_tokens - len(state['generated_token_ids_so_far']),
                    temperature=0,
                    top_p=1.0,
                )
                self._llm_engine.add_request(
                    prompt=None,
                    prompt_token_ids=new_prompt_token_ids,
                    sampling_params=sampling_params,
                    arrival_time=time.monotonic()
                )
                
                # Initialize request state preserving start time
                self._request_states[seq_id] = {
                    'seq_id': seq_id,
                    'prompt': state['prompt'],
                    'prompt_token_ids': new_prompt_token_ids,
                    'current_text': state['generated_text_so_far'],
                    'current_token_ids': state['generated_token_ids_so_far'],
                    'finished': False,
                    'original_request': request,
                    'start_time': original_start_time  # Preserve original start time
                }
            else:  # For never executed requests, add original request
                self._llm_engine.add_request(
                    **self._get_input_params(request, time.monotonic())
                )
                
                # Initialize request state with original start time
                self._request_states[seq_id] = {
                    'seq_id': seq_id,
                    'prompt': state['prompt'],
                    'prompt_token_ids': state['prompt_token_ids'],
                    'current_text': "",
                    'current_token_ids': [],
                    'finished': False,
                    'original_request': request,
                    'start_time': original_start_time  # Preserve original start time
                }
            
            # Add to pending queue
            self._pending_requests[seq_id] = self._request_states[seq_id]
            
        logger.info(f"Loaded progress with {len(self._finished_requests)} finished requests and {len(self._pending_requests)} pending requests")
        logger.info(f"Total requests in self._requests: {len(self._requests)}")

    def run(self) -> None:
        """Main run method that handles different upgrade strategies"""
        self._llm_engine.reset_metrics()
        if not self._is_new_runner:
            self._add_requests()

        if self._config.upgrade_strategy == UpgradeStrategy.DECODE_UPGRADE or self._config.upgrade_strategy == UpgradeStrategy.PREFILL_UPGRADE:
            return self._run_with_overlap()
        elif self._config.upgrade_strategy == UpgradeStrategy.BASIC_UPGRADE:
            return self._run_without_overlap()
        else:
            return self._run_normal()  # Just run normally for NO_UPGRADE

    def _run_with_overlap(self) -> dict:
        """Run with overlap serving during upgrade"""
        status = self._run_normal()
    
        if status == "UPGRADE_NEEDED":
            # Initialize upgrade state
            if self.upgrade_state is None:
                logger.error("Upgrade state not set. Cannot proceed with overlap serving.")
                return {"status": "ERROR"}

            if self._config.upgrade_strategy == UpgradeStrategy.DECODE_UPGRADE:
                preemption_result = self.prepare_for_decode_upgrade()
            else:
                preemption_result = self.prepare_for_prefill_upgrade()
            if preemption_result["status"] != "PREEMPTION_COMPLETE":
                logger.error("Preemption failed")
                return {"status": "ERROR"}
            
            self.upgrade_state.set_preemption_complete()
            return {"status": "UPGRADE_NEEDED"}

        if status == "COMPLETED":
            self._latency_tracker.plot_cdf()
            stats = self._latency_tracker.get_statistics()
            logger.info("Latency Statistics:")
            for key, value in stats.items():
                logger.info(f"{key}: {value:.2f}s")
            
            self._llm_engine.cleanup()
            return "COMPLETED"

    def _run_without_overlap(self) -> dict:
        """Run without overlap serving during upgrade"""
        status = self._run_normal()

        if status == "UPGRADE_NEEDED":
            progress = self.save_progress()
            metrics = self._llm_engine.get_metric_store()
            self._llm_engine.cleanup()
            return {"status": "UPGRADE_NEEDED", "progress": progress, "metrics": metrics}

        self._llm_engine.pull_worker_metrics()
        metric_store = self._llm_engine.get_metric_store()
        self._llm_engine.cleanup()
        return metric_store
    
    def run_during_overlap(self) -> dict:
        """Continue serving requests during overlap phase"""
        serving_result = self.run_during_upgrade()
        if serving_result["status"] == "READY_FOR_HANDOVER":
            return serving_result
        else:
            logger.error("Serving during upgrade failed")
            return {"status": "ERROR"}
    
class BenchmarkRunnerLauncher:
    def __init__(
        self, 
        config: Config, 
        new_config: Config = None,
    ) -> None:
        self._config = config
        self._new_config = new_config
        self._is_multi_replica = self._config.cluster_num_replicas > 1


        ray.init(ignore_reinit_error=True)
        
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
        
        required_blocks = self.calculate_upgrade_blocks()
        self._config.upgrade_required_blocks = required_blocks
        logger.info(f"Required blocks for upgrade: {required_blocks}")
    
    def calculate_upgrade_blocks(self) -> int:
        """
        Calculate required blocks for model upgrade using engine configs.
        
        Returns:
            int: Number of blocks required for the upgrade
        """
        if not self._new_config:
            raise ValueError("New config is required to calculate upgrade blocks")

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
        
        required_blocks = calculate_required_blocks(
            model_config=model_config,
            parallel_config=parallel_config,
            new_parallel_config=new_parallel_config,
            attention_backend=self._config.model_attention_backend
        )
        
        logger.info(f"Calculated required blocks for upgrade: {required_blocks}")
        return required_blocks
    
    def _validate_cluster_resources(self):
        num_replicas = self._config.cluster_num_replicas
        tp_degree = self._config.model_tensor_parallel_degree
        pp_degree = self._config.model_pipeline_parallel_degree
        num_gpus_required = num_replicas * tp_degree * pp_degree

        available_resources = ray.available_resources()

        assert (
            available_resources["GPU"] >= num_gpus_required
        ), f"Insufficient GPUs. Required: {num_gpus_required}, Available: {available_resources['GPU']}"

    def _get_replica_resource_mapping(self) -> ReplicaResourceMapping:
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

    def _create_runners(self):
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

    def run_with_upgrade(self):
        """Run benchmark with configurable upgrade strategy"""
        if not self._is_multi_replica:
            if self._config.upgrade_strategy == UpgradeStrategy.DECODE_UPGRADE or self._config.upgrade_strategy == UpgradeStrategy.PREFILL_UPGRADE:
                return self._run_with_overlap_upgrade()
            elif self._config.upgrade_strategy == UpgradeStrategy.BASIC_UPGRADE:
                return self._run_without_overlap_upgrade()
            else:
                return self._run_normal()  # Just run normally for NO_UPGRADE
        else:
            # Handle multi-replica upgrade
            pass  # TODO: Implement multi-replica upgrade support

    def _run_with_overlap_upgrade(self):
        """Run single replica upgrade with overlap serving"""
        upgrade_state = UpgradeState()
        new_runner = None
        saved_progress = None

        def init_new_runner():
            nonlocal new_runner
            new_runner = BenchmarkRunner(
                0, 
                self._new_config, 
                self.replica_resource_mapping["0"],
                is_new_runner=True
            )
            upgrade_state.set_weights_loaded()
            logger.info("New runner initialized")

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
                # self._runner._llm_engine.cleanup()
                # del self._runner
                # log_memory_usage("AFTER RUNNER DELETION")

                saved_progress = result["progress"]

                # Wait for weight loading to complete
                init_thread.join()
                # log_memory_usage("AFTER INIT THREAD JOIN")
                if old_workers:
                    logger.info(f"Cleaning up {len(old_workers)} old Ray workers")
                    for worker in old_workers:
                        try:
                            ray.kill(worker)
                        except Exception as e:
                            logger.error(f"Error cleaning up old worker: {e}")

                # Force CUDA cleanup
                # torch.cuda.empty_cache()
                # torch.cuda.synchronize()
                # torch.cuda.ipc_collect()
                # log_memory_usage("AFTER FORCE CLEANUP")

                if new_runner is not None:
                    new_runner._llm_engine.init_rest()
                    new_runner.load_progress(saved_progress)
                    final_result = new_runner.run()
                    logger.info(f"New runner completed with status: {final_result}")
                else:
                    logger.error("New runner initialization failed")
        wandb.finish()
    
    def _run_without_overlap_upgrade(self):
        """Run single replica upgrade without overlap serving"""
        result = self._runner.run()
        
        if isinstance(result, dict) and result["status"] == "UPGRADE_NEEDED":
            progress = result["progress"]
            
            # Clean up Ray resources
            ray.shutdown()
            ray.init(ignore_reinit_error=True)
            
            # Create new runner
            new_runner = BenchmarkRunner(
                0, 
                self._new_config, 
                self.replica_resource_mapping["0"],
                is_new_runner=True
            )
            new_runner.load_progress(progress)
            
            # Run second phase
            new_runner.run()
        wandb.finish()

    def _run_normal(self):
        """Run without any upgrade"""
        result = self._runner.run()
        wandb.finish()
        return result