"""
BenchmarkRunner for executing LLM benchmarks with upgrade capabilities.
"""

import logging
import os
import time
from typing import Dict, Any, Optional

from tqdm import tqdm

from sarathi import LLMEngine, SamplingParams
from sarathi.benchmark.config import Config
from sarathi.benchmark.entities import Request
from sarathi.benchmark.request_generator import RequestGeneratorRegistry
from sarathi.benchmark.custom_types import ResourceMapping
from sarathi.benchmark.utils.random import set_seeds

# Import from the upgrade_utils package
from sarathi.benchmark.upgrade_utils.latency_tracker import LatencyTracker
from sarathi.benchmark.upgrade_utils.upgrade_state import UpgradeState

from sarathi.core.datatypes.request_output import RequestOutput
from sarathi.engine.pipeline_parallel_llm_engine import PipelineParallelLLMEngine

from sarathi.config import UpgradeStrategy

logger = logging.getLogger(__name__)

class BenchmarkRunner:
    """
    Runner for executing LLM benchmarks with upgrade capabilities.
    
    This class handles running benchmark requests against an LLM model with support
    for different upgrade strategies:
    - No upgrade (normal execution)
    - No-serve upgrade (stop serving during upgrade)
    - Decode-only upgrade (continue serving decode requests during upgrade)
    - Prefill-only upgrade (continue serving prefill requests during upgrade)
    """
    
    def __init__(
        self,
        replica_id: int,
        config: Config,
        replica_resource_mapping: ResourceMapping = [],
        is_new_runner: bool = False
    ) -> None:
        """
        Initialize the benchmark runner.
        
        Args:
            replica_id: ID of the replica (for multi-replica setups)
            config: Benchmark configuration
            replica_resource_mapping: Mapping of resources for this replica
            is_new_runner: Whether this is a new runner for post-upgrade execution
        """
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
        if self._config.upgrade_serving_strategy == UpgradeStrategy.ServingStrategy.DECODE_ONLY:
            base_output_dir = f"{base_output_dir}/decode_upgrade"
            if self._is_new_runner:
                base_output_dir = f"{base_output_dir}_new"
        elif self._config.upgrade_serving_strategy == UpgradeStrategy.ServingStrategy.PREFILL_ONLY:
            base_output_dir = f"{base_output_dir}/prefill_upgrade"
            if self._is_new_runner:
                base_output_dir = f"{base_output_dir}_new"
        elif self._config.upgrade_serving_strategy == UpgradeStrategy.ServingStrategy.NO_SERVE:
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
            time=self._config.upgrade_time,
            strategy=self._config.upgrade_serving_strategy,
            required_blocks=self._config.upgrade_required_blocks,
            pages_per_block=self._config.pages_per_block,
            engine_type=upgrade_engine_type,
            drain_strategy=self._config.upgrade_drain_strategy,
            drain_timeout=self._config.upgrade_drain_timeout,
            kickout_strategy=self._config.upgrade_kickout_strategy,
            selection_policy=self._config.upgrade_selection_policy,
            serving_strategy=self._config.upgrade_serving_strategy,
            reschedule_policy=self._config.upgrade_reschedule_policy,
        )
        self._llm_engine.upgrade_config.drain_strategy 

        if not self._is_new_runner or self._config.upgrade_serving_strategy == UpgradeStrategy.ServingStrategy.NO_SERVE:
            self._llm_engine.init_rest()
        
        self._latency_tracker = LatencyTracker(output_dir)
        self.is_pipeline_engine = isinstance(self._llm_engine, PipelineParallelLLMEngine)

    def set_upgrade_state(self, upgrade_state: UpgradeState) -> None:
        """Set the upgrade state object for coordination during upgrade."""
        self.upgrade_state = upgrade_state

    def _get_input_params(self, request: Request, first_request_time: float) -> dict:
        """Get input parameters for a request."""
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
    
    def warmup(self) -> None:
        """Warmup the engine with a sample request."""
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
        """Add all requests and initialize their states."""
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
        """Update internal state tracking for a request."""
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

    def run_during_upgrade(self) -> dict:
        """Continue running with reduced capacity during upgrade."""
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
        pbar.close()
        progress = self.save_progress()
        return {"status": "READY_FOR_HANDOVER", "progress": progress}

    def _run_normal(self) -> str:
        """Original run logic until upgrade needed or completion."""
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
        
        # Check if we're in wait mode
        is_wait_mode = False
        if self._llm_engine.upgrade_config.drain_strategy == UpgradeStrategy.DrainStrategy.WAIT_THEN_KICKOUT:
            is_wait_mode = True
        # Track drain mode
        drain_mode_start_time = None

        while num_processed_requests < len(self._requests):
            elapsed_time = time.monotonic() - start_time
            
            if not self._is_new_runner and self._config.upgrade_time and elapsed_time > self._config.upgrade_time:
                # wait mode will not stop immediately, but send a signal to do draining
                if is_wait_mode:
                    # First time entering drain mode
                    if drain_mode_start_time is None:
                        logger.info(f"Replica {self._replica_id} reached upgrade time after {elapsed_time:.2f} seconds")
                        self._llm_engine.scheduler.set_drain()
                        drain_mode_start_time = time.monotonic()
                        logger.info(f"Replica {self._replica_id} entered drain mode - no new requests will be scheduled")
                    
                    # Check if we've exceeded the drain timeout
                    drain_elapsed = time.monotonic() - drain_mode_start_time
                    drain_timeout = getattr(self._llm_engine.upgrade_config, "drain_timeout", float('inf'))
                    
                    if drain_elapsed > drain_timeout or self._llm_engine.scheduler.has_enough_blocks(self._llm_engine.upgrade_config.required_blocks):
                        # Time to exit - prepare if needed
                        should_exit = True
                    else:
                        # Continue in drain mode
                        should_exit = False
                else:
                    # Not in wait mode, exit immediately
                    should_exit = True
                
                # If we need to exit, handle pipeline engine first if applicable
                if should_exit:
                    if is_pipeline_engine:
                        # Signal stop if not already in drain mode
                        if not is_wait_mode or drain_mode_start_time is None:
                            logger.info(f"Replica {self._replica_id} signaling pipeline to stop scheduling")
                        self._llm_engine.signal_stop_scheduling()
                        logger.info(f"Replica {self._replica_id} signaled for upgrade after {elapsed_time:.2f} seconds")
                        while self._llm_engine.has_inflight_batches():
                            time.sleep(0.01)  # Small sleep to prevent busy waiting
                    
                    # Exit message differs based on reason
                    logger.info(f"Replica {self._replica_id} stopping for upgrade after {elapsed_time:.2f} seconds")
                    if is_wait_mode:
                        if drain_elapsed > drain_timeout:
                            logger.info(f"Replica {self._replica_id} exceeded drain timeout of {drain_timeout:.2f} seconds, stopping for upgrade")
                        else:
                            logger.info(f"Replica {self._replica_id} has enough blocks before drain timeout, stopping for upgrade")
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
        """Save the current progress of requests."""
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
        """Load saved progress to resume requests."""
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

    def run(self) -> Any:
        """Main run method that handles different upgrade strategies."""
        self._llm_engine.reset_metrics()
        if not self._is_new_runner:
            self._add_requests()

        if self._config.upgrade_serving_strategy == UpgradeStrategy.ServingStrategy.DECODE_ONLY or self._config.upgrade_serving_strategy == UpgradeStrategy.ServingStrategy.PREFILL_ONLY:
            return self._run_with_overlap()
        elif self._config.upgrade_serving_strategy == UpgradeStrategy.ServingStrategy.NO_SERVE:
            return self._run_without_overlap()
        else:
            return self._run_normal()  # Just run normally for NO_UPGRADE

    def _run_with_overlap(self) -> dict:
        """Run with overlap serving during upgrade."""
        status = self._run_normal()
    
        if status == "UPGRADE_NEEDED":
            # Initialize upgrade state
            if self.upgrade_state is None:
                logger.error("Upgrade state not set. Cannot proceed with overlap serving.")
                return {"status": "ERROR"}

            self._llm_engine.prepare_for_upgrade()
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
        """Run without overlap serving during upgrade."""
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
        """Continue serving requests during overlap phase."""
        serving_result = self.run_during_upgrade()
        if serving_result["status"] == "READY_FOR_HANDOVER":
            return serving_result
        else:
            logger.error("Serving during upgrade failed")
            return {"status": "ERROR"}