"""
Utilities for calculating block sizes and memory requirements for model upgrades.
"""

import math
import logging
import torch
from sarathi.config import ModelConfig, ParallelConfig

logger = logging.getLogger(__name__)

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
    """
    Get the size in bytes of a torch dtype.
    
    Args:
        dtype: PyTorch data type
        
    Returns:
        Size in bytes
    """
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
    PAGE_SIZE = 4 * 1024 * 1024  # 2MB in bytes
    
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
) -> tuple:
    """
    Calculate required blocks for model upgrade based on GPU memory and model parameters.
    
    Args:
        model_config: ModelConfig instance
        parallel_config: Current ParallelConfig
        new_parallel_config: New ParallelConfig for the upgrade
        attention_backend: Attention backend type
        
    Returns:
        Tuple of (required_blocks, pages_per_block)
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
    safety_margin = 1.1
    required_memory = new_model_memory_per_gpu * safety_margin
    
    # Calculate number of blocks needed
    required_blocks = math.ceil(required_memory / current_block_size)
    PAGE_SIZE = 4 * 1024 * 1024
    pages_per_block = current_block_size / PAGE_SIZE
    logger.info("Memory Calculation Details:")
    logger.info(f"New model weights per GPU: {new_params_per_gpu:,} parameters")
    logger.info(f"Memory per parameter: {bytes_per_param} bytes")
    logger.info(f"Raw model memory needed: {new_model_memory_per_gpu / (1024*1024):,.2f} MB")
    logger.info(f"With {safety_margin}x safety margin: {required_memory / (1024*1024):,.2f} MB")
    logger.info("\nBlock Size Details:")
    logger.info(f"Current block size: {current_block_size / (1024*1024):,.2f} MB")
    logger.info(f"Required blocks for upgrade: {required_blocks:,}")
    logger.info(f"Pages per block: {pages_per_block:,}")
    
    return required_blocks, pages_per_block