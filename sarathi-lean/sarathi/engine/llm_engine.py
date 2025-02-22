from sarathi.engine.arg_utils import EngineArgs
from sarathi.engine.base_llm_engine import BaseLLMEngine
from sarathi.engine.pipeline_parallel_llm_engine import PipelineParallelLLMEngine


class LLMEngine:

    @classmethod
    def from_engine_args(cls, **kwargs) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_configs = EngineArgs(**kwargs).create_engine_configs()
        parallel_config = engine_configs[2]
        if parallel_config.pipeline_parallel_size > 1:
            engine = PipelineParallelLLMEngine(*engine_configs)
        else:
            engine = BaseLLMEngine(*engine_configs)

        return engine
    
    @classmethod
    def get_engine_configs(cls, **kwargs):
        """
        Returns engine configurations without creating an engine.
        
        Args:
            **kwargs: Engine arguments to be passed to EngineArgs
            
        Returns:
            tuple: Engine configuration tuple from EngineArgs.create_engine_configs()
        """
        return EngineArgs(**kwargs).create_engine_configs()
