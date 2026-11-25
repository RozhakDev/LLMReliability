from typing import Any

from qwen3_eval.utils.cache import cleanup_hf_model_cache, purge_other_hf_caches
from qwen3_eval.utils.logger import close_logger_handlers, setup_logger

_GPU_EXPORTS = frozenset(
    {
        "clear_gpu_memory",
        "log_hardware_status",
        "resolve_torch_dtype",
    }
)


def __getattr__(name: str) -> Any:
    if name in _GPU_EXPORTS:
        from qwen3_eval.utils import gpu as gpu_mod

        return getattr(gpu_mod, name)
    raise AttributeError(f"module 'qwen3_eval.utils' has no attribute {name!r}")


__all__ = [
    "cleanup_hf_model_cache",
    "clear_gpu_memory",
    "close_logger_handlers",
    "log_hardware_status",
    "purge_other_hf_caches",
    "resolve_torch_dtype",
    "setup_logger",
]
