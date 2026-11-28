from typing import Any, Dict

from qwen3_eval.utils.cache import cleanup_hf_model_cache, purge_other_hf_caches
from qwen3_eval.utils.lazy import resolve_lazy_export
from qwen3_eval.utils.logger import close_logger_handlers, setup_logger

_LAZY_EXPORTS: Dict[str, str] = {
    "clear_gpu_memory": "qwen3_eval.utils.gpu",
    "log_hardware_status": "qwen3_eval.utils.gpu",
    "resolve_torch_dtype": "qwen3_eval.utils.gpu",
}


def __getattr__(name: str) -> Any:
    return resolve_lazy_export(__name__, name, _LAZY_EXPORTS)


__all__ = [
    "cleanup_hf_model_cache",
    "clear_gpu_memory",
    "close_logger_handlers",
    "log_hardware_status",
    "purge_other_hf_caches",
    "resolve_torch_dtype",
    "setup_logger",
]
