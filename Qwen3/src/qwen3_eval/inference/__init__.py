from typing import Any, Dict

from qwen3_eval.inference.tasks import (
    build_evaluation_tasks,
    load_processed_combinations,
)
from qwen3_eval.utils.lazy import resolve_lazy_export

_LAZY_EXPORTS: Dict[str, str] = {
    "run_inference": "qwen3_eval.inference.runner",
}


def __getattr__(name: str) -> Any:
    return resolve_lazy_export(__name__, name, _LAZY_EXPORTS)


__all__ = [
    "build_evaluation_tasks",
    "load_processed_combinations",
    "run_inference",
]
