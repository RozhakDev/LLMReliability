from typing import Any

from qwen3_eval.inference.tasks import (
    build_evaluation_tasks,
    load_processed_combinations,
)

_RUNNER_EXPORTS = frozenset({"run_inference"})


def __getattr__(name: str) -> Any:
    if name in _RUNNER_EXPORTS:
        from qwen3_eval.inference import runner

        return getattr(runner, name)
    raise AttributeError(f"module 'qwen3_eval.inference' has no attribute {name!r}")


__all__ = [
    "build_evaluation_tasks",
    "load_processed_combinations",
    "run_inference",
]
