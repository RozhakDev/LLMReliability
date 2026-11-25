from typing import Any

from qwen3_eval.metrics.artifacts import EvaluationArtifacts
from qwen3_eval.metrics.contingency import (
    PositionalContingency,
    build_paired_positional_frame,
    summarize_positional_contingency,
)
from qwen3_eval.metrics.inspection import (
    build_parsed_predictions_preview,
    build_positional_bias_diagnostics,
)
from qwen3_eval.metrics.parser import load_and_parse_results, parse_decision

_RELIABILITY_EXPORTS = frozenset({"evaluate_model"})


def __getattr__(name: str) -> Any:
    if name in _RELIABILITY_EXPORTS:
        from qwen3_eval.metrics import reliability

        return getattr(reliability, name)
    raise AttributeError(f"module 'qwen3_eval.metrics' has no attribute {name!r}")


__all__ = [
    "EvaluationArtifacts",
    "PositionalContingency",
    "build_paired_positional_frame",
    "build_parsed_predictions_preview",
    "build_positional_bias_diagnostics",
    "evaluate_model",
    "load_and_parse_results",
    "parse_decision",
    "summarize_positional_contingency",
]
