from typing import Any, Dict

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
from qwen3_eval.utils.lazy import resolve_lazy_export

_LAZY_EXPORTS: Dict[str, str] = {
    "evaluate_model": "qwen3_eval.metrics.reliability",
}


def __getattr__(name: str) -> Any:
    return resolve_lazy_export(__name__, name, _LAZY_EXPORTS)


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
