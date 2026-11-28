from typing import Any, Dict

from qwen3_eval.data.labels import (
    flip_candidate_label,
    map_positional_to_logical,
    order_candidates,
    resolve_human_winner,
)
from qwen3_eval.data.profiling import (
    build_domain_distribution,
    build_preprocessing_summary,
    build_sample_detail_table,
    build_sample_overview_table,
    compute_domain_stats,
)
from qwen3_eval.data.schema import validate_dataset_schema
from qwen3_eval.utils.lazy import resolve_lazy_export

_LAZY_EXPORTS: Dict[str, str] = {
    "prepare_evaluation_dataset": "qwen3_eval.data.indopref_loader",
}


def __getattr__(name: str) -> Any:
    return resolve_lazy_export(__name__, name, _LAZY_EXPORTS)


__all__ = [
    "build_domain_distribution",
    "build_preprocessing_summary",
    "build_sample_detail_table",
    "build_sample_overview_table",
    "compute_domain_stats",
    "flip_candidate_label",
    "map_positional_to_logical",
    "order_candidates",
    "prepare_evaluation_dataset",
    "resolve_human_winner",
    "validate_dataset_schema",
]
