from typing import Any

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
)
from qwen3_eval.data.schema import validate_dataset_schema

_LOADER_EXPORTS = frozenset({"prepare_evaluation_dataset"})


def __getattr__(name: str) -> Any:
    if name in _LOADER_EXPORTS:
        from qwen3_eval.data import indopref_loader

        return getattr(indopref_loader, name)
    raise AttributeError(f"module 'qwen3_eval.data' has no attribute {name!r}")


__all__ = [
    "build_domain_distribution",
    "build_preprocessing_summary",
    "build_sample_detail_table",
    "build_sample_overview_table",
    "flip_candidate_label",
    "map_positional_to_logical",
    "order_candidates",
    "prepare_evaluation_dataset",
    "resolve_human_winner",
    "validate_dataset_schema",
]
