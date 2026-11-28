from typing import Any, Dict

from qwen3_eval.models.runtime_utils import (
    configure_deterministic_generation,
    resolve_input_device,
    resolve_pad_token_id,
    warn_if_offloaded,
)
from qwen3_eval.utils.lazy import resolve_lazy_export

_LAZY_EXPORTS: Dict[str, str] = {
    "load_model_and_tokenizer": "qwen3_eval.models.loader",
    "load_tokenizer_only": "qwen3_eval.models.loader",
}


def __getattr__(name: str) -> Any:
    return resolve_lazy_export(__name__, name, _LAZY_EXPORTS)


__all__ = [
    "configure_deterministic_generation",
    "load_model_and_tokenizer",
    "load_tokenizer_only",
    "resolve_input_device",
    "resolve_pad_token_id",
    "warn_if_offloaded",
]
