from typing import Any

from qwen3_eval.models.runtime_utils import (
    configure_deterministic_generation,
    resolve_input_device,
    resolve_pad_token_id,
    warn_if_offloaded,
)

_LOADER_EXPORTS = frozenset({"load_model_and_tokenizer", "load_tokenizer_only"})


def __getattr__(name: str) -> Any:
    if name in _LOADER_EXPORTS:
        from qwen3_eval.models import loader

        return getattr(loader, name)
    raise AttributeError(f"module 'qwen3_eval.models' has no attribute {name!r}")


__all__ = [
    "configure_deterministic_generation",
    "load_model_and_tokenizer",
    "load_tokenizer_only",
    "resolve_input_device",
    "resolve_pad_token_id",
    "warn_if_offloaded",
]
