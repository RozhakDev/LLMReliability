from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

from qwen3_eval.config.io_utils import load_yaml_mapping
from qwen3_eval.config.schemas import ModelSpec, SingleModelCatalog
from qwen3_eval.config.settings import load_experiment_config


@lru_cache(maxsize=8)
def _load_single_model_cached(resolved_path: str) -> ModelSpec:
    raw = load_yaml_mapping(Path(resolved_path), label="Model YAML config")
    return SingleModelCatalog.model_validate(raw).model


def load_model_spec(
    model_config_path: Optional[Union[str, Path]] = None,
) -> ModelSpec:
    target = (
        Path(model_config_path)
        if model_config_path is not None
        else load_experiment_config().model_config_path
    )
    return _load_single_model_cached(str(target.resolve()))
