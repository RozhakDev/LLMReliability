from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

from llm_exp.config.schemas import ModelsCatalog, ModelSpec
from llm_exp.config.settings import load_experiment_config


@lru_cache(maxsize=8)
def _load_catalog(resolved_path: str) -> Dict[str, ModelSpec]:
    yaml_path = Path(resolved_path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Model registry YAML not found: {yaml_path}")
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    catalog = ModelsCatalog.model_validate(raw)
    return {spec.key: spec for spec in catalog.models}


def load_model_registry(
    models_config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, ModelSpec]:
    target = (
        Path(models_config_path)
        if models_config_path is not None
        else load_experiment_config().models_config_path
    )
    return dict(_load_catalog(str(target.resolve())))


def get_model_spec(
    model_key: str,
    models_config_path: Optional[Union[str, Path]] = None,
) -> ModelSpec:
    registry = load_model_registry(models_config_path=models_config_path)
    if model_key not in registry:
        raise KeyError(f"Unknown model key '{model_key}'. Valid options: {', '.join(registry)}")
    return registry[model_key]


def list_model_keys(
    models_config_path: Optional[Union[str, Path]] = None,
) -> List[str]:
    return list(load_model_registry(models_config_path=models_config_path).keys())
