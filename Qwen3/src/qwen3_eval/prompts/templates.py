from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

from qwen3_eval.config.io_utils import load_yaml_mapping
from qwen3_eval.config.schemas import PromptsCatalog
from qwen3_eval.config.settings import load_experiment_config


@lru_cache(maxsize=8)
def _load_prompts_cached(resolved_path: str) -> PromptsCatalog:
    raw = load_yaml_mapping(Path(resolved_path), label="Prompts YAML config")
    return PromptsCatalog.model_validate(raw)


def load_prompt_catalog(
    prompts_config_path: Optional[Union[str, Path]] = None,
) -> PromptsCatalog:
    target = (
        Path(prompts_config_path)
        if prompts_config_path is not None
        else load_experiment_config().prompts_config_path
    )
    return _load_prompts_cached(str(target.resolve()))


def get_system_prompt(prompts_config_path: Optional[Union[str, Path]] = None) -> str:
    return load_prompt_catalog(prompts_config_path=prompts_config_path).system_prompt


def format_prompt(
    input_text: str,
    resp_a: str,
    resp_b: str,
    prompt_id: str,
    prompts_config_path: Optional[Union[str, Path]] = None,
) -> str:
    catalog = load_prompt_catalog(prompts_config_path=prompts_config_path)
    if prompt_id not in catalog.templates:
        valid = ", ".join(catalog.templates.keys())
        raise ValueError(f"Unsupported prompt_id '{prompt_id}'. Expected one of: {valid}.")
    return catalog.templates[prompt_id].format(
        input_text=input_text,
        resp_a=resp_a,
        resp_b=resp_b,
    )
