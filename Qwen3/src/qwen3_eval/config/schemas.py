from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _validate_safe_relative_path(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"Field '{field_name}' must not be empty.")
    candidate = Path(cleaned)
    if candidate.is_absolute() or bool(candidate.anchor) or cleaned.startswith(("/", "\\")) or ".." in candidate.parts:
        raise ValueError(f"Field '{field_name}' must be a relative path without '..': '{value}'.")
    return cleaned


class _StrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SubdirsConfig(_StrictModel):
    data_raw: str
    data_processed: str
    results: str
    logs: str

    @field_validator("data_raw", "data_processed", "results", "logs")
    @classmethod
    def _check_safe_subdir(cls, value: str, info: Any) -> str:
        return _validate_safe_relative_path(value, str(info.field_name))


class WorkspaceConfig(_StrictModel):
    default_local_root: str = Field(min_length=1)
    reserved_folder_names: Tuple[str, ...]
    protected_drive_parents: Tuple[str, ...]
    subdirs: SubdirsConfig


class DatasetConfig(_StrictModel):
    hf_id: str = Field(min_length=1)
    split: str = Field(min_length=1)
    raw_filename: str
    eval_filename: str
    required_columns: Tuple[str, ...] = Field(min_length=1)
    text_columns: Tuple[str, ...] = Field(min_length=1)
    min_silver_records: int = Field(ge=1)
    min_raw_records: int = Field(ge=1)
    sample_size: Optional[int] = Field(default=None, ge=1)

    @field_validator("raw_filename", "eval_filename")
    @classmethod
    def _check_safe_filename(cls, value: str, info: Any) -> str:
        return _validate_safe_relative_path(value, str(info.field_name))


class EvaluationConfig(_StrictModel):
    random_seed: int = Field(ge=0)
    prompts: Tuple[str, ...] = Field(min_length=1)
    positions: Tuple[str, ...] = Field(min_length=1)
    bootstrap_iterations: int = Field(ge=1)
    alpha: float = Field(gt=0.0, lt=1.0)


class InferenceConfig(_StrictModel):
    max_new_tokens: int = Field(ge=1)
    log_interval: int = Field(ge=1)
    gc_interval: int = Field(ge=1)
    do_sample: bool
    cleanup_cache_after: bool
    min_vram_gb: float = Field(ge=0.0)
    min_disk_gb: float = Field(ge=0.0)


class LoggingConfig(_StrictModel):
    level: str = Field(min_length=1)
    timezone: str
    date_subfolder: bool
    timestamp_format: str = Field(min_length=1)
    filename_pattern: str = Field(min_length=1)
    error_filename_pattern: str = Field(min_length=1)
    separate_error_log: bool


class VisualizationConfig(_StrictModel):
    dpi: int = Field(ge=50)
    style: str = Field(min_length=1)
    font_scale: float = Field(gt=0.0)
    font_family: str = Field(min_length=1)


class ArtifactsConfig(_StrictModel):
    output_jsonl: str
    metrics_summary_csv: str
    metrics_summary_parquet: str
    detailed_results_parquet: str
    domain_accuracy_csv: str
    prompt_stability_csv: str
    executive_summary_csv: str

    @field_validator(
        "output_jsonl",
        "metrics_summary_csv",
        "metrics_summary_parquet",
        "detailed_results_parquet",
        "domain_accuracy_csv",
        "prompt_stability_csv",
        "executive_summary_csv",
    )
    @classmethod
    def _check_safe_artifact_name(cls, value: str, info: Any) -> str:
        return _validate_safe_relative_path(value, str(info.field_name))


class ExperimentYAMLConfig(_StrictModel):
    workspace: WorkspaceConfig
    dataset: DatasetConfig
    evaluation: EvaluationConfig
    inference: InferenceConfig
    logging: LoggingConfig
    visualization: VisualizationConfig
    artifacts: ArtifactsConfig


class ModelSpec(_StrictModel):
    key: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    hf_id: str = Field(min_length=1)
    estimated_vram_gb: float = Field(gt=0.0)
    use_enable_thinking_flag: bool = True
    trust_remote_code: bool = False


class SingleModelCatalog(_StrictModel):
    model: ModelSpec


class PromptsCatalog(_StrictModel):
    system_prompt: str = Field(min_length=1)
    templates: Dict[str, str] = Field(min_length=1)
