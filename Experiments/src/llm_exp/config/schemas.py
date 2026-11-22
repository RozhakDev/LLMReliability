from typing import Dict, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SubdirsConfig(_StrictModel):
    data_raw: str
    data_processed: str
    results: str
    comparison: str
    logs: str


class WorkspaceConfig(_StrictModel):
    default_local_root: str
    reserved_folder_names: Tuple[str, ...]
    protected_drive_parents: Tuple[str, ...]
    subdirs: SubdirsConfig


class DatasetConfig(_StrictModel):
    hf_id: str
    split: str
    raw_filename: str
    eval_filename: str
    required_columns: Tuple[str, ...]
    text_columns: Tuple[str, ...]
    min_silver_records: int
    min_raw_records: int
    sample_size: Optional[int] = None


class EvaluationConfig(_StrictModel):
    random_seed: int
    prompts: Tuple[str, ...]
    positions: Tuple[str, ...]
    bootstrap_iterations: int
    alpha: float


class InferenceConfig(_StrictModel):
    max_new_tokens: int
    log_interval: int
    gc_interval: int
    do_sample: bool
    cleanup_cache_after: bool
    min_vram_gb: float
    min_disk_gb: float


class LoggingConfig(_StrictModel):
    level: str
    timezone: str
    date_subfolder: bool
    timestamp_format: str
    filename_pattern: str
    error_filename_pattern: str
    separate_error_log: bool


class VisualizationConfig(_StrictModel):
    dpi: int
    style: str
    font_scale: float
    font_family: str


class ArtifactsConfig(_StrictModel):
    output_jsonl: str
    metrics_summary_csv: str
    metrics_summary_parquet: str
    detailed_results_parquet: str
    domain_accuracy_csv: str
    prompt_stability_csv: str
    executive_summary_csv: str
    comparison_summary_csv: str
    comparison_prompt_breakdown_csv: str
    comparison_prompt_stability_csv: str
    comparison_domain_accuracy_csv: str


class ExperimentYAMLConfig(_StrictModel):
    workspace: WorkspaceConfig
    dataset: DatasetConfig
    evaluation: EvaluationConfig
    inference: InferenceConfig
    logging: LoggingConfig
    visualization: VisualizationConfig
    artifacts: ArtifactsConfig


class ModelSpec(_StrictModel):
    key: str
    display_name: str
    hf_id: str
    loader_type: str
    estimated_vram_gb: float
    is_primary: bool = False
    is_multimodal_processor: bool = False
    use_enable_thinking_flag: bool = False
    trust_remote_code: bool = False
    requires_gated_access: bool = False
    primary_strength: str = ""


class ModelsCatalog(_StrictModel):
    models: Tuple[ModelSpec, ...] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _ensure_unique_keys(self) -> "ModelsCatalog":
        keys = [m.key for m in self.models]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate model keys found in models.yaml.")
        return self


class PromptsCatalog(_StrictModel):
    system_prompt: str
    templates: Dict[str, str]
