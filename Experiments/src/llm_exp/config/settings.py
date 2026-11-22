from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional, Union

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml

from llm_exp.config.schemas import (
    ArtifactsConfig,
    DatasetConfig,
    EvaluationConfig,
    ExperimentYAMLConfig,
    InferenceConfig,
    LoggingConfig,
    VisualizationConfig,
    WorkspaceConfig,
)


def get_package_experiments_dir() -> Path:
    return Path(__file__).resolve().parents[3]


class RuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file_encoding="utf-8", extra="ignore")

    drive_mount_point: str = Field(default="")
    project_root: str = Field(default="")
    experiments_dir: str = Field(default="")
    models_cache_dir: str = Field(default="")
    experiment_config_path: str = Field(default="configs/experiment.yaml")
    models_config_path: str = Field(default="configs/models.yaml")
    prompts_config_path: str = Field(default="configs/prompts.yaml")
    hf_secrets_file: str = Field(default="")
    hf_token: str = Field(default="")
    log_level: str = Field(default="")
    log_timezone: str = Field(default="")
    hf_hub_disable_progress_bars: bool = Field(default=True)


@dataclass(frozen=True)
class ExperimentConfig:
    env: RuntimeSettings
    experiments_dir: Path
    project_root: Path
    models_cache_dir: Path
    experiment_config_path: Path
    models_config_path: Path
    prompts_config_path: Path
    workspace: WorkspaceConfig
    dataset: DatasetConfig
    evaluation: EvaluationConfig
    inference: InferenceConfig
    logging: LoggingConfig
    visualization: VisualizationConfig
    artifacts: ArtifactsConfig


def _resolve_path(raw: str, base_dir: Path) -> Path:
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else (base_dir / candidate)


def _find_env_file(base_dir: Path, explicit_env: Optional[Union[str, Path]]) -> Optional[Path]:
    if explicit_env is not None:
        env_path = Path(explicit_env)
        if not env_path.is_file():
            raise FileNotFoundError(f"Specified env file not found: {env_path}")
        return env_path
    for name in (".env", ".env.example"):
        candidate = base_dir / name
        if candidate.is_file():
            return candidate
    return None


def _resolve_workspace_roots(
    env_cfg: RuntimeSettings,
    yaml_cfg: ExperimentYAMLConfig,
    project_root: Optional[Union[str, Path]],
    models_cache_dir: Optional[Union[str, Path]],
) -> tuple[Path, Path]:
    drive_mount = env_cfg.drive_mount_point.strip()
    drive_mounted = bool(drive_mount) and (os.path.ismount(drive_mount) or os.path.isdir(drive_mount))

    if project_root is not None:
        resolved_root = Path(project_root)
    elif os.environ.get("PROJECT_ROOT", "").strip():
        resolved_root = Path(os.environ["PROJECT_ROOT"].strip())
    elif env_cfg.project_root.strip():
        uses_drive = bool(drive_mount) and env_cfg.project_root.strip().startswith(drive_mount)
        resolved_root = (
            Path(env_cfg.project_root.strip())
            if (drive_mounted or not uses_drive)
            else Path(yaml_cfg.workspace.default_local_root)
        )
    else:
        resolved_root = Path(yaml_cfg.workspace.default_local_root)

    if models_cache_dir is not None:
        resolved_cache = Path(models_cache_dir)
    elif os.environ.get("MODELS_CACHE_DIR", "").strip():
        resolved_cache = Path(os.environ["MODELS_CACHE_DIR"].strip())
    elif env_cfg.models_cache_dir.strip() and drive_mounted:
        resolved_cache = Path(env_cfg.models_cache_dir.strip())
    else:
        resolved_cache = resolved_root / ".hf_cache"

    return resolved_root, resolved_cache


def load_experiment_config(
    env_file: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
    models_config_path: Optional[Union[str, Path]] = None,
    prompts_config_path: Optional[Union[str, Path]] = None,
    project_root: Optional[Union[str, Path]] = None,
    models_cache_dir: Optional[Union[str, Path]] = None,
    experiments_dir: Optional[Union[str, Path]] = None,
) -> ExperimentConfig:
    pkg_dir = get_package_experiments_dir()
    os_exp = os.environ.get("EXPERIMENTS_DIR", "").strip()
    exp_dir = (
        Path(experiments_dir)
        if experiments_dir is not None
        else (Path(os_exp) if (os_exp and Path(os_exp).is_dir()) else pkg_dir)
    )

    resolved_env = _find_env_file(exp_dir, env_file)
    env_cfg = RuntimeSettings(_env_file=resolved_env) if resolved_env else RuntimeSettings()

    exp_yaml = Path(config_path) if config_path else _resolve_path(env_cfg.experiment_config_path, exp_dir)
    mod_yaml = Path(models_config_path) if models_config_path else _resolve_path(env_cfg.models_config_path, exp_dir)
    prm_yaml = Path(prompts_config_path) if prompts_config_path else _resolve_path(env_cfg.prompts_config_path, exp_dir)

    if not exp_yaml.is_file():
        raise FileNotFoundError(f"Experiment YAML config not found: {exp_yaml}")
    yaml_cfg = ExperimentYAMLConfig.model_validate(yaml.safe_load(exp_yaml.read_text(encoding="utf-8")))

    resolved_root, resolved_cache = _resolve_workspace_roots(env_cfg, yaml_cfg, project_root, models_cache_dir)
    log_cfg = yaml_cfg.logging.model_copy(
        update={
            "level": (env_cfg.log_level or yaml_cfg.logging.level).upper(),
            "timezone": env_cfg.log_timezone or yaml_cfg.logging.timezone,
        }
    )

    return ExperimentConfig(
        env=env_cfg,
        experiments_dir=exp_dir,
        project_root=resolved_root,
        models_cache_dir=resolved_cache,
        experiment_config_path=exp_yaml,
        models_config_path=mod_yaml,
        prompts_config_path=prm_yaml,
        workspace=yaml_cfg.workspace,
        dataset=yaml_cfg.dataset,
        evaluation=yaml_cfg.evaluation,
        inference=yaml_cfg.inference,
        logging=log_cfg,
        visualization=yaml_cfg.visualization,
        artifacts=yaml_cfg.artifacts,
    )
