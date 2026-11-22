from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from llm_exp.config.settings import ExperimentConfig


@dataclass(frozen=True)
class ExperimentPaths:
    config: ExperimentConfig
    project_root: Path
    models_cache_dir: Path

    def __post_init__(self) -> None:
        reserved = {name.lower() for name in self.config.workspace.reserved_folder_names}
        parents = {name.lower() for name in self.config.workspace.protected_drive_parents}
        parts = [part.lower() for part in self.project_root.parts]

        exact_hit = self.project_root.name.lower() in reserved
        nested_hit = any(parts[i] in parents and parts[i + 1] in reserved for i in range(len(parts) - 1))
        if exact_hit or nested_hit:
            blocked = ", ".join(self.config.workspace.reserved_folder_names)
            raise ValueError(f"Unsafe project_root '{self.project_root}': reserved folder ({blocked}) is protected.")

    @classmethod
    def from_config(
        cls,
        config: ExperimentConfig,
        project_root: Optional[Union[str, Path]] = None,
        models_cache_dir: Optional[Union[str, Path]] = None,
    ) -> "ExperimentPaths":
        return cls(
            config=config,
            project_root=Path(project_root) if project_root is not None else config.project_root,
            models_cache_dir=Path(models_cache_dir) if models_cache_dir is not None else config.models_cache_dir,
        )

    @property
    def data_raw_dir(self) -> Path:
        return self.project_root / self.config.workspace.subdirs.data_raw

    @property
    def data_processed_dir(self) -> Path:
        return self.project_root / self.config.workspace.subdirs.data_processed

    @property
    def raw_parquet_path(self) -> Path:
        return self.data_raw_dir / self.config.dataset.raw_filename

    @property
    def eval_parquet_path(self) -> Path:
        return self.data_processed_dir / self.config.dataset.eval_filename

    @property
    def results_root_dir(self) -> Path:
        return self.project_root / self.config.workspace.subdirs.results

    @property
    def comparison_dir(self) -> Path:
        return self.project_root / self.config.workspace.subdirs.comparison

    @property
    def logs_dir(self) -> Path:
        return self.project_root / self.config.workspace.subdirs.logs

    def model_results_dir(self, model_slug: str) -> Path:
        return self.results_root_dir / model_slug

    def ensure_base_dirs(self) -> None:
        for directory in (
            self.data_raw_dir,
            self.data_processed_dir,
            self.models_cache_dir,
            self.results_root_dir,
            self.comparison_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
