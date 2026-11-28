import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Union

try:
    from google.colab import drive as colab_drive
    from google.colab import userdata as colab_userdata
except ImportError:
    colab_drive = None
    colab_userdata = None

from llm_exp.config.paths import ExperimentPaths
from llm_exp.config.settings import ExperimentConfig, load_experiment_config


def mount_google_drive(mount_point: str, logger: Optional[logging.Logger] = None) -> bool:
    if colab_drive is None or not mount_point:
        return False
    if not (os.path.ismount(mount_point) or os.path.isdir(os.path.join(mount_point, "MyDrive"))):
        colab_drive.mount(mount_point)
    if logger:
        logger.info("drive_mounted | mount=%s", mount_point)
    return True


def authenticate_huggingface(
    secrets_path: str = "",
    explicit_token: str = "",
    logger: Optional[logging.Logger] = None,
) -> bool:
    token = explicit_token.strip()
    if not token and secrets_path and os.path.exists(secrets_path):
        token = str(json.loads(Path(secrets_path).read_text(encoding="utf-8")).get("HF_TOKEN", "")).strip()
    if not token and colab_userdata is not None:
        try:
            token = str(colab_userdata.get("HF_TOKEN") or "").strip()
        except Exception:
            pass
    if not token:
        token = os.environ.get("HF_TOKEN", "").strip()

    if not token:
        if logger:
            logger.warning("hf_token_missing | impact=gated_models_unavailable")
        return False

    try:
        from huggingface_hub import login as hf_login
    except ImportError:
        return False

    hf_login(token=token, add_to_git_credential=False)
    if logger:
        logger.info("hf_auth_ok")
    return True


def configure_hf_cache_env(models_cache_dir: Path, disable_progress_bars: bool = True) -> None:
    os.environ["HF_HOME"] = str(models_cache_dir)
    os.environ["HF_HUB_CACHE"] = str(models_cache_dir / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(models_cache_dir / "datasets")
    os.environ["HF_XET_CACHE"] = str(models_cache_dir / "xet")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if disable_progress_bars:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


def bootstrap_environment(
    config: Optional[ExperimentConfig] = None,
    env_file: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
    project_root: Optional[Union[str, Path]] = None,
    models_cache_dir: Optional[Union[str, Path]] = None,
    authenticate_hf: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Tuple[ExperimentConfig, ExperimentPaths]:
    cfg = config or load_experiment_config(
        env_file=env_file,
        config_path=config_path,
        project_root=project_root,
        models_cache_dir=models_cache_dir,
    )
    if mount_google_drive(cfg.env.drive_mount_point, logger=logger) and config is None:
        cfg = load_experiment_config(
            env_file=env_file,
            config_path=config_path,
            project_root=project_root,
            models_cache_dir=models_cache_dir,
        )

    paths = ExperimentPaths.from_config(cfg, project_root=project_root, models_cache_dir=models_cache_dir)
    paths.ensure_base_dirs()
    configure_hf_cache_env(paths.models_cache_dir, cfg.env.hf_hub_disable_progress_bars)
    if authenticate_hf:
        authenticate_huggingface(cfg.env.hf_secrets_file, cfg.env.hf_token, logger=logger)

    if logger:
        logger.info("workspace_ready | root=%s | cache=%s", paths.project_root, paths.models_cache_dir)
    return cfg, paths
