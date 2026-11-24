import logging
import os
from pathlib import Path

from dotenv import dotenv_values
from pydantic import ValidationError
import pytest

from qwen3_eval.config import (
    ExperimentConfig,
    ExperimentPaths,
    SubdirsConfig,
    authenticate_huggingface,
    configure_hf_cache_env,
    load_experiment_config,
    load_model_spec,
)
from qwen3_eval.config.io_utils import load_yaml_mapping

pytestmark = pytest.mark.unit


def test_env_example_points_to_dedicated_qwen3_drive_layout(project_dir: Path) -> None:
    env_example = project_dir / ".env.example"
    assert env_example.is_file()
    vals = dotenv_values(env_example)
    assert vals["PROJECT_ROOT"] == "/content/drive/MyDrive/LLMReliabilityQwen3"
    assert vals["QWEN3_DIR"] == "/content/drive/MyDrive/LLMReliabilityQwen3/Qwen3"
    assert vals["MODELS_CACHE_DIR"] == "/content/models"
    assert vals["MODEL_CONFIG_PATH"] == "configs/model.yaml"
    assert vals["LOG_TIMEZONE"] == "Asia/Jakarta"
    assert vals["HF_TOKEN"] == ""


def test_config_precedence_os_over_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    custom_env = tmp_path / ".env"
    custom_env.write_text("LOG_LEVEL=DEBUG\nLOG_TIMEZONE=UTC\n", encoding="utf-8")

    cfg_env = load_experiment_config(env_file=custom_env, project_root=tmp_path / "qwen3_root")
    assert cfg_env.logging.level == "DEBUG"
    assert cfg_env.logging.timezone == "UTC"
    assert cfg_env.project_root == tmp_path / "qwen3_root"

    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    cfg_os = load_experiment_config(env_file=custom_env, project_root=tmp_path / "qwen3_root")
    assert cfg_os.logging.level == "WARNING"


def test_single_model_spec_loads_default_and_custom_yaml(tmp_path: Path) -> None:
    spec = load_model_spec()
    assert spec.key == "qwen3_32b"
    assert spec.display_name == "Qwen3 32B"
    assert spec.hf_id == "Qwen/Qwen3-32B"
    assert spec.estimated_vram_gb == 61.0
    assert spec.use_enable_thinking_flag is True
    assert spec.trust_remote_code is False
    assert not hasattr(spec, "requires_gated_access")

    custom_model = tmp_path / "custom_model.yaml"
    custom_model.write_text(
        "model:\n"
        "  key: qwen3_custom\n"
        "  display_name: Qwen3 Custom\n"
        "  hf_id: Qwen/Qwen3-14B\n"
        "  estimated_vram_gb: 28.0\n",
        encoding="utf-8",
    )
    custom_spec = load_model_spec(custom_model)
    assert custom_spec.key == "qwen3_custom"
    assert custom_spec.hf_id == "Qwen/Qwen3-14B"


@pytest.mark.parametrize(
    "unsafe_root",
    [
        Path("/content/drive/MyDrive/LLMReliability"),
        Path("/content/drive/MyDrive/LLMReliability/results"),
        Path("LLMReliability"),
        Path("/content/drive/MyDrive/LLMReliabilityEXP"),
        Path("/content/drive/MyDrive/LLMReliabilityEXP/results"),
        Path("LLMReliabilityEXP"),
        Path("/content/drive/MyDrive/LLMReliabilityQwen3/../LLMReliability"),
        Path("/content/drive/MyDrive/LLMReliabilityQwen3/../LLMReliabilityEXP/results"),
    ],
)
def test_experiment_paths_blocks_reserved_roots_and_traversal(
    qwen3_config: ExperimentConfig,
    unsafe_root: Path,
) -> None:
    with pytest.raises(ValueError, match="reserved folder"):
        ExperimentPaths.from_config(qwen3_config, project_root=unsafe_root)


@pytest.mark.parametrize("bad_subdir", ["../LLMReliability", "/tmp/escape", "   "])
def test_subdirs_config_rejects_traversal_and_absolute_paths(bad_subdir: str) -> None:
    with pytest.raises(ValidationError):
        SubdirsConfig(data_raw="data/raw", data_processed="data/processed", results=bad_subdir, logs="logs")


def test_load_yaml_mapping_rejects_empty_and_invalid_yaml(tmp_path: Path) -> None:
    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty YAML mapping"):
        load_yaml_mapping(empty_yaml)

    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty YAML mapping"):
        load_yaml_mapping(list_yaml)

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("key: [unclosed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid YAML syntax"):
        load_yaml_mapping(bad_yaml)


def test_authenticate_huggingface_handles_corrupt_secrets_safely(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    test_logger: logging.Logger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    corrupt_secrets = tmp_path / "secrets.json"
    corrupt_secrets.write_text("{not_valid_json", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger=test_logger.name):
        ok = authenticate_huggingface(secrets_path=str(corrupt_secrets), logger=test_logger)

    assert ok is False
    assert "hf_secrets_invalid" in caplog.text


def test_experiment_paths_creates_base_dirs_and_direct_results_dir(qwen3_paths: ExperimentPaths) -> None:
    assert qwen3_paths.data_raw_dir.is_dir()
    assert qwen3_paths.data_processed_dir.is_dir()
    assert qwen3_paths.results_dir.is_dir()
    assert qwen3_paths.results_dir == qwen3_paths.project_root / "results"
    assert qwen3_paths.logs_dir.is_dir()


def test_configure_hf_cache_env_sets_xet_guards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    monkeypatch.delenv("HF_XET_CACHE", raising=False)
    cache_dir = tmp_path / "models_cache"

    configure_hf_cache_env(cache_dir, disable_progress_bars=True)

    assert os.environ["HF_HOME"] == str(cache_dir)
    assert os.environ["HF_HUB_CACHE"] == str(cache_dir / "hub")
    assert os.environ["HF_DATASETS_CACHE"] == str(cache_dir / "datasets")
    assert os.environ["HF_XET_CACHE"] == str(cache_dir / "xet")
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"


def test_pipeline_script_validates_stage_and_max_tasks(project_dir: Path) -> None:
    import sys

    scripts_dir = str(project_dir / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from run_pipeline import execute_pipeline

    with pytest.raises(ValueError, match="Unsupported stage"):
        execute_pipeline(stage="invalid_stage")
    with pytest.raises(ValueError, match="max_tasks"):
        execute_pipeline(stage="infer", max_tasks=0)
