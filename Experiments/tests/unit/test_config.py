import os
from pathlib import Path

from dotenv import dotenv_values
import pytest

from llm_exp.config import (
    ExperimentConfig,
    ExperimentPaths,
    configure_hf_cache_env,
    get_model_spec,
    list_model_keys,
    load_experiment_config,
    load_model_registry,
)

pytestmark = pytest.mark.unit


def test_env_example_points_to_final_drive_layout(exp_root: Path) -> None:
    env_example = exp_root / ".env.example"
    assert env_example.is_file()
    vals = dotenv_values(env_example)
    assert vals["PROJECT_ROOT"] == "/content/drive/MyDrive/LLMReliabilityEXP"
    assert vals["EXPERIMENTS_DIR"] == "/content/drive/MyDrive/LLMReliabilityEXP/Experiments"
    assert vals["MODELS_CACHE_DIR"] == "/content/models"
    assert vals["LOG_TIMEZONE"] == "Asia/Jakarta"
    assert vals["HF_TOKEN"] == ""


def test_config_precedence_os_over_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    custom_env = tmp_path / ".env"
    custom_env.write_text("LOG_LEVEL=DEBUG\nLOG_TIMEZONE=UTC\n", encoding="utf-8")

    cfg_env = load_experiment_config(env_file=custom_env, project_root=tmp_path / "exp_root")
    assert cfg_env.logging.level == "DEBUG"
    assert cfg_env.logging.timezone == "UTC"
    assert cfg_env.project_root == tmp_path / "exp_root"

    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    cfg_os = load_experiment_config(env_file=custom_env, project_root=tmp_path / "exp_root")
    assert cfg_os.logging.level == "WARNING"


def test_model_registry_loads_default_and_custom_yaml(tmp_path: Path) -> None:
    assert list_model_keys() == [
        "qwen3_32b",
        "gemma3_27b_goto",
        "aya_expanse_32b",
        "mistral_small_3_1_24b",
        "glm_4_32b",
    ]
    registry = load_model_registry()
    assert get_model_spec("qwen3_32b").is_primary is True
    assert registry["aya_expanse_32b"].hf_id == "CohereLabs/aya-expanse-32b"
    assert registry["glm_4_32b"].hf_id == "zai-org/GLM-4-32B-0414"
    assert registry["mistral_small_3_1_24b"].loader_type == "mistral3"
    assert registry["gemma3_27b_goto"].loader_type == "gemma3"

    with pytest.raises(KeyError, match="Unknown model key"):
        get_model_spec("nonexistent_model")

    custom_models = tmp_path / "custom_models.yaml"
    custom_models.write_text(
        "models:\n"
        "  - key: demo_model\n"
        "    display_name: Demo 7B\n"
        "    hf_id: org/demo-7b\n"
        "    loader_type: causal_lm\n"
        "    estimated_vram_gb: 14.0\n",
        encoding="utf-8",
    )
    assert list_model_keys(custom_models) == ["demo_model"]
    assert load_model_registry(custom_models)["demo_model"].hf_id == "org/demo-7b"


@pytest.mark.parametrize(
    "unsafe_root",
    [
        Path("/content/drive/MyDrive/LLMReliability"),
        Path("/content/drive/MyDrive/LLMReliability/results"),
        Path("LLMReliability"),
    ],
)
def test_experiment_paths_blocks_reserved_roots(exp_config: ExperimentConfig, unsafe_root: Path) -> None:
    with pytest.raises(ValueError, match="LLMReliability"):
        ExperimentPaths.from_config(exp_config, project_root=unsafe_root)


def test_experiment_paths_creates_base_dirs(exp_paths: ExperimentPaths) -> None:
    assert exp_paths.data_raw_dir.is_dir()
    assert exp_paths.data_processed_dir.is_dir()
    assert exp_paths.results_root_dir.is_dir()
    assert exp_paths.comparison_dir.is_dir()
    assert exp_paths.logs_dir.is_dir()


def test_configure_hf_cache_env_sets_xet_guards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    monkeypatch.delenv("HF_XET_CACHE", raising=False)
    cache_dir = tmp_path / "models_cache"

    configure_hf_cache_env(cache_dir, disable_progress_bars=True)

    assert os.environ["HF_HOME"] == str(cache_dir)
    assert os.environ["HF_HUB_CACHE"] == str(cache_dir / "hub")
    assert os.environ["HF_XET_CACHE"] == str(cache_dir / "xet")
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"
