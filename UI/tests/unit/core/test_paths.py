"""Unit test untuk resolusi path direktori proyek."""
from pathlib import Path
from llm_reliability.core.paths import PROJECT_ROOT, DATA_DIR, MODELS_DIR


def test_project_root_is_ui_directory():
    assert PROJECT_ROOT.name == "UI"
    assert (PROJECT_ROOT / "pyproject.toml").exists()


def test_data_and_models_directories_exist():
    assert DATA_DIR == PROJECT_ROOT / "data"
    assert MODELS_DIR == DATA_DIR / "models"
    assert DATA_DIR.exists()
