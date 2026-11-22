import logging
from pathlib import Path

import pytest

from llm_exp.config import ExperimentConfig
from llm_exp.utils.gpu import cleanup_hf_model_cache, clear_gpu_memory
from llm_exp.utils.logger import setup_logger

pytestmark = pytest.mark.unit


def test_setup_logger_creates_dated_run_log_and_delayed_error_log(
    tmp_path: Path,
    exp_config: ExperimentConfig,
) -> None:
    logs_dir = tmp_path / "logs"

    ok_logger = setup_logger("test.unit.ok", log_dir=logs_dir, run_name="qwen_ok", logging_config=exp_config.logging)
    ok_logger.info("infer_done | model=qwen3_32b | done=6/6 | complete=True")
    for h in list(ok_logger.handlers):
        h.close()
        ok_logger.removeHandler(h)

    ok_files = sorted(logs_dir.rglob("*.log"))
    assert len(ok_files) == 1
    assert ok_files[0].name.startswith("qwen_ok_")
    assert not any(f.name.endswith("_errors.log") for f in ok_files)

    err_logger = setup_logger("test.unit.err", log_dir=logs_dir, run_name="qwen_err", logging_config=exp_config.logging)
    try:
        raise RuntimeError("Simulated GPU OOM")
    except RuntimeError:
        err_logger.exception("stage_failed | model=qwen3_32b | stage=infer")
    for h in list(err_logger.handlers):
        h.close()
        err_logger.removeHandler(h)

    err_files = sorted(logs_dir.rglob("qwen_err*_errors.log"))
    assert len(err_files) == 1
    content = err_files[0].read_text(encoding="utf-8")
    assert "Simulated GPU OOM" in content
    assert "Traceback" in content


def test_cleanup_hf_model_cache_removes_hub_dir(tmp_path: Path, test_logger: logging.Logger) -> None:
    cache_dir = tmp_path / "models"
    repo_dir = cache_dir / "hub" / "models--Qwen--Qwen3-32B"
    repo_dir.mkdir(parents=True)
    (repo_dir / "weights.bin").write_bytes(b"data")

    assert cleanup_hf_model_cache("Qwen/Qwen3-32B", cache_dir, test_logger) is True
    assert not repo_dir.exists()
    assert cleanup_hf_model_cache("Qwen/Qwen3-32B", cache_dir, test_logger) is False


def test_clear_gpu_memory_runs_safely(test_logger: logging.Logger) -> None:
    clear_gpu_memory(logger=test_logger)
