import logging
from pathlib import Path

import pytest

from llm_exp.config import ExperimentConfig
from llm_exp.utils.gpu import cleanup_hf_model_cache, clear_gpu_memory, purge_other_hf_caches
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


def test_cleanup_hf_model_cache_removes_hub_and_xet_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    test_logger: logging.Logger,
) -> None:
    fake_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    cache_dir = tmp_path / "models"
    repo_dir = cache_dir / "hub" / "models--Qwen--Qwen3-32B"
    xet_local = cache_dir / "xet"
    xet_home = fake_home / ".cache" / "huggingface" / "xet"
    for target in (repo_dir, xet_local, xet_home):
        target.mkdir(parents=True)
        (target / "chunk.bin").write_bytes(b"data")

    assert cleanup_hf_model_cache("Qwen/Qwen3-32B", cache_dir, test_logger) is True
    assert not repo_dir.exists()
    assert not xet_local.exists()
    assert not xet_home.exists()
    assert cleanup_hf_model_cache("Qwen/Qwen3-32B", cache_dir, test_logger) is False


def test_setup_logger_respects_log_timezone_env_when_config_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOG_TIMEZONE", "Asia/Jakarta")
    boot_logger = setup_logger("test.unit.tz_env")
    try:
        formatter = boot_logger.handlers[0].formatter
        assert getattr(formatter, "_tz", None) is not None
        assert str(getattr(formatter, "_tz")) == "Asia/Jakarta"
    finally:
        for h in list(boot_logger.handlers):
            h.close()
            boot_logger.removeHandler(h)


def test_purge_other_hf_caches_keeps_active_model_and_removes_stale_and_xet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    test_logger: logging.Logger,
) -> None:
    fake_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    cache_dir = tmp_path / "models"
    active_dir = cache_dir / "models--zai-org--GLM-4-32B-0414"
    stale_local = cache_dir / "models--CohereLabs--aya-expanse-32b"
    stale_home = fake_home / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-32B"
    xet_home = fake_home / ".cache" / "huggingface" / "xet"
    locks_dir = cache_dir / ".locks"
    datasets_dir = cache_dir / "datasets"

    for target in (active_dir, stale_local, stale_home, xet_home, locks_dir, datasets_dir):
        target.mkdir(parents=True)
        (target / "f.bin").write_bytes(b"x")

    removed = purge_other_hf_caches("zai-org/GLM-4-32B-0414", cache_dir, test_logger)
    assert removed == 4
    assert active_dir.exists()
    assert datasets_dir.exists()
    assert not stale_local.exists()
    assert not stale_home.exists()
    assert not xet_home.exists()
    assert not locks_dir.exists()


def test_clear_gpu_memory_runs_safely(test_logger: logging.Logger) -> None:
    clear_gpu_memory(logger=test_logger)
