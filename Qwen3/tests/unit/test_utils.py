import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from qwen3_eval.config import ExperimentConfig
from qwen3_eval.utils.gpu import cleanup_hf_model_cache, clear_gpu_memory, purge_other_hf_caches
from qwen3_eval.utils.logger import close_logger_handlers, setup_logger

pytestmark = pytest.mark.unit


def test_setup_logger_creates_dated_run_log_and_delayed_error_log(
    tmp_path: Path,
    qwen3_config: ExperimentConfig,
) -> None:
    logs_dir = tmp_path / "logs"

    ok_logger = setup_logger("test.qwen3.ok", log_dir=logs_dir, run_name="qwen_ok", logging_config=qwen3_config.logging)
    ok_logger.info("infer_done | model=qwen3_32b | done=6/6 | complete=True")
    close_logger_handlers(ok_logger)

    ok_files = sorted(logs_dir.rglob("*.log"))
    assert len(ok_files) == 1
    assert ok_files[0].name.startswith("qwen_ok_")
    assert not any(f.name.endswith("_errors.log") for f in ok_files)

    err_logger = setup_logger("test.qwen3.err", log_dir=logs_dir, run_name="qwen_err", logging_config=qwen3_config.logging)
    try:
        raise RuntimeError("Simulated GPU OOM")
    except RuntimeError:
        err_logger.exception("stage_failed | model=qwen3_32b | stage=infer")
    close_logger_handlers(err_logger)

    err_files = sorted(logs_dir.rglob("qwen_err*_errors.log"))
    assert len(err_files) == 1
    content = err_files[0].read_text(encoding="utf-8")
    assert "Simulated GPU OOM" in content
    assert "Traceback" in content


def test_setup_logger_sanitizes_run_name_against_path_traversal(
    tmp_path: Path,
    qwen3_config: ExperimentConfig,
) -> None:
    logs_dir = tmp_path / "logs"
    logger = setup_logger(
        "test.qwen3.sanitize",
        log_dir=logs_dir,
        run_name="../../escape\\bad name",
        logging_config=qwen3_config.logging,
    )
    close_logger_handlers(logger)

    created = sorted(logs_dir.rglob("*.log"))
    assert len(created) == 1
    assert created[0].resolve().is_relative_to(logs_dir.resolve())
    assert ".." not in created[0].name


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


def test_cleanup_hf_model_cache_survives_oserror_gracefully(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    test_logger: logging.Logger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    cache_dir = tmp_path / "models"
    repo_dir = cache_dir / "hub" / "models--Qwen--Qwen3-32B"
    repo_dir.mkdir(parents=True)

    with (
        patch("qwen3_eval.utils.gpu.shutil.rmtree", side_effect=PermissionError("locked")),
        caplog.at_level(logging.WARNING, logger=test_logger.name),
    ):
        assert cleanup_hf_model_cache("Qwen/Qwen3-32B", cache_dir, test_logger) is False

    assert "hf_cache_remove_failed" in caplog.text


def test_setup_logger_respects_log_timezone_env_when_config_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOG_TIMEZONE", "Asia/Jakarta")
    boot_logger = setup_logger("test.qwen3.tz_env")
    try:
        formatter = boot_logger.handlers[0].formatter
        assert getattr(formatter, "_tz", None) is not None
        assert str(getattr(formatter, "_tz")) == "Asia/Jakarta"
    finally:
        close_logger_handlers(boot_logger)


def test_purge_other_hf_caches_keeps_active_qwen3_and_removes_stale_and_xet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    test_logger: logging.Logger,
) -> None:
    fake_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    cache_dir = tmp_path / "models"
    active_dir = cache_dir / "models--Qwen--Qwen3-32B"
    stale_local = cache_dir / "models--org--stale-model-a"
    stale_home = fake_home / ".cache" / "huggingface" / "hub" / "models--org--stale-model-b"
    xet_home = fake_home / ".cache" / "huggingface" / "xet"
    locks_dir = cache_dir / ".locks"
    datasets_dir = cache_dir / "datasets"

    for target in (active_dir, stale_local, stale_home, xet_home, locks_dir, datasets_dir):
        target.mkdir(parents=True)
        (target / "f.bin").write_bytes(b"x")

    removed = purge_other_hf_caches("Qwen/Qwen3-32B", cache_dir, test_logger)
    assert removed == 4
    assert active_dir.exists()
    assert datasets_dir.exists()
    assert not stale_local.exists()
    assert not stale_home.exists()
    assert not xet_home.exists()
    assert not locks_dir.exists()


def test_clear_gpu_memory_runs_safely(test_logger: logging.Logger) -> None:
    clear_gpu_memory(logger=test_logger)
