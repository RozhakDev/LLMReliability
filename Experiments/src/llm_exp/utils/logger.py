from datetime import datetime
import logging
import os
from pathlib import Path
import sys
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from llm_exp.config.schemas import LoggingConfig


def _resolve_tz(tz_name: str) -> Optional[ZoneInfo]:
    if not tz_name:
        return None
    try:
        return ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        return None


class _TimezoneFormatter(logging.Formatter):
    def __init__(self, fmt: str, datefmt: str, tz: Optional[ZoneInfo]) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self._tz = tz

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        dt = datetime.fromtimestamp(record.created, tz=self._tz)
        return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S")


def _parse_level(level: int | str) -> int:
    if isinstance(level, int):
        return level
    return int(getattr(logging, str(level).upper(), logging.INFO))


def setup_logger(
    name: str = "llm_exp",
    log_dir: Optional[Path] = None,
    run_name: Optional[str] = None,
    logging_config: Optional[LoggingConfig] = None,
    level: Optional[int | str] = None,
) -> logging.Logger:
    resolved_level = _parse_level(
        level if level is not None else (logging_config.level if logging_config else logging.INFO)
    )
    tz_name = logging_config.timezone if logging_config else os.environ.get("LOG_TIMEZONE", "").strip()
    tz = _resolve_tz(tz_name)
    now = datetime.now(tz=tz)

    logger = logging.getLogger(name)
    logger.setLevel(resolved_level)
    logger.propagate = False

    formatter = _TimezoneFormatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        tz=tz,
    )

    for existing_handler in list(logger.handlers):
        existing_handler.close()
        logger.removeHandler(existing_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(resolved_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        ts_fmt = logging_config.timestamp_format if logging_config else "%Y%m%d_%H%M%S"
        date_sub = logging_config.date_subfolder if logging_config else True
        fn_pat = logging_config.filename_pattern if logging_config else "{run_name}_{timestamp}.log"
        err_pat = logging_config.error_filename_pattern if logging_config else "{run_name}_{timestamp}_errors.log"
        split_err = logging_config.separate_error_log if logging_config else True

        safe_run = (run_name or name.split(".")[-1]).replace("/", "_").replace(" ", "_")
        timestamp = now.strftime(ts_fmt)
        target_dir = (log_dir / now.strftime("%Y-%m-%d")) if date_sub else log_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        run_log_path = target_dir / fn_pat.format(run_name=safe_run, timestamp=timestamp)
        file_handler = logging.FileHandler(run_log_path, encoding="utf-8")
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        if split_err:
            err_log_path = target_dir / err_pat.format(run_name=safe_run, timestamp=timestamp)
            err_handler = logging.FileHandler(err_log_path, encoding="utf-8", delay=True)
            err_handler.setLevel(logging.ERROR)
            err_handler.setFormatter(formatter)
            logger.addHandler(err_handler)

        logger.info(
            "log_init | run_id=%s_%s | tz=%s | path=%s",
            safe_run,
            timestamp,
            tz_name or "local",
            run_log_path,
        )

    return logger
