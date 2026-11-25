import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Dict, FrozenSet, List, Optional

import yaml


def load_yaml_mapping(path: Path, label: str = "YAML config") -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML syntax in {label} '{path}': {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{label} '{path}' must contain a non-empty YAML mapping.")
    return raw


def atomic_write_via_temp(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        writer(tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    def _write(tmp_path: Path) -> None:
        with open(tmp_path, "w", encoding=encoding) as handle:
            handle.write(content)
            handle.flush()

    atomic_write_via_temp(path, _write)


def read_valid_jsonl_records(
    jsonl_path: Path,
    required_fields: FrozenSet[str],
    phase: str,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    corrupt_lines = 0
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if not isinstance(rec, dict):
                    raise TypeError("record must be a JSON object")
                missing = required_fields.difference(rec.keys())
                if missing:
                    raise KeyError(f"missing fields: {sorted(missing)}")
                if "sample_id" in required_fields:
                    rec["sample_id"] = int(rec["sample_id"])
                records.append(rec)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                corrupt_lines += 1
                if logger is not None:
                    logger.warning(
                        "jsonl_skip_line | phase=%s | line=%d | path=%s | err=%s",
                        phase,
                        line_no,
                        jsonl_path,
                        exc,
                    )

    if corrupt_lines > 0 and logger is not None:
        logger.warning(
            "jsonl_corrupt_summary | phase=%s | skipped=%d | path=%s",
            phase,
            corrupt_lines,
            jsonl_path,
        )
    return records
