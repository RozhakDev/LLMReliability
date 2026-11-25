import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from qwen3_eval.config.io_utils import atomic_write_text, read_valid_jsonl_records
from qwen3_eval.data.labels import (
    flip_candidate_label,
    map_positional_to_logical,
    resolve_human_winner,
)

_REQUIRED_RECORD_FIELDS = frozenset(
    {"sample_id", "domain", "prompt_id", "position", "human_preference", "raw_output"}
)


def parse_decision(raw_output: str) -> str:
    clean = str(raw_output).strip().upper()
    match = re.search(r"\b([AB])\b", clean)
    return match.group(1) if match else "INVALID"


def _reconcile_ground_truth_records(
    records: List[Dict[str, Any]],
    df_eval: pd.DataFrame,
    out_file: Path,
    logger: logging.Logger,
) -> None:
    eval_gt_map = {int(row["original_id"]): resolve_human_winner(row) for _, row in df_eval.iterrows()}
    records_updated = False
    for rec in records:
        sample_id = int(rec["sample_id"])
        if sample_id in eval_gt_map:
            true_winner = eval_gt_map[sample_id]
            expected_human = true_winner if rec["position"] == "AB" else flip_candidate_label(true_winner)
            if rec.get("human_preference") != expected_human:
                rec["human_preference"] = expected_human
                records_updated = True

    if records_updated:
        serialized = "".join(json.dumps(rec, ensure_ascii=False) + "\n" for rec in records)
        atomic_write_text(out_file, serialized)
        logger.info("gt_reconciled | path=%s", out_file)


def _annotate_decision_columns(records: List[Dict[str, Any]]) -> None:
    for rec in records:
        position = str(rec["position"])
        parsed = parse_decision(rec["raw_output"])
        rec["parsed_decision"] = parsed
        rec["logical_decision"] = map_positional_to_logical(parsed, position)
        rec["logical_human"] = map_positional_to_logical(str(rec["human_preference"]), position)


def load_and_parse_results(
    out_file: Path,
    logger: logging.Logger,
    df_eval: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    if not out_file.exists():
        raise FileNotFoundError(f"Inference output file not found: {out_file}")

    records = read_valid_jsonl_records(out_file, _REQUIRED_RECORD_FIELDS, phase="eval", logger=logger)
    if not records:
        raise ValueError(f"No valid JSONL records found in '{out_file}'.")

    if df_eval is not None and not df_eval.empty:
        _reconcile_ground_truth_records(records, df_eval, out_file, logger)

    _annotate_decision_columns(records)
    df_all = pd.DataFrame(records).drop_duplicates(
        subset=["sample_id", "prompt_id", "position"],
        keep="last",
    )
    invalid_count = int((df_all["parsed_decision"] == "INVALID").sum())
    failure_rate = (invalid_count / len(df_all)) * 100.0 if len(df_all) > 0 else 0.0
    logger.info("eval_parsed | total=%d | invalid=%d | invalid_pct=%.2f", len(df_all), invalid_count, failure_rate)

    df_valid = df_all[df_all["parsed_decision"] != "INVALID"].copy()
    if not df_valid.empty:
        df_valid["logical_decision"] = df_valid["logical_decision"].astype(int)
        df_valid["logical_human"] = df_valid["logical_human"].astype(int)
    return df_all, df_valid, invalid_count
