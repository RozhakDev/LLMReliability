import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from qwen3_eval.config.io_utils import atomic_write_text
from qwen3_eval.inference.tasks import determine_human_winner

_REQUIRED_RECORD_FIELDS = frozenset(
    {"sample_id", "domain", "prompt_id", "position", "human_preference", "raw_output"}
)


def parse_decision(raw_output: str) -> str:
    clean = str(raw_output).strip().upper()
    match = re.search(r"\b([AB])\b", clean)
    return match.group(1) if match else "INVALID"


def load_and_parse_results(
    out_file: Path,
    logger: logging.Logger,
    df_eval: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    if not out_file.exists():
        raise FileNotFoundError(f"Inference output file not found: {out_file}")

    records: List[Dict[str, Any]] = []
    corrupt_lines = 0
    with open(out_file, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if not isinstance(rec, dict):
                    raise TypeError("record must be a JSON object")
                missing = _REQUIRED_RECORD_FIELDS.difference(rec.keys())
                if missing:
                    raise KeyError(f"missing fields: {sorted(missing)}")
                rec["sample_id"] = int(rec["sample_id"])
                records.append(rec)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                corrupt_lines += 1
                logger.warning("jsonl_skip_line | phase=eval | line=%d | path=%s | err=%s", line_no, out_file, exc)

    if corrupt_lines > 0:
        logger.warning("jsonl_corrupt_summary | phase=eval | skipped=%d | path=%s", corrupt_lines, out_file)

    if not records:
        raise ValueError(f"No valid JSONL records found in '{out_file}'.")

    if df_eval is not None and not df_eval.empty:
        eval_gt_map = {int(row["original_id"]): determine_human_winner(row) for _, row in df_eval.iterrows()}
        records_updated = False
        for rec in records:
            sample_id = int(rec["sample_id"])
            if sample_id in eval_gt_map:
                true_winner = eval_gt_map[sample_id]
                expected_human = true_winner if rec["position"] == "AB" else ("B" if true_winner == "A" else "A")
                if rec.get("human_preference") != expected_human:
                    rec["human_preference"] = expected_human
                    records_updated = True
        if records_updated:
            serialized = "".join(json.dumps(rec, ensure_ascii=False) + "\n" for rec in records)
            atomic_write_text(out_file, serialized)
            logger.info("gt_reconciled | path=%s", out_file)

    for rec in records:
        parsed = parse_decision(rec["raw_output"])
        rec["parsed_decision"] = parsed

        if parsed == "INVALID":
            rec["logical_decision"] = "INVALID"
        elif rec["position"] == "AB":
            rec["logical_decision"] = 1 if parsed == "A" else 2
        else:
            rec["logical_decision"] = 2 if parsed == "A" else 1

        if rec["position"] == "AB":
            rec["logical_human"] = 1 if rec["human_preference"] == "A" else 2
        else:
            rec["logical_human"] = 2 if rec["human_preference"] == "A" else 1

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
