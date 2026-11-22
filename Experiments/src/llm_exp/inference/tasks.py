import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from llm_exp.config.settings import load_experiment_config


def determine_human_winner(row_data: Any) -> str:
    if "actual_score" in row_data:
        val = str(row_data["actual_score"]).strip().upper()
        if "ASSISTANT A" in val or val in {"A", "1", "1.0"}:
            return "A"
        if "ASSISTANT B" in val or val in {"B", "2", "2.0"}:
            return "B"
        try:
            parsed_val = json.loads(str(row_data["actual_score"]).replace("'", '"'))
            if isinstance(parsed_val, (list, tuple)) and len(parsed_val) >= 2:
                return "A" if float(parsed_val[0]) >= float(parsed_val[1]) else "B"
        except Exception:
            pass

    if "model_1" in row_data and "chosen_model" in row_data:
        if str(row_data["chosen_model"]).strip() == str(row_data["model_1"]).strip():
            return "A"
        if "rejected_model" in row_data and str(row_data["rejected_model"]).strip() == str(row_data["model_1"]).strip():
            return "B"

    if "label" in row_data:
        label_val = str(row_data["label"]).strip().upper()
        if "ASSISTANT A" in label_val or label_val in {"1", "0", "A"}:
            return "A"
        if "ASSISTANT B" in label_val or label_val in {"2", "B"}:
            return "B"

    return "A"


def load_processed_combinations(
    out_file: Path,
    logger: Optional[logging.Logger] = None,
) -> Set[Tuple[int, str, str]]:
    processed: Set[Tuple[int, str, str]] = set()
    if not out_file.exists():
        return processed

    corrupt_lines = 0
    with open(out_file, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                processed.add((int(record["sample_id"]), str(record["prompt_id"]), str(record["position"])))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                corrupt_lines += 1
                if logger is not None:
                    logger.warning("jsonl_skip_line | phase=resume | line=%d | path=%s | err=%s", line_no, out_file, exc)

    if corrupt_lines > 0 and logger is not None:
        logger.warning("jsonl_corrupt_summary | phase=resume | skipped=%d | path=%s", corrupt_lines, out_file)
    return processed


def build_evaluation_tasks(
    df_eval: pd.DataFrame,
    prompts: Optional[Sequence[str]] = None,
    positions: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    if prompts is None or positions is None:
        eval_cfg = load_experiment_config().evaluation
        prompts = prompts if prompts is not None else eval_cfg.prompts
        positions = positions if positions is not None else eval_cfg.positions

    tasks: List[Dict[str, Any]] = []
    for _, row in df_eval.iterrows():
        sample_id = int(row["original_id"])
        for prompt_id in prompts:
            for position in positions:
                tasks.append(
                    {
                        "row": row,
                        "sample_id": sample_id,
                        "prompt_id": str(prompt_id),
                        "position": str(position),
                        "key": (sample_id, str(prompt_id), str(position)),
                    }
                )
    return tasks
