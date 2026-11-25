import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from qwen3_eval.config.io_utils import read_valid_jsonl_records
from qwen3_eval.config.settings import load_experiment_config

_RESUME_REQUIRED_FIELDS = frozenset({"sample_id", "prompt_id", "position"})


def load_processed_combinations(
    out_file: Path,
    logger: Optional[logging.Logger] = None,
) -> Set[Tuple[int, str, str]]:
    if not out_file.exists():
        return set()
    records = read_valid_jsonl_records(out_file, _RESUME_REQUIRED_FIELDS, phase="resume", logger=logger)
    return {(int(rec["sample_id"]), str(rec["prompt_id"]), str(rec["position"])) for rec in records}


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
