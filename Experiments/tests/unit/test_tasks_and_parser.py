import json
import logging
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

from llm_exp.inference import build_evaluation_tasks, determine_human_winner, load_processed_combinations
from llm_exp.metrics.parser import load_and_parse_results, parse_decision

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("raw_output", "expected"),
    [
        ("A", "A"),
        ("  b \n", "B"),
        ("Jawaban saya: A.", "A"),
        ("(B)", "B"),
        ("Saya ragu keduanya bagus", "INVALID"),
        ("AB", "INVALID"),
        ("", "INVALID"),
    ],
)
def test_parse_decision(raw_output: str, expected: str) -> None:
    assert parse_decision(raw_output) == expected


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"actual_score": "Assistant A"}, "A"),
        ({"actual_score": "Assistant B"}, "B"),
        ({"actual_score": "1.0"}, "A"),
        ({"actual_score": "2"}, "B"),
        ({"actual_score": "[2.0, 1.0]"}, "A"),
        ({"actual_score": "[1.0, 3.5]"}, "B"),
        ({"model_1": "m1", "chosen_model": "m1"}, "A"),
        ({"model_1": "m1", "chosen_model": "m2", "rejected_model": "m1"}, "B"),
        ({"label": "Assistant B"}, "B"),
        ({"label": "0"}, "A"),
        ({}, "A"),
    ],
)
def test_determine_human_winner_priorities(row: dict, expected: str) -> None:
    assert determine_human_winner(row) == expected


def test_build_evaluation_tasks_matrix(make_eval_df: Callable[[int], pd.DataFrame]) -> None:
    df = make_eval_df(3)
    tasks = build_evaluation_tasks(df)
    assert len(tasks) == 18
    assert tasks[0]["key"] == (100, "P1", "AB")
    assert tasks[1]["key"] == (100, "P1", "BA")


def test_corrupt_eof_recovery_reconciliation_and_int_dtype(
    tmp_path: Path,
    make_eval_df: Callable[[int], pd.DataFrame],
    test_logger: logging.Logger,
) -> None:
    df_eval = make_eval_df(1)
    out_file = tmp_path / "main_outputs_final.jsonl"
    rec_ab = {
        "sample_id": 100,
        "domain": "SHP-AskEngineers",
        "prompt_id": "P1",
        "position": "AB",
        "human_preference": "B",
        "raw_output": "A",
    }
    rec_ba = {
        "sample_id": 100,
        "domain": "SHP-AskEngineers",
        "prompt_id": "P1",
        "position": "BA",
        "human_preference": "B",
        "raw_output": "Tidak yakin",
    }
    out_file.write_text(
        json.dumps(rec_ab) + "\n" + json.dumps(rec_ba) + "\n" + '{"sample_id": 101, "raw_out',
        encoding="utf-8",
    )

    assert load_processed_combinations(out_file, logger=test_logger) == {(100, "P1", "AB"), (100, "P1", "BA")}
    df_all, df_valid, invalid_count = load_and_parse_results(out_file, logger=test_logger, df_eval=df_eval)
    assert len(df_all) == 2
    assert invalid_count == 1
    assert len(df_valid) == 1
    assert df_valid["human_preference"].iloc[0] == "A"
    assert pd.api.types.is_integer_dtype(df_valid["logical_decision"])
    assert pd.api.types.is_integer_dtype(df_valid["logical_human"])


def test_load_and_parse_results_rejects_empty_file(tmp_path: Path, test_logger: logging.Logger) -> None:
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="No valid JSONL records"):
        load_and_parse_results(empty_file, logger=test_logger)
