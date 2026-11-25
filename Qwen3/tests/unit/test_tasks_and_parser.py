import json
import logging
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

from qwen3_eval.config.io_utils import read_valid_jsonl_records
from qwen3_eval.data.labels import map_positional_to_logical, order_candidates, resolve_human_winner
from qwen3_eval.inference import build_evaluation_tasks, load_processed_combinations
from qwen3_eval.metrics.parser import load_and_parse_results, parse_decision

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
def test_resolve_human_winner_priorities(row: dict, expected: str) -> None:
    assert resolve_human_winner(row) == expected


def test_build_evaluation_tasks_matrix(make_eval_df: Callable[[int], pd.DataFrame]) -> None:
    df = make_eval_df(3)
    tasks = build_evaluation_tasks(df)
    assert len(tasks) == 18
    assert tasks[0]["key"] == (100, "P1", "AB")
    assert tasks[1]["key"] == (100, "P1", "BA")


def test_corrupt_eof_malformed_record_reconciliation_and_int_dtype(
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
    malformed_missing_keys = {"sample_id": 100, "prompt_id": "P2"}
    out_file.write_text(
        json.dumps(rec_ab)
        + "\n"
        + json.dumps(rec_ba)
        + "\n"
        + json.dumps(malformed_missing_keys)
        + "\n"
        + '{"sample_id": 101, "raw_out',
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
    assert not list(tmp_path.glob("*.tmp"))


def test_load_and_parse_results_rejects_empty_file(tmp_path: Path, test_logger: logging.Logger) -> None:
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="No valid JSONL records"):
        load_and_parse_results(empty_file, logger=test_logger)


@pytest.mark.parametrize(
    ("position", "expected_pair", "expected_human"),
    [
        ("AB", ("R1", "R2"), "A"),
        ("BA", ("R2", "R1"), "B"),
    ],
)
def test_order_candidates_swaps_responses_and_label(
    position: str,
    expected_pair: tuple[str, str],
    expected_human: str,
) -> None:
    row = {"response_1": "R1", "response_2": "R2", "actual_score": "Assistant A"}
    resp_a, resp_b, label = order_candidates(row, position)
    assert (resp_a, resp_b) == expected_pair
    assert label == expected_human


@pytest.mark.parametrize(
    ("choice", "position", "expected"),
    [
        ("A", "AB", 1),
        ("B", "AB", 2),
        ("A", "BA", 2),
        ("B", "BA", 1),
        ("INVALID", "AB", "INVALID"),
        ("INVALID", "BA", "INVALID"),
    ],
)
def test_map_positional_to_logical(choice: str, position: str, expected: int | str) -> None:
    assert map_positional_to_logical(choice, position) == expected


def test_read_valid_jsonl_records_skips_corrupt_and_missing_fields(
    tmp_path: Path,
    test_logger: logging.Logger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    jsonl_file = tmp_path / "sample.jsonl"
    jsonl_file.write_text(
        '{"sample_id": "42", "prompt_id": "P1", "position": "AB"}\n'
        '{"sample_id": 43}\n'
        "not-json\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger=test_logger.name):
        records = read_valid_jsonl_records(
            jsonl_file,
            frozenset({"sample_id", "prompt_id", "position"}),
            phase="resume",
            logger=test_logger,
        )
    assert len(records) == 1
    assert records[0]["sample_id"] == 42
    assert "jsonl_corrupt_summary | phase=resume | skipped=2" in caplog.text
