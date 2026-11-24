import logging
from typing import Callable

import pandas as pd
import pytest

from qwen3_eval.config import ExperimentConfig
from qwen3_eval.data import validate_dataset_schema

pytestmark = pytest.mark.unit


def test_validate_dataset_schema_accepts_valid_df(
    make_eval_df: Callable[[int], pd.DataFrame],
    qwen3_config: ExperimentConfig,
    test_logger: logging.Logger,
) -> None:
    df = make_eval_df(4)
    assert validate_dataset_schema(df, test_logger, dataset_config=qwen3_config.dataset) is True


def test_validate_dataset_schema_rejects_missing_column(
    make_eval_df: Callable[[int], pd.DataFrame],
    qwen3_config: ExperimentConfig,
    test_logger: logging.Logger,
) -> None:
    df = make_eval_df(3).drop(columns=["actual_score"])
    assert validate_dataset_schema(df, test_logger, dataset_config=qwen3_config.dataset) is False


def test_validate_dataset_schema_rejects_null_values(
    make_eval_df: Callable[[int], pd.DataFrame],
    qwen3_config: ExperimentConfig,
    test_logger: logging.Logger,
) -> None:
    df = make_eval_df(3)
    df.loc[1, "domain"] = None
    assert validate_dataset_schema(df, test_logger, dataset_config=qwen3_config.dataset) is False


@pytest.mark.parametrize("text_col", ["input", "response_1", "response_2"])
def test_validate_dataset_schema_rejects_blank_text(
    make_eval_df: Callable[[int], pd.DataFrame],
    qwen3_config: ExperimentConfig,
    test_logger: logging.Logger,
    text_col: str,
) -> None:
    df = make_eval_df(3)
    df.loc[0, text_col] = "   "
    assert validate_dataset_schema(df, test_logger, dataset_config=qwen3_config.dataset) is False
