import logging
from typing import Optional, Sequence

import pandas as pd

from qwen3_eval.config.schemas import DatasetConfig
from qwen3_eval.config.settings import load_experiment_config


def validate_dataset_schema(
    df: pd.DataFrame,
    logger: logging.Logger,
    dataset_config: Optional[DatasetConfig] = None,
    required_columns: Optional[Sequence[str]] = None,
    text_columns: Optional[Sequence[str]] = None,
) -> bool:
    ds_cfg = dataset_config or load_experiment_config().dataset
    req_cols = list(required_columns if required_columns is not None else ds_cfg.required_columns)
    txt_cols = list(text_columns if text_columns is not None else ds_cfg.text_columns)

    missing = [col for col in req_cols if col not in df.columns]
    if missing:
        logger.error("schema_invalid | missing=%s", missing)
        return False

    if int(df[req_cols].isnull().sum().sum()) > 0:
        logger.error("schema_invalid | reason=nulls")
        return False

    for text_col in txt_cols:
        if (df[text_col].astype(str).str.strip() == "").any():
            logger.error("schema_invalid | reason=empty_text | col=%s", text_col)
            return False

    logger.info("schema_ok | rows=%d | cols=%d", df.shape[0], df.shape[1])
    return True
