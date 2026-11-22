import logging
from typing import Optional

from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

from llm_exp.config import ExperimentConfig, ExperimentPaths, load_experiment_config
from llm_exp.data.schema import validate_dataset_schema


def prepare_evaluation_dataset(
    paths: ExperimentPaths,
    logger: logging.Logger,
    config: Optional[ExperimentConfig] = None,
    sample_size: Optional[int] = None,
    force_regenerate: bool = False,
) -> pd.DataFrame:
    cfg = config or paths.config or load_experiment_config()
    ds_cfg = cfg.dataset
    target_sample_size = sample_size if sample_size is not None else ds_cfg.sample_size
    seed = cfg.evaluation.random_seed

    eval_path = paths.eval_parquet_path
    raw_path = paths.raw_parquet_path
    df_raw: Optional[pd.DataFrame] = None

    if eval_path.exists() and not force_regenerate:
        cached_silver = pd.read_parquet(eval_path)
        size_matched = (
            (target_sample_size is None and len(cached_silver) >= ds_cfg.min_silver_records)
            or (target_sample_size is not None and len(cached_silver) == target_sample_size)
        )
        if size_matched:
            if not validate_dataset_schema(cached_silver, logger, dataset_config=ds_cfg):
                raise ValueError("Cached Silver dataset failed schema verification.")
            logger.info("dataset_cache_hit | split=eval | rows=%d | path=%s", len(cached_silver), eval_path)
            return cached_silver
        logger.info(
            "dataset_cache_miss | split=eval | cached=%d | target=%s",
            len(cached_silver),
            f">={ds_cfg.min_silver_records}" if target_sample_size is None else target_sample_size,
        )

    if raw_path.exists() and not force_regenerate:
        cached_raw = pd.read_parquet(raw_path)
        if len(cached_raw) > ds_cfg.min_raw_records:
            df_raw = cached_raw
            logger.info("dataset_cache_hit | split=raw | rows=%d | path=%s", len(df_raw), raw_path)
        else:
            logger.warning(
                "dataset_cache_invalid | split=raw | rows=%d | min_rows=%d",
                len(cached_raw),
                ds_cfg.min_raw_records,
            )

    if df_raw is None:
        datasets_cache_dir = paths.models_cache_dir / "datasets"
        datasets_cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("dataset_fetch | hf_id=%s | split=%s", ds_cfg.hf_id, ds_cfg.split)
        dataset = load_dataset(
            ds_cfg.hf_id,
            split=ds_cfg.split,
            cache_dir=str(datasets_cache_dir),
        )
        df_raw = dataset.to_pandas()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_parquet(raw_path, index=False)
        logger.info("dataset_saved | split=raw | rows=%d | path=%s", len(df_raw), raw_path)

    if not validate_dataset_schema(df_raw, logger, dataset_config=ds_cfg):
        raise ValueError("Raw Bronze dataset failed schema verification.")

    duplicates = int(df_raw.duplicated(subset=["original_id"]).sum())
    if duplicates > 0:
        df_clean = df_raw.drop_duplicates(subset=["original_id"]).copy()
        logger.info("dataset_dedup | raw=%d | removed=%d | unique=%d", len(df_raw), duplicates, len(df_clean))
    else:
        df_clean = df_raw.copy()

    if target_sample_size is not None and len(df_clean) > target_sample_size:
        df_eval, _ = train_test_split(
            df_clean,
            train_size=target_sample_size,
            stratify=df_clean["domain"],
            random_state=seed,
        )
        cohort_mode = "subsample_stratified"
    else:
        df_eval = df_clean.copy()
        cohort_mode = "full_census"

    eval_path.parent.mkdir(parents=True, exist_ok=True)
    df_eval.to_parquet(eval_path, index=False)
    logger.info("dataset_saved | split=eval | mode=%s | rows=%d | path=%s", cohort_mode, len(df_eval), eval_path)
    return df_eval
