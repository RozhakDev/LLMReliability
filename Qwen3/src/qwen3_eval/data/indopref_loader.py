import logging
from pathlib import Path
from typing import Optional, Tuple

from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

from qwen3_eval.config import ExperimentConfig, ExperimentPaths
from qwen3_eval.config.io_utils import atomic_write_via_temp
from qwen3_eval.config.schemas import DatasetConfig
from qwen3_eval.data.schema import validate_dataset_schema


def _try_load_cached_eval(
    eval_path: Path,
    target_sample_size: Optional[int],
    ds_cfg: DatasetConfig,
    logger: logging.Logger,
) -> Optional[pd.DataFrame]:
    if not eval_path.exists():
        return None
    try:
        cached_silver = pd.read_parquet(eval_path)
    except (OSError, ValueError) as exc:
        logger.warning("dataset_cache_corrupt | split=eval | path=%s | err=%s", eval_path, exc)
        return None

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
    return None


def _load_or_fetch_raw(
    raw_path: Path,
    models_cache_dir: Path,
    ds_cfg: DatasetConfig,
    force_regenerate: bool,
    logger: logging.Logger,
) -> pd.DataFrame:
    if raw_path.exists() and not force_regenerate:
        try:
            cached_raw = pd.read_parquet(raw_path)
        except (OSError, ValueError) as exc:
            logger.warning("dataset_cache_corrupt | split=raw | path=%s | err=%s", raw_path, exc)
            cached_raw = None

        if cached_raw is not None:
            if len(cached_raw) > ds_cfg.min_raw_records:
                logger.info("dataset_cache_hit | split=raw | rows=%d | path=%s", len(cached_raw), raw_path)
                return cached_raw
            logger.warning(
                "dataset_cache_invalid | split=raw | rows=%d | min_rows=%d",
                len(cached_raw),
                ds_cfg.min_raw_records,
            )

    datasets_cache_dir = models_cache_dir / "datasets"
    datasets_cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("dataset_fetch | hf_id=%s | split=%s", ds_cfg.hf_id, ds_cfg.split)
    dataset = load_dataset(ds_cfg.hf_id, split=ds_cfg.split, cache_dir=str(datasets_cache_dir))
    df_raw = dataset.to_pandas()
    atomic_write_via_temp(raw_path, lambda tmp: df_raw.to_parquet(tmp, index=False))
    logger.info("dataset_saved | split=raw | rows=%d | path=%s", len(df_raw), raw_path)
    return df_raw


def _deduplicate_and_subsample(
    df_raw: pd.DataFrame,
    target_sample_size: Optional[int],
    seed: int,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, str]:
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
        return df_eval, "subsample_stratified"
    return df_clean.copy(), "full_census"


def prepare_evaluation_dataset(
    paths: ExperimentPaths,
    logger: logging.Logger,
    config: Optional[ExperimentConfig] = None,
    sample_size: Optional[int] = None,
    force_regenerate: bool = False,
) -> pd.DataFrame:
    cfg = config or paths.config
    ds_cfg = cfg.dataset
    target_sample_size = sample_size if sample_size is not None else ds_cfg.sample_size
    if target_sample_size is not None and target_sample_size < 1:
        raise ValueError(f"sample_size must be >= 1, got {target_sample_size}.")

    eval_path = paths.eval_parquet_path
    if not force_regenerate:
        cached = _try_load_cached_eval(eval_path, target_sample_size, ds_cfg, logger)
        if cached is not None:
            return cached

    df_raw = _load_or_fetch_raw(
        paths.raw_parquet_path,
        paths.models_cache_dir,
        ds_cfg,
        force_regenerate,
        logger,
    )
    if not validate_dataset_schema(df_raw, logger, dataset_config=ds_cfg):
        raise ValueError("Raw Bronze dataset failed schema verification.")

    df_eval, cohort_mode = _deduplicate_and_subsample(df_raw, target_sample_size, cfg.evaluation.random_seed, logger)
    atomic_write_via_temp(eval_path, lambda tmp: df_eval.to_parquet(tmp, index=False))
    logger.info("dataset_saved | split=eval | mode=%s | rows=%d | path=%s", cohort_mode, len(df_eval), eval_path)
    return df_eval
