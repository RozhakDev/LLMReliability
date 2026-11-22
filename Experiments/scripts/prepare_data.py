import argparse
from pathlib import Path
import sys
from typing import Optional

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm_exp.config import bootstrap_environment
from llm_exp.data.indopref_loader import prepare_evaluation_dataset
from llm_exp.utils.logger import setup_logger


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare shared Silver evaluation dataset.")
    parser.add_argument("--env-file", type=str, default=None, help="Path to .env file.")
    parser.add_argument("--config", type=str, default=None, help="Path to experiment.yaml.")
    parser.add_argument("--project-root", type=str, default=None, help="Override workspace root directory.")
    parser.add_argument("--models-cache", type=str, default=None, help="Override local HF cache directory.")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional stratified subsample size.")
    parser.add_argument("--force", action="store_true", help="Force re-download and re-deduplication.")
    args = parser.parse_args(argv)

    boot_logger = setup_logger("llm_exp.prepare_data")
    cfg, paths = bootstrap_environment(
        env_file=args.env_file,
        config_path=args.config,
        project_root=args.project_root,
        models_cache_dir=args.models_cache,
        authenticate_hf=True,
        logger=boot_logger,
    )
    logger = setup_logger(
        "llm_exp.prepare_data",
        log_dir=paths.logs_dir,
        run_name="prepare_data",
        logging_config=cfg.logging,
    )

    try:
        df_eval = prepare_evaluation_dataset(
            paths=paths,
            logger=logger,
            config=cfg,
            sample_size=args.sample_size,
            force_regenerate=args.force,
        )
        logger.info("dataset_ready | rows=%d | path=%s", len(df_eval), paths.eval_parquet_path)
        return 0
    except Exception:
        logger.exception("dataset_failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
