import argparse
from pathlib import Path
import sys
from typing import Optional

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm_exp.config import bootstrap_environment, get_model_spec
from llm_exp.data.indopref_loader import prepare_evaluation_dataset
from llm_exp.inference.runner import run_model_inference
from llm_exp.metrics.reliability import evaluate_single_model
from llm_exp.utils.logger import setup_logger
from llm_exp.visualization.single_model import generate_single_model_plots


def execute_model_pipeline(
    model_key: str,
    stage: str = "all",
    env_file: Optional[str] = None,
    config_path: Optional[str] = None,
    project_root: Optional[str] = None,
    models_cache: Optional[str] = None,
    max_tasks: Optional[int] = None,
    cleanup_cache_after: Optional[bool] = None,
    show_inline: bool = False,
) -> int:
    boot_logger = setup_logger(f"llm_exp.{model_key}")
    cfg, paths = bootstrap_environment(
        env_file=env_file,
        config_path=config_path,
        project_root=project_root,
        models_cache_dir=models_cache,
        authenticate_hf=(stage in {"all", "infer"}),
        logger=boot_logger,
    )
    spec = get_model_spec(model_key, models_config_path=cfg.models_config_path)
    logger = setup_logger(
        f"llm_exp.{spec.key}",
        log_dir=paths.logs_dir,
        run_name=f"{spec.key}_{stage}",
        logging_config=cfg.logging,
    )

    try:
        df_eval = (
            prepare_evaluation_dataset(paths=paths, logger=logger, config=cfg)
            if stage in {"all", "infer"} or not paths.eval_parquet_path.exists()
            else pd.read_parquet(paths.eval_parquet_path)
        )

        if stage in {"all", "infer"}:
            run_model_inference(
                spec=spec,
                df_eval=df_eval,
                paths=paths,
                logger=logger,
                config=cfg,
                cleanup_cache_after=cleanup_cache_after,
                max_tasks=max_tasks,
            )

        if stage in {"all", "eval", "plot"}:
            artifacts = evaluate_single_model(spec=spec, paths=paths, logger=logger, df_eval=df_eval, config=cfg)
            if stage in {"all", "plot"}:
                generate_single_model_plots(
                    artifacts=artifacts,
                    logger=logger,
                    config=cfg,
                    show_inline=show_inline,
                )

        logger.info("stage_done | model=%s | stage=%s", spec.key, stage)
        return 0
    except Exception:
        logger.exception("stage_failed | model=%s | stage=%s", spec.key, stage)
        raise


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run single-model evaluation stage.")
    parser.add_argument("--model", required=True, help="Model key from models.yaml.")
    parser.add_argument("--stage", default="all", choices=["all", "infer", "eval", "plot"], help="Pipeline stage.")
    parser.add_argument("--env-file", type=str, default=None, help="Path to .env file.")
    parser.add_argument("--config", type=str, default=None, help="Path to experiment.yaml.")
    parser.add_argument("--project-root", type=str, default=None, help="Override workspace root directory.")
    parser.add_argument("--models-cache", type=str, default=None, help="Override local cache directory.")
    parser.add_argument("--max-tasks", type=int, default=None, help="Optional limit on pending inference tasks.")
    parser.add_argument("--no-cleanup-cache", action="store_true", help="Keep downloaded weights in local cache.")
    parser.add_argument("--show-inline", action="store_true", help="Display multi-panel plots inline.")
    args = parser.parse_args(argv)

    return execute_model_pipeline(
        model_key=args.model,
        stage=args.stage,
        env_file=args.env_file,
        config_path=args.config,
        project_root=args.project_root,
        models_cache=args.models_cache,
        max_tasks=args.max_tasks,
        cleanup_cache_after=False if args.no_cleanup_cache else None,
        show_inline=args.show_inline,
    )


if __name__ == "__main__":
    raise SystemExit(main())
