import argparse
from pathlib import Path
import sys
from typing import Optional

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qwen3_eval.config import bootstrap_environment, load_model_spec
from qwen3_eval.utils.logger import close_logger_handlers, setup_logger

VALID_STAGES = frozenset({"all", "infer", "eval", "plot"})


def execute_pipeline(
    stage: str = "all",
    env_file: Optional[str] = None,
    config_path: Optional[str] = None,
    project_root: Optional[str] = None,
    models_cache: Optional[str] = None,
    max_tasks: Optional[int] = None,
    cleanup_cache_after: Optional[bool] = None,
    show_inline: bool = False,
) -> int:
    if stage not in VALID_STAGES:
        raise ValueError(f"Unsupported stage '{stage}'. Expected one of: {', '.join(sorted(VALID_STAGES))}.")
    if max_tasks is not None and max_tasks < 1:
        raise ValueError(f"max_tasks must be >= 1 when specified, got {max_tasks}.")

    boot_logger = setup_logger("qwen3_eval.pipeline")
    try:
        cfg, paths = bootstrap_environment(
            env_file=env_file,
            config_path=config_path,
            project_root=project_root,
            models_cache_dir=models_cache,
            authenticate_hf=(stage in {"all", "infer"}),
            logger=boot_logger,
        )
    finally:
        close_logger_handlers(boot_logger)

    spec = load_model_spec(model_config_path=cfg.model_config_path)
    logger = setup_logger(
        f"qwen3_eval.{spec.key}",
        log_dir=paths.logs_dir,
        run_name=f"{spec.key}_{stage}",
        logging_config=cfg.logging,
    )

    try:
        if stage in {"all", "infer"} or not paths.eval_parquet_path.exists():
            from qwen3_eval.data.indopref_loader import prepare_evaluation_dataset

            df_eval = prepare_evaluation_dataset(paths=paths, logger=logger, config=cfg)
        else:
            df_eval = pd.read_parquet(paths.eval_parquet_path)

        if stage in {"all", "infer"}:
            from qwen3_eval.inference.runner import run_inference

            run_inference(
                spec=spec,
                df_eval=df_eval,
                paths=paths,
                logger=logger,
                config=cfg,
                cleanup_cache_after=cleanup_cache_after,
                max_tasks=max_tasks,
            )

        if stage in {"all", "eval", "plot"}:
            from qwen3_eval.metrics.reliability import evaluate_model

            artifacts = evaluate_model(spec=spec, paths=paths, logger=logger, df_eval=df_eval, config=cfg)
            if stage in {"all", "plot"}:
                from qwen3_eval.visualization.plotter import generate_evaluation_plots

                generate_evaluation_plots(
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
    finally:
        close_logger_handlers(logger)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run single-model Qwen3 evaluation pipeline.")
    parser.add_argument("--stage", default="all", choices=sorted(VALID_STAGES), help="Pipeline stage.")
    parser.add_argument("--env-file", type=str, default=None, help="Path to .env file.")
    parser.add_argument("--config", type=str, default=None, help="Path to experiment.yaml.")
    parser.add_argument("--project-root", type=str, default=None, help="Override workspace root directory.")
    parser.add_argument("--models-cache", type=str, default=None, help="Override local cache directory.")
    parser.add_argument("--max-tasks", type=int, default=None, help="Optional limit on pending inference tasks.")
    parser.add_argument("--no-cleanup-cache", action="store_true", help="Keep downloaded weights in local cache.")
    parser.add_argument("--show-inline", action="store_true", help="Display multi-panel plots inline.")
    args = parser.parse_args(argv)

    return execute_pipeline(
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
