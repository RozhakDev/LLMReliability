import argparse
from pathlib import Path
import sys
from typing import Optional

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm_exp.config import bootstrap_environment
from llm_exp.metrics.aggregator import aggregate_all_models
from llm_exp.utils.logger import setup_logger
from llm_exp.visualization.comparison import generate_comparison_plots


def execute_comparison(
    env_file: Optional[str] = None,
    config_path: Optional[str] = None,
    project_root: Optional[str] = None,
    recompute_from_jsonl: bool = True,
    show_inline: bool = False,
) -> int:
    boot_logger = setup_logger("llm_exp.comparison")
    cfg, paths = bootstrap_environment(
        env_file=env_file,
        config_path=config_path,
        project_root=project_root,
        authenticate_hf=False,
        logger=boot_logger,
    )
    logger = setup_logger(
        "llm_exp.comparison",
        log_dir=paths.logs_dir,
        run_name="comparison",
        logging_config=cfg.logging,
    )

    try:
        tables = aggregate_all_models(
            paths=paths,
            logger=logger,
            config=cfg,
            recompute_from_jsonl=recompute_from_jsonl,
        )
        generate_comparison_plots(tables=tables, logger=logger, config=cfg, show_inline=show_inline)
        logger.info("compare_done | models=%d | path=%s", len(tables.summary_df), tables.comparison_dir)
        return 0
    except Exception:
        logger.exception("compare_failed")
        raise


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate metrics and render cross-model comparison plots.")
    parser.add_argument("--env-file", type=str, default=None, help="Path to .env file.")
    parser.add_argument("--config", type=str, default=None, help="Path to experiment.yaml.")
    parser.add_argument("--project-root", type=str, default=None, help="Override workspace root directory.")
    parser.add_argument("--no-recompute", action="store_true", help="Read existing CSV summaries instead of JSONL.")
    parser.add_argument("--show-inline", action="store_true", help="Display comparison figures inline.")
    args = parser.parse_args(argv)

    return execute_comparison(
        env_file=args.env_file,
        config_path=args.config,
        project_root=args.project_root,
        recompute_from_jsonl=not args.no_recompute,
        show_inline=args.show_inline,
    )


if __name__ == "__main__":
    raise SystemExit(main())
