import logging
from typing import Callable

import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from llm_exp.config import ExperimentConfig, ExperimentPaths, get_model_spec, list_model_keys
from llm_exp.metrics.aggregator import aggregate_all_models
from llm_exp.metrics.reliability import evaluate_single_model
from llm_exp.metrics.statistics import bootstrap_kappa, compute_positional_metrics
from llm_exp.visualization import generate_comparison_plots, generate_single_model_plots

pytestmark = [pytest.mark.integration, pytest.mark.statsmodels]


def test_bootstrap_kappa_determinism_and_constant_guard() -> None:
    y_true = [1, 2, 1, 2, 1, 1, 2, 2, 1, 2]
    y_pred = [1, 2, 1, 1, 1, 1, 2, 1, 1, 2]
    assert bootstrap_kappa(y_true, y_pred, n_iterations=200, seed=42) == bootstrap_kappa(
        y_true, y_pred, n_iterations=200, seed=42
    )
    assert bootstrap_kappa([1, 1, 1], [1, 1, 1], n_iterations=20, seed=42) == (0.0, 0.0)


def test_positional_metrics_handles_missing_ba_column() -> None:
    df_only_ab = pd.DataFrame(
        [
            {"sample_id": 100, "prompt_id": "P1", "position": "AB", "parsed_decision": "A", "logical_decision": 1, "logical_human": 1},
            {"sample_id": 101, "prompt_id": "P1", "position": "AB", "parsed_decision": "A", "logical_decision": 1, "logical_human": 1},
        ]
    )
    df_pos, _, _, _, stat, pval, is_sig = compute_positional_metrics(df_only_ab)
    assert df_pos.empty
    assert stat == 0.0
    assert pval == 1.0
    assert is_sig is False


def test_five_model_eval_aggregation_and_plots(
    exp_config: ExperimentConfig,
    exp_paths: ExperimentPaths,
    make_eval_df: Callable[[int], pd.DataFrame],
    write_synthetic_jsonl: Callable[..., None],
    test_logger: logging.Logger,
) -> None:
    df_eval = make_eval_df(12)
    df_eval.to_parquet(exp_paths.eval_parquet_path, index=False)

    for offset, key in enumerate(list_model_keys(exp_config.models_config_path)):
        spec = get_model_spec(key, exp_config.models_config_path)
        out_file = exp_paths.model_results_dir(key) / exp_config.artifacts.output_jsonl
        write_synthetic_jsonl(
            out_file,
            df_eval,
            seed=100 + offset,
            invalid_every=15 if offset == 1 else 0,
            perfect=(offset == 2),
        )
        artifacts = evaluate_single_model(spec, exp_paths, test_logger, df_eval=df_eval, config=exp_config)
        if offset == 0:
            assert len(artifacts.df_valid) == 72
            assert artifacts.invalid_count == 0
            generate_single_model_plots(artifacts, test_logger, config=exp_config, dpi=72, show_inline=False)
            for name in (
                "plot_model_accuracy",
                "plot_slot_selection_distribution",
                "plot_positional_consistency",
                "plot_positional_flip_rate",
                "plot_cohen_kappa",
                "plot_parsing_success_ratio",
                "plot_domain_accuracy",
                "plot_positional_heatmap_p1",
                "plot_positional_heatmap_p2",
                "plot_positional_heatmap_p3",
                "plot_prompt_consistency",
                "plot_prompt_flip_rate",
            ):
                assert (artifacts.results_dir / f"{name}.png").is_file()
                assert (artifacts.results_dir / f"{name}.svg").is_file()

    tables = aggregate_all_models(exp_paths, test_logger, config=exp_config, recompute_from_jsonl=True)
    assert len(tables.summary_df) == 5
    assert "Primary Strength" in tables.summary_df.columns
    assert "Evaluations (N)" in tables.summary_df.columns

    tables_cached = aggregate_all_models(exp_paths, test_logger, config=exp_config, recompute_from_jsonl=False)
    assert len(tables_cached.summary_df) == 5

    generate_comparison_plots(tables, test_logger, config=exp_config, dpi=72, show_inline=False)
    for cmp_name in (
        "comparison_accuracy_by_prompt",
        "comparison_positional_consistency",
        "comparison_cohen_kappa",
        "comparison_3way_prompt_stability",
        "comparison_domain_accuracy_heatmap",
    ):
        assert (tables.comparison_dir / f"{cmp_name}.png").is_file()
        assert (tables.comparison_dir / f"{cmp_name}.svg").is_file()
