import logging
from typing import Callable

import pandas as pd
import pytest

from qwen3_eval.config import ExperimentConfig, ExperimentPaths, load_model_spec
from qwen3_eval.metrics import (
    build_parsed_predictions_preview,
    build_positional_bias_diagnostics,
)
from qwen3_eval.metrics.reliability import evaluate_model
from qwen3_eval.metrics.statistics import bootstrap_kappa, compute_positional_metrics
from qwen3_eval.visualization import generate_evaluation_plots

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


def test_single_model_evaluation_tables_and_all_12_plots(
    qwen3_config: ExperimentConfig,
    qwen3_paths: ExperimentPaths,
    make_eval_df: Callable[[int], pd.DataFrame],
    write_synthetic_jsonl: Callable[..., None],
    test_logger: logging.Logger,
) -> None:
    df_eval = make_eval_df(12)
    df_eval.to_parquet(qwen3_paths.eval_parquet_path, index=False)

    spec = load_model_spec(qwen3_config.model_config_path)
    out_file = qwen3_paths.results_dir / qwen3_config.artifacts.output_jsonl
    write_synthetic_jsonl(out_file, df_eval, seed=100, invalid_every=15, perfect=False)

    artifacts = evaluate_model(spec, qwen3_paths, test_logger, df_eval=df_eval, config=qwen3_config)
    assert artifacts.results_dir == qwen3_paths.results_dir
    assert len(artifacts.df_all) == 72
    assert artifacts.invalid_count == 5
    assert len(artifacts.df_valid) == 67

    preview_df = build_parsed_predictions_preview(artifacts, n_rows=6)
    assert len(preview_df) == 6
    assert {"mapped_choice", "mapped_human", "is_correct"}.issubset(set(preview_df.columns))

    diag_df = build_positional_bias_diagnostics(artifacts, alpha=qwen3_config.evaluation.alpha)
    diag_map = dict(zip(diag_df["Diagnostic Metric"], diag_df["Value"]))
    mcnemar_key = f"McNemar Continuity-Corrected (alpha={qwen3_config.evaluation.alpha})"
    assert f"chi2 = {artifacts.mcnemar_stat:.4f}" in diag_map[mcnemar_key]

    art = qwen3_config.artifacts
    for table_file in (
        art.metrics_summary_csv,
        art.metrics_summary_parquet,
        art.detailed_results_parquet,
        art.domain_accuracy_csv,
        art.prompt_stability_csv,
        art.executive_summary_csv,
    ):
        assert (artifacts.results_dir / table_file).is_file()

    generate_evaluation_plots(artifacts, test_logger, config=qwen3_config, dpi=72, show_inline=False)
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
