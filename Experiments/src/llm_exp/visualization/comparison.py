import logging
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns

from llm_exp.config import ExperimentConfig, load_experiment_config
from llm_exp.metrics.aggregator import CrossModelComparisonTables
from llm_exp.visualization.common import annotate_bars, configure_matplotlib_theme, export_panels, plot_bars


def _plot_accuracy_by_prompt(tables: CrossModelComparisonTables, logger: logging.Logger, dpi: int, show_inline: bool) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    sns.barplot(data=tables.prompt_breakdown_df, x="Model", y="Accuracy (%)", hue="Prompt", ax=ax, palette="Blues_d", edgecolor="black", alpha=0.88)
    ax.axhline(50.0, color="gray", ls="--", lw=1.5, zorder=1, label="Chance Baseline (50%)")
    ax.set(xlabel="Evaluator Model (16-bit BF16)", ylabel="Accuracy (%)", ylim=(0, 105))
    ax.set_title("Model Accuracy Comparison across Prompt Strategies", fontweight="bold", pad=10)
    ax.tick_params(axis="x", rotation=12)
    ax.legend(loc="upper right", ncol=2)
    annotate_bars(ax, fmt="{:.1f}%")
    export_panels(fig, ax, ["comparison_accuracy_by_prompt"], tables.comparison_dir, logger, dpi, show_inline)


def _plot_consistency_and_kappa(tables: CrossModelComparisonTables, logger: logging.Logger, dpi: int, show_inline: bool) -> None:
    summary = tables.summary_df
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 4.8))
    cons_vals = summary["Overall Consistency (%)"].fillna(0.0).tolist()
    kappa_vals = summary["P1 Cohen Kappa"].fillna(0.0).tolist()
    min_k = min(kappa_vals) if kappa_vals else 0.0
    k_ylim = (0.0, 1.15) if min_k >= 0.0 else (min_k - 0.15, 1.15)

    plot_bars(axes[0], summary["Model"].tolist(), cons_vals, "Blues_d", "A. Overall Positional Consistency across Models", "Evaluator Model", "Consistency Rate (%)", ylim=(0.0, 105.0))
    axes[0].tick_params(axis="x", rotation=12)
    plot_bars(axes[1], summary["Model"].tolist(), kappa_vals, "Greens_d", "B. Cohen's Kappa Agreement (P1 Baseline) across Models", "Evaluator Model", "Cohen's Kappa Coefficient", ylim=k_ylim, fmt="{:.3f}")
    axes[1].axhline(0.40, color="darkgreen", ls=":", lw=1.5, zorder=1, label="Moderate Agreement (0.40)")
    axes[1].axhline(0.60, color="navy", ls="--", lw=1.2, zorder=1, label="Substantial Agreement (0.60)")
    axes[1].tick_params(axis="x", rotation=12)
    axes[1].legend(loc="upper right", fontsize=9)
    export_panels(fig, axes, ["comparison_positional_consistency", "comparison_cohen_kappa"], tables.comparison_dir, logger, dpi, show_inline)


def _plot_stability_and_domains(tables: CrossModelComparisonTables, logger: logging.Logger, dpi: int, show_inline: bool) -> None:
    summary, dom = tables.summary_df, tables.domain_matrix_df
    if summary["3-Way Prompt Consistency (%)"].notna().any():
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        plot_bars(ax, summary["Model"].tolist(), summary["3-Way Prompt Consistency (%)"].fillna(0.0).tolist(), "Purples_d", "Simultaneous 3-Way Prompt Stability across Models", "Evaluator Model", "3-Way Consistency Rate (%)", ylim=(0.0, 105.0))
        ax.tick_params(axis="x", rotation=12)
        export_panels(fig, ax, ["comparison_3way_prompt_stability"], tables.comparison_dir, logger, dpi, show_inline)

    if not dom.empty:
        fig, ax = plt.subplots(figsize=(max(8.5, 2.0 * len(dom.columns) + 2.5), max(4.8, 0.55 * len(dom.index) + 1.5)))
        sns.heatmap(dom, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax, linewidths=0.6, linecolor="gray", cbar_kws={"label": "Accuracy (%)"})
        ax.set(xlabel="Evaluator Model", ylabel="Topical Domain")
        ax.set_title("Topical Domain Accuracy Matrix across Evaluator Models (%)", fontweight="bold", pad=10)
        ax.tick_params(axis="y", rotation=0)
        ax.tick_params(axis="x", rotation=12)
        export_panels(fig, ax, ["comparison_domain_accuracy_heatmap"], tables.comparison_dir, logger, dpi, show_inline)


def generate_comparison_plots(
    tables: CrossModelComparisonTables,
    logger: logging.Logger,
    config: Optional[ExperimentConfig] = None,
    dpi: Optional[int] = None,
    show_inline: bool = False,
) -> None:
    cfg = config or load_experiment_config()
    resolved_dpi = dpi if dpi is not None else cfg.visualization.dpi
    configure_matplotlib_theme(cfg.visualization)
    _plot_accuracy_by_prompt(tables, logger, resolved_dpi, show_inline)
    _plot_consistency_and_kappa(tables, logger, resolved_dpi, show_inline)
    _plot_stability_and_domains(tables, logger, resolved_dpi, show_inline)
