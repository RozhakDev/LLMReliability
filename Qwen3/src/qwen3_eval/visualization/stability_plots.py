import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from qwen3_eval.metrics.artifacts import EvaluationArtifacts
from qwen3_eval.metrics.contingency import build_paired_positional_frame
from qwen3_eval.visualization.common import export_panels, plot_bars


def plot_positional_heatmaps(
    artifacts: EvaluationArtifacts,
    prompts: List[str],
    logger: logging.Logger,
    dpi: int,
    show_inline: bool,
) -> None:
    if not prompts:
        logger.warning("plot_skip | figure=positional_heatmaps | reason=empty_prompts")
        return
    fig, axes = plt.subplots(1, len(prompts), figsize=(16.5, 4.6))
    for ax, p_id in zip(np.atleast_1d(axes), prompts):
        sub = artifacts.df_valid[artifacts.df_valid["prompt_id"] == p_id]
        paired = build_paired_positional_frame(sub)
        ct = pd.crosstab(paired["AB"], paired["BA"]).reindex(index=[1, 2], columns=[1, 2], fill_value=0)
        ct.index = ct.columns = ["Resp. 1", "Resp. 2"]
        n_pair = max(len(paired), 1)
        annot = [[f"{v}\n({v / n_pair * 100:.1f}%)" for v in row] for row in ct.to_numpy()]
        cons = (np.diag(ct).sum() / len(paired) * 100) if len(paired) > 0 else 0.0
        sns.heatmap(
            ct,
            annot=annot,
            fmt="",
            cmap="Blues",
            ax=ax,
            cbar=False,
            square=True,
            annot_kws={"fontsize": 11, "fontweight": "bold"},
            linewidths=1,
            linecolor="gray",
        )
        ax.set(xlabel="Choice in BA Order (Reversed)", ylabel="Choice in AB Order (Standard)")
        ax.set_title(f"{p_id}: Positional Matrix (Consistent: {cons:.1f}%)", fontweight="bold", pad=10)
        ax.tick_params(axis="y", rotation=0)
    export_panels(fig, axes, [f"plot_positional_heatmap_{p.lower()}" for p in prompts], artifacts.results_dir, logger, dpi, show_inline)


def plot_prompt_stability(
    artifacts: EvaluationArtifacts,
    prompts: List[str],
    logger: logging.Logger,
    dpi: int,
    show_inline: bool,
) -> None:
    df_stab = artifacts.df_prompt_stab
    if df_stab.empty or len(prompts) < 3 or not set(prompts[:3]).issubset(set(df_stab.columns)):
        return
    p1, p2, p3 = prompts[:3]
    pairs = [(f"{p1} vs {p2}", p1, p2), (f"{p1} vs {p3}", p1, p3), (f"{p2} vs {p3}", p2, p3)]
    labels = [lbl for lbl, _, _ in pairs]
    cons_rates = [float((df_stab[a] == df_stab[b]).mean() * 100) for _, a, b in pairs]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.6))
    plot_bars(
        axes[0],
        labels,
        cons_rates,
        "Blues_d",
        "A. Cross-Prompt Decision Consistency (Higher is Better)",
        "Prompt Pairwise Comparison",
        "Consistency Rate (%)",
    )
    plot_bars(
        axes[1],
        labels,
        [100.0 - c for c in cons_rates],
        "Reds_d",
        "B. Cross-Prompt Decision Flip Rate (Lower is Better)",
        "Prompt Pairwise Comparison",
        "Flip Rate (%)",
    )
    export_panels(fig, axes, ["plot_prompt_consistency", "plot_prompt_flip_rate"], artifacts.results_dir, logger, dpi, show_inline)
