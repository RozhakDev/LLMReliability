import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from qwen3_eval.data.labels import resolve_human_winner
from qwen3_eval.visualization.common import annotate_bars, export_panels, plot_bars


def plot_dataset_profile(
    df_eval: pd.DataFrame,
    results_dir: Path,
    logger: logging.Logger,
    dpi: int,
    show_inline: bool,
) -> None:
    if df_eval.empty or "domain" not in df_eval.columns:
        logger.warning("plot_skip | figure=dataset_profile | reason=empty_dataset")
        return

    dom_counts = (
        df_eval.groupby("domain", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["count", "domain"], ascending=[True, False])
    )
    sides = df_eval.apply(resolve_human_winner, axis=1)
    total = len(df_eval)
    pct_a = float((sides == "A").sum() / total * 100.0)
    pct_b = float((sides == "B").sum() / total * 100.0)

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 4.8), gridspec_kw={"width_ratios": [1.35, 1.0]})
    bars = axes[0].barh(
        dom_counts["domain"],
        dom_counts["count"],
        color=sns.color_palette("viridis", len(dom_counts)),
        edgecolor="black",
        alpha=0.85,
    )
    mean_dom = float(dom_counts["count"].mean())
    max_dom = float(dom_counts["count"].max())
    axes[0].axvline(mean_dom, color="navy", ls="--", lw=1.4, zorder=1, label=f"Mean per Domain ({mean_dom:.1f})")
    axes[0].legend(loc="lower right", fontsize=9)
    annotate_bars(axes[0], bars=bars, fmt="{:.0f}", horizontal=True)
    axes[0].set(xlabel="Number of Unique Samples", ylabel="Topical Domain", xlim=(0.0, max(max_dom * 1.18, 10.0)))
    axes[0].set_title("A. Sample Distribution across IndoPref Domains", fontweight="bold", pad=10)

    plot_bars(
        axes[1],
        ["Assistant A (Resp. 1)", "Assistant B (Resp. 2)"],
        [pct_a, pct_b],
        "Set2",
        "B. Human Reference Preference Balance",
        "Reference Preferred Candidate",
        "Sample Proportion (%)",
    )
    axes[1].axhline(50.0, color="gray", ls="--", lw=1.5, zorder=1, label="Balanced Split (50%)")
    axes[1].legend(loc="upper right")
    export_panels(
        fig,
        axes,
        ["plot_dataset_domain_distribution", "plot_dataset_preference_balance"],
        results_dir,
        logger,
        dpi,
        show_inline,
    )
