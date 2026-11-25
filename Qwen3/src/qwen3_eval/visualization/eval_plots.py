import logging
from typing import List

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns

from qwen3_eval.metrics.artifacts import EvaluationArtifacts
from qwen3_eval.visualization.common import annotate_bars, export_panels, plot_bars


def plot_accuracy_and_slots(
    artifacts: EvaluationArtifacts,
    prompts: List[str],
    logger: logging.Logger,
    dpi: int,
    show_inline: bool,
) -> None:
    df_valid = artifacts.df_valid
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.6))
    acc_s = (
        df_valid.assign(ok=(df_valid["logical_decision"] == df_valid["logical_human"]) * 100)
        .groupby("prompt_id")["ok"]
        .agg(["mean", "count"])
        .reindex(prompts)
        .fillna(0)
    )
    acc_err = (1.96 * np.sqrt(acc_s["mean"] * (100 - acc_s["mean"]) / acc_s["count"].replace(0, np.nan)).fillna(0)).tolist()
    plot_bars(
        axes[0],
        prompts,
        acc_s["mean"].tolist(),
        "Blues_d",
        "A. Model Accuracy against Human Preference",
        "Prompt Strategy",
        "Accuracy (%)",
        yerr=acc_err,
    )
    axes[0].axhline(50.0, color="gray", ls="--", lw=1.5, zorder=1, label="Chance Baseline (50%)")
    axes[0].legend(loc="upper right")

    df_pos = df_valid.groupby(["position", "parsed_decision"]).size().unstack(fill_value=0).reindex(columns=["A", "B"], fill_value=0)
    df_pos = (df_pos.div(df_pos.sum(axis=1).replace(0, np.nan), axis=0) * 100).fillna(0).reset_index().melt(id_vars="position", var_name="Slot", value_name="Pct")
    df_pos["Slot"] = df_pos["Slot"].map({"A": "Position A (1st)", "B": "Position B (2nd)"})
    if not df_pos.empty:
        sns.barplot(data=df_pos, x="Slot", y="Pct", hue="position", ax=axes[1], palette="Set2", edgecolor="black", alpha=0.85)
        axes[1].axhline(50.0, color="gray", ls="--", lw=1.5, zorder=1, label="Ideal Unbiased Split (50%)")
        axes[1].legend(title="Presentation Order", loc="upper right")
        annotate_bars(axes[1], fmt="{:.1f}%")
    else:
        axes[1].text(0.5, 0.5, "No valid predictions", ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set(xlabel="Displayed Position Slot", ylabel="Selection Proportion (%)", ylim=(0, 100))
    axes[1].set_title("B. Positional Slot Selection Distribution", fontweight="bold", pad=10)
    export_panels(fig, axes, ["plot_model_accuracy", "plot_slot_selection_distribution"], artifacts.results_dir, logger, dpi, show_inline)


def plot_positional_and_kappa(
    artifacts: EvaluationArtifacts,
    logger: logging.Logger,
    dpi: int,
    show_inline: bool,
) -> None:
    df_m = artifacts.df_metrics_table
    if df_m.empty:
        logger.warning("plot_skip | figure=positional_and_kappa | reason=empty_metrics")
        return
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 4.6))
    ci = df_m["Kappa 95% CI"].str.strip("[]").str.split(",", expand=True).astype(float)
    k_vals = df_m["Cohen Kappa"].astype(float)
    k_err = [(k_vals - ci[0]).clip(lower=0).tolist(), (ci[1] - k_vals).clip(lower=0).tolist()]

    plot_bars(
        axes[0],
        df_m["Prompt"].tolist(),
        df_m["Consistency (%)"].tolist(),
        "Blues_d",
        "A. Positional Consistency (Higher is Better)",
        "Prompt Strategy",
        "Consistency Rate (%)",
    )
    plot_bars(
        axes[1],
        df_m["Prompt"].tolist(),
        df_m["Flip Rate (%)"].tolist(),
        "Reds_d",
        "B. Decision Flip Rate (Lower is Better)",
        "Prompt Strategy",
        "Flip Rate (%)",
    )
    plot_bars(
        axes[2],
        df_m["Prompt"].tolist(),
        k_vals.tolist(),
        "Greens_d",
        "C. Cohen's Kappa Agreement with Human Preference",
        "Prompt Strategy",
        "Cohen's Kappa Coefficient",
        ylim=(0.0, 1.15),
        yerr=k_err,
        fmt="{:.3f}",
    )
    axes[2].axhline(0.40, color="darkgreen", ls=":", lw=1.5, zorder=1, label="Moderate Agreement (0.40)")
    axes[2].axhline(0.60, color="navy", ls="--", lw=1.2, zorder=1, label="Substantial Agreement (0.60)")
    axes[2].legend(loc="upper right", fontsize=9)
    export_panels(
        fig,
        axes,
        ["plot_positional_consistency", "plot_positional_flip_rate", "plot_cohen_kappa"],
        artifacts.results_dir,
        logger,
        dpi,
        show_inline,
    )


def plot_parsing_and_domains(
    artifacts: EvaluationArtifacts,
    logger: logging.Logger,
    dpi: int,
    show_inline: bool,
) -> None:
    df_valid, inv = artifacts.df_valid, artifacts.invalid_count
    df_dom = artifacts.df_domain_accuracy.sort_values("is_correct", ascending=True)
    total = len(df_valid) + inv
    v_pct = (len(df_valid) / total) * 100 if total > 0 else 0.0
    i_pct = (inv / total) * 100 if total > 0 else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 4.8), gridspec_kw={"width_ratios": [1.0, 1.35]})
    if total > 0:
        axes[0].pie(
            [len(df_valid), inv] if inv > 0 else [len(df_valid)],
            colors=["#27ae60", "#e74c3c"] if inv > 0 else ["#27ae60"],
            startangle=90,
            wedgeprops=dict(width=0.4, edgecolor="white", lw=2),
        )
    axes[0].text(0, 0, f"{v_pct:.1f}%\nValid\n(N={total:,})", ha="center", va="center", fontweight="bold", fontsize=11.5)
    axes[0].set_title("A. Decision Format Parsing Success Ratio", fontweight="bold", pad=10)
    axes[0].legend(
        handles=[
            Patch(facecolor="#27ae60", label=f"Valid (A/B): {len(df_valid):,} ({v_pct:.1f}%)"),
            Patch(facecolor="#e74c3c", label=f"Invalid: {inv:,} ({i_pct:.1f}%)"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=2,
        fontsize=9,
    )

    mean_acc = float(df_valid["is_correct"].mean()) if not df_valid.empty else 0.0
    if not df_dom.empty:
        bars = axes[1].barh(df_dom["domain"], df_dom["is_correct"], color=sns.color_palette("viridis", len(df_dom)), edgecolor="black", alpha=0.85)
        axes[1].axvline(mean_acc, color="red", ls="--", lw=1.5, zorder=1, label=f"Overall Mean ({mean_acc:.1f}%)")
        axes[1].legend(loc="lower right", fontsize=9)
        annotate_bars(axes[1], bars=bars, fmt="{:.1f}%", horizontal=True)
    else:
        axes[1].text(0.5, 0.5, "No valid predictions", ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set(xlabel="Accuracy Rate (%)", ylabel="Topical Domain", xlim=(0, 105))
    axes[1].set_title("B. Model Evaluator Accuracy across Topical Domains", fontweight="bold", pad=10)
    export_panels(fig, axes, ["plot_parsing_success_ratio", "plot_domain_accuracy"], artifacts.results_dir, logger, dpi, show_inline)
