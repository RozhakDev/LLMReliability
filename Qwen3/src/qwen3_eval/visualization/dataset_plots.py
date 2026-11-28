import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qwen3_eval.config.settings import load_experiment_config
from qwen3_eval.data.labels import resolve_human_winner
from qwen3_eval.data.profiling import compute_domain_stats
from qwen3_eval.visualization.common import export_panels

_STEM_COLOR = "#64748b"
_MARKER_COLOR = "#1e293b"
_MEAN_COLOR = "#475569"
_ASSISTANT_A_COLOR = "#4c72b0"
_ASSISTANT_B_COLOR = "#dd8452"
_DIVIDER_COLOR = "#94a3b8"
_MIN_SEGMENT_LABEL_PCT = 14.0


def _build_balance_frame(dom_stats: pd.DataFrame, df_eval: pd.DataFrame) -> pd.DataFrame:
    sides = df_eval.apply(resolve_human_winner, axis=1)
    overall_a = float((sides == "A").mean() * 100.0)
    overall_row = pd.DataFrame(
        [{"domain": f"Overall (N={len(df_eval):,})", "a_pct": overall_a, "b_pct": 100.0 - overall_a}]
    )
    return pd.concat([dom_stats[["domain", "a_pct", "b_pct"]], overall_row], ignore_index=True)


def _draw_domain_lollipop_panel(
    ax: plt.Axes,
    dom_stats: pd.DataFrame,
    total: int,
    dataset_name: str,
) -> None:
    y_dom = np.arange(len(dom_stats))
    mean_dom = float(dom_stats["count"].mean())
    max_dom = float(dom_stats["count"].max())
    x_pad = max(max_dom * 0.03, 0.8)

    ax.axvline(mean_dom, color=_MEAN_COLOR, ls=":", lw=1.4, zorder=1, label=f"Mean ({mean_dom:.1f})")
    ax.hlines(y=y_dom, xmin=0.0, xmax=dom_stats["count"], color=_STEM_COLOR, lw=2.2, alpha=0.85, zorder=2)
    ax.scatter(dom_stats["count"], y_dom, color=_MARKER_COLOR, edgecolor="white", linewidth=0.9, s=78, zorder=3)
    for y_idx, (raw_cnt, raw_pct) in enumerate(zip(dom_stats["count"], dom_stats["share_pct"])):
        cnt = int(raw_cnt)
        pct = float(raw_pct)
        ax.text(
            cnt + x_pad,
            y_idx,
            f"{cnt} ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=9.2,
            fontweight="bold",
            color=_MARKER_COLOR,
        )
    ax.set_yticks(y_dom)
    ax.set_yticklabels(dom_stats["domain"])
    ax.set(
        xlabel="Unique Prompt Samples (Count and Share)",
        ylabel="Topical Domain",
        xlim=(0.0, max(max_dom * 1.34, 10.0)),
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(f"A. {dataset_name} Sample Count per Domain (N={total:,})", fontweight="bold", pad=10)


def _draw_preference_balance_panel(ax: plt.Axes, balance_df: pd.DataFrame, n_domains: int) -> None:
    y_bal = np.arange(len(balance_df))
    ax.barh(
        y_bal,
        balance_df["a_pct"],
        height=0.64,
        color=_ASSISTANT_A_COLOR,
        edgecolor="white",
        lw=1.0,
        alpha=0.92,
        label="Assistant A (response_1)",
    )
    ax.barh(
        y_bal,
        balance_df["b_pct"],
        left=balance_df["a_pct"],
        height=0.64,
        color=_ASSISTANT_B_COLOR,
        edgecolor="white",
        lw=1.0,
        alpha=0.92,
        label="Assistant B (response_2)",
    )
    ax.axvline(50.0, color=_MARKER_COLOR, ls="--", lw=1.3, zorder=4, label="50% Balance")
    ax.axhline(n_domains - 0.5, color=_DIVIDER_COLOR, ls="-", lw=1.0, zorder=3)
    for y_idx, (raw_a, raw_b) in enumerate(zip(balance_df["a_pct"], balance_df["b_pct"])):
        a_val = float(raw_a)
        b_val = float(raw_b)
        if a_val >= _MIN_SEGMENT_LABEL_PCT:
            ax.text(
                a_val / 2.0,
                y_idx,
                f"{a_val:.1f}%",
                va="center",
                ha="center",
                color="white",
                fontsize=8.8,
                fontweight="bold",
            )
        if b_val >= _MIN_SEGMENT_LABEL_PCT:
            ax.text(
                a_val + b_val / 2.0,
                y_idx,
                f"{b_val:.1f}%",
                va="center",
                ha="center",
                color="white",
                fontsize=8.8,
                fontweight="bold",
            )
    ax.set_yticks(y_bal)
    ax.set_yticklabels(balance_df["domain"])
    ax.set(xlabel="Human Reference Share (%)", ylabel="", xlim=(0.0, 100.0))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=8.8, frameon=False)
    ax.set_title("B. Human Reference Preference Balance (Input Ground Truth)", fontweight="bold", pad=10)


def plot_dataset_profile(
    df_eval: pd.DataFrame,
    results_dir: Path,
    logger: logging.Logger,
    dpi: Optional[int] = None,
    show_inline: bool = False,
    dataset_name: Optional[str] = None,
) -> None:
    dom_stats = compute_domain_stats(df_eval, ascending=True)
    if dom_stats.empty:
        logger.warning("plot_skip | figure=dataset_profile | reason=empty_dataset")
        return

    if dpi is None or dataset_name is None:
        exp_cfg = load_experiment_config()
        resolved_dpi = dpi if dpi is not None else exp_cfg.visualization.dpi
        resolved_name = dataset_name or exp_cfg.dataset.hf_id.split("/")[-1]
    else:
        resolved_dpi = dpi
        resolved_name = dataset_name

    balance_df = _build_balance_frame(dom_stats, df_eval)
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 5.2), gridspec_kw={"width_ratios": [1.05, 1.15]})
    _draw_domain_lollipop_panel(axes[0], dom_stats, len(df_eval), resolved_name)
    _draw_preference_balance_panel(axes[1], balance_df, len(dom_stats))
    export_panels(
        fig,
        axes,
        ["plot_dataset_domain_distribution", "plot_dataset_preference_balance"],
        results_dir,
        logger,
        resolved_dpi,
        show_inline,
    )
