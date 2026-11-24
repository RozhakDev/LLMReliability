import logging
from pathlib import Path
import re
from typing import Any, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from qwen3_eval.config.schemas import VisualizationConfig
from qwen3_eval.config.settings import load_experiment_config


def configure_matplotlib_theme(viz_config: Optional[VisualizationConfig] = None) -> None:
    cfg = viz_config or load_experiment_config().visualization
    sns.set_theme(style=cfg.style, font_scale=cfg.font_scale)
    plt.rcParams.update(
        {
            "font.sans-serif": cfg.font_family,
            "font.family": "sans-serif",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 9.5,
            "figure.dpi": cfg.dpi,
            "savefig.dpi": cfg.dpi,
        }
    )


def annotate_bars(
    ax: plt.Axes,
    bars: Optional[Sequence[Any]] = None,
    yerr: Optional[Sequence[float]] = None,
    fmt: str = "{:.2f}%",
    horizontal: bool = False,
) -> None:
    targets = bars if bars is not None else ax.patches
    for idx, bar in enumerate(targets):
        val = bar.get_width() if horizontal else bar.get_height()
        span = bar.get_height() if horizontal else bar.get_width()
        if not np.isnan(val) and val > 0 and span > 0:
            err = float(yerr[idx]) if yerr is not None and idx < len(yerr) else 0.0
            xy = (
                (val + err, bar.get_y() + span / 2.0)
                if horizontal
                else (bar.get_x() + span / 2.0, val + err)
            )
            ax.annotate(
                fmt.format(val),
                xy,
                xytext=(4, 0) if horizontal else (0, 4),
                textcoords="offset points",
                ha="left" if horizontal else "center",
                va="center" if horizontal else "bottom",
                fontsize=9.5 if horizontal else 10,
                fontweight="bold",
                zorder=5,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.9),
            )


def plot_bars(
    ax: plt.Axes,
    x: Sequence[str],
    y: Sequence[float],
    palette: str,
    title: str,
    xlabel: str,
    ylabel: str,
    ylim: Tuple[float, float] = (0.0, 100.0),
    yerr: Optional[Union[Sequence[float], List[List[float]]]] = None,
    fmt: str = "{:.2f}%",
) -> None:
    bars = ax.bar(
        x,
        y,
        width=0.55,
        yerr=yerr,
        capsize=5 if yerr is not None else 0,
        color=sns.color_palette(palette, len(x)),
        edgecolor="black",
        alpha=0.85,
    )
    ax.set(xlabel=xlabel, ylabel=ylabel, ylim=ylim)
    ax.set_title(title, fontweight="bold", pad=10)
    err_top: Optional[Sequence[float]] = None
    if isinstance(yerr, list) and len(yerr) == 2 and isinstance(yerr[1], list):
        err_top = yerr[1]
    elif yerr is not None:
        err_top = [float(v) for v in yerr if not isinstance(v, list)]
    annotate_bars(ax, bars=bars, yerr=err_top, fmt=fmt)


def export_panels(
    fig: plt.Figure,
    axes: Any,
    names: List[str],
    results_dir: Path,
    logger: logging.Logger,
    dpi: int = 300,
    show_inline: bool = False,
) -> None:
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        ax_list = list(np.atleast_1d(axes))

        for target_ax, base_name in zip(ax_list, names):
            orig_title = target_ax.get_title()
            target_ax.set_title(re.sub(r"^[A-Z]\.\s+", "", orig_title), fontweight="bold", pad=10)
            for ax in ax_list:
                ax.set_visible(ax is target_ax)

            full_base = results_dir / base_name
            fig.savefig(f"{full_base}.png", dpi=dpi, bbox_inches="tight")
            fig.savefig(f"{full_base}.svg", format="svg", bbox_inches="tight")
            target_ax.set_title(orig_title, fontweight="bold", pad=10)
            logger.info("fig_saved | name=%s | formats=png,svg | path=%s", base_name, results_dir)

        for ax in ax_list:
            ax.set_visible(True)

        if show_inline:
            plt.show()
    finally:
        plt.close(fig)
