from typing import Any, Dict

from qwen3_eval.utils.lazy import resolve_lazy_export
from qwen3_eval.visualization.display import (
    build_multiline_html_table,
    configure_notebook_display,
    display_multiline_table,
)

_LAZY_EXPORTS: Dict[str, str] = {
    "configure_matplotlib_theme": "qwen3_eval.visualization.common",
    "plot_dataset_profile": "qwen3_eval.visualization.dataset_plots",
    "plot_accuracy_and_slots": "qwen3_eval.visualization.eval_plots",
    "plot_parsing_and_domains": "qwen3_eval.visualization.eval_plots",
    "plot_positional_and_kappa": "qwen3_eval.visualization.eval_plots",
    "generate_evaluation_plots": "qwen3_eval.visualization.plotter",
    "plot_positional_heatmaps": "qwen3_eval.visualization.stability_plots",
    "plot_prompt_stability": "qwen3_eval.visualization.stability_plots",
}


def __getattr__(name: str) -> Any:
    return resolve_lazy_export(__name__, name, _LAZY_EXPORTS)


__all__ = [
    "build_multiline_html_table",
    "configure_matplotlib_theme",
    "configure_notebook_display",
    "display_multiline_table",
    "generate_evaluation_plots",
    "plot_accuracy_and_slots",
    "plot_dataset_profile",
    "plot_parsing_and_domains",
    "plot_positional_and_kappa",
    "plot_positional_heatmaps",
    "plot_prompt_stability",
]
