import logging
from typing import Optional

from qwen3_eval.config import ExperimentConfig, load_experiment_config
from qwen3_eval.metrics.reliability import EvaluationArtifacts
from qwen3_eval.visualization.common import configure_matplotlib_theme
from qwen3_eval.visualization.panels import (
    plot_accuracy_and_slots,
    plot_parsing_and_domains,
    plot_positional_and_kappa,
    plot_positional_heatmaps,
    plot_prompt_stability,
)


def generate_evaluation_plots(
    artifacts: EvaluationArtifacts,
    logger: logging.Logger,
    config: Optional[ExperimentConfig] = None,
    dpi: Optional[int] = None,
    show_inline: bool = False,
) -> None:
    cfg = config or load_experiment_config()
    resolved_dpi = dpi if dpi is not None else cfg.visualization.dpi
    prompts = list(cfg.evaluation.prompts)
    configure_matplotlib_theme(cfg.visualization)

    plot_accuracy_and_slots(artifacts, prompts, logger, resolved_dpi, show_inline)
    plot_positional_and_kappa(artifacts, logger, resolved_dpi, show_inline)
    plot_parsing_and_domains(artifacts, logger, resolved_dpi, show_inline)
    plot_positional_heatmaps(artifacts, prompts, logger, resolved_dpi, show_inline)
    plot_prompt_stability(artifacts, prompts, logger, resolved_dpi, show_inline)
