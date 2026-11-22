import logging
from typing import Optional

from llm_exp.config import ExperimentConfig, load_experiment_config
from llm_exp.metrics.reliability import ModelEvaluationArtifacts
from llm_exp.visualization.common import configure_matplotlib_theme
from llm_exp.visualization.panels import (
    plot_accuracy_and_slots,
    plot_parsing_and_domains,
    plot_positional_and_kappa,
    plot_positional_heatmaps,
    plot_prompt_stability,
)


def generate_single_model_plots(
    artifacts: ModelEvaluationArtifacts,
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
