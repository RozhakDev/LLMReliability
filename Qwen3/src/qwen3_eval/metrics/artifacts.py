from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class EvaluationArtifacts:
    model_key: str
    df_all: pd.DataFrame
    df_valid: pd.DataFrame
    invalid_count: int
    df_metrics_table: pd.DataFrame
    df_prompt_stability_summary: pd.DataFrame
    df_executive_summary: pd.DataFrame
    df_domain_accuracy: pd.DataFrame
    df_prompt_stab: pd.DataFrame
    position_consistency: float
    position_flip: float
    first_pos_rate: float
    mcnemar_stat: float
    mcnemar_pvalue: float
    is_significant: bool
    prompt_consistency_3way: float
    prompt_flip_3way: float
    results_dir: Path
