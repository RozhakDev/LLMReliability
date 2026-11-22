from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from llm_exp.config import ExperimentConfig, ExperimentPaths, ModelSpec
from llm_exp.metrics.parser import load_and_parse_results
from llm_exp.metrics.statistics import (
    compute_positional_metrics,
    compute_prompt_metrics_table,
    compute_prompt_stability,
)


@dataclass
class ModelEvaluationArtifacts:
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


def _build_executive_kpi(
    spec: ModelSpec,
    cohort_size: int,
    valid_count: int,
    pos_cons: float,
    pos_flip: float,
    mcnemar_p: float,
    is_sig: bool,
    alpha: float,
    cons_3way: float,
    flip_3way: float,
    primary_prompt: str,
    base_acc: Optional[float],
    base_kappa: Optional[float],
) -> pd.DataFrame:
    rows: List[Dict[str, str]] = [
        {"Metric Category": "Model Evaluator", "Specification": f"{spec.display_name} ({spec.hf_id})", "Value": "16-bit BF16"},
        {"Metric Category": "Cohort Population", "Specification": "IndoPref Sensus Prompt Unik", "Value": f"{cohort_size} samples ({valid_count:,} evaluations)"},
        {"Metric Category": "Positional Reliability", "Specification": "Consistency (AB == BA)", "Value": f"{pos_cons:.2f}%"},
        {"Metric Category": "Positional Reliability", "Specification": "Preference Flip Rate", "Value": f"{pos_flip:.2f}%"},
        {"Metric Category": "Positional Bias Test", "Specification": f"McNemar Test (alpha={alpha})", "Value": f"p = {mcnemar_p:.2e} ({'Significant' if is_sig else 'Not Significant'})"},
        {"Metric Category": "Prompt Stability", "Specification": "Simultaneous 3-Way Consistency", "Value": f"{cons_3way:.2f}%"},
        {"Metric Category": "Prompt Stability", "Specification": "Simultaneous 3-Way Flip Rate", "Value": f"{flip_3way:.2f}%"},
    ]
    if base_acc is not None and base_kappa is not None:
        rows.append({"Metric Category": "Baseline Performance", "Specification": f"{primary_prompt}_AB Condition", "Value": f"Accuracy = {base_acc:.2f}% | Kappa = {base_kappa:.3f}"})
    return pd.DataFrame(rows)


def evaluate_single_model(
    spec: ModelSpec,
    paths: ExperimentPaths,
    logger: logging.Logger,
    df_eval: Optional[pd.DataFrame] = None,
    config: Optional[ExperimentConfig] = None,
) -> ModelEvaluationArtifacts:
    cfg = config or paths.config
    ev, art = cfg.evaluation, cfg.artifacts
    results_dir = paths.model_results_dir(spec.key)
    results_dir.mkdir(parents=True, exist_ok=True)

    df_all, df_valid, invalid_count = load_and_parse_results(results_dir / art.output_jsonl, logger=logger, df_eval=df_eval)
    df_pos, pos_cons, pos_flip, first_pos, mc_stat, mc_p, is_sig = compute_positional_metrics(df_valid, alpha=ev.alpha)
    logger.info(
        "pos_metrics | model=%s | cons_pct=%.2f | flip_pct=%.2f | slot_a_pct=%.2f | mcnemar_p=%.5e",
        spec.key, pos_cons, pos_flip, first_pos, mc_p,
    )

    df_metrics, base_acc, base_kappa = compute_prompt_metrics_table(
        df_valid, df_pos, prompts=ev.prompts, n_iterations=ev.bootstrap_iterations, alpha=ev.alpha, seed=ev.random_seed
    )
    df_metrics.to_csv(results_dir / art.metrics_summary_csv, index=False)
    df_metrics.to_parquet(results_dir / art.metrics_summary_parquet, index=False)

    df_export = df_valid.copy()
    if df_export.empty:
        df_export["is_correct"] = pd.Series(dtype=int)
        df_dom = pd.DataFrame(columns=["domain", "is_correct"])
    else:
        df_export["is_correct"] = (df_export["logical_decision"] == df_export["logical_human"]).astype(int) * 100
        df_dom = df_export.groupby("domain")["is_correct"].mean().reset_index().sort_values("is_correct", ascending=False)
    df_export.to_parquet(results_dir / art.detailed_results_parquet, index=False)
    df_dom.to_csv(results_dir / art.domain_accuracy_csv, index=False)

    df_stab, df_stab_sum, cons_3, flip_3 = compute_prompt_stability(df_valid, prompts=ev.prompts)
    if not df_stab_sum.empty:
        df_stab_sum.to_csv(results_dir / art.prompt_stability_csv, index=False)

    cohort_size = len(df_eval) if df_eval is not None else int(df_all["sample_id"].nunique())
    df_kpi = _build_executive_kpi(
        spec, cohort_size, len(df_valid), pos_cons, pos_flip, mc_p, is_sig, ev.alpha, cons_3, flip_3, ev.prompts[0], base_acc, base_kappa
    )
    df_kpi.to_csv(results_dir / art.executive_summary_csv, index=False)
    logger.info("eval_tables_saved | model=%s | path=%s", spec.key, results_dir)

    return ModelEvaluationArtifacts(
        model_key=spec.key, df_all=df_all, df_valid=df_export, invalid_count=invalid_count,
        df_metrics_table=df_metrics, df_prompt_stability_summary=df_stab_sum, df_executive_summary=df_kpi,
        df_domain_accuracy=df_dom, df_prompt_stab=df_stab, position_consistency=pos_cons, position_flip=pos_flip,
        first_pos_rate=first_pos, mcnemar_stat=mc_stat, mcnemar_pvalue=mc_p, is_significant=is_sig,
        prompt_consistency_3way=cons_3, prompt_flip_3way=flip_3, results_dir=results_dir,
    )
