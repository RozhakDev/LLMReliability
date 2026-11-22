from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

from llm_exp.config import ExperimentConfig, ExperimentPaths, ModelSpec, load_model_registry
from llm_exp.metrics.reliability import evaluate_single_model


@dataclass
class CrossModelComparisonTables:
    summary_df: pd.DataFrame
    prompt_breakdown_df: pd.DataFrame
    domain_matrix_df: pd.DataFrame
    prompt_stability_df: pd.DataFrame
    comparison_dir: Path


def _parse_mcnemar_p(kpi_path: Path) -> float:
    if not kpi_path.exists():
        return float("nan")
    try:
        df_kpi = pd.read_csv(kpi_path)
        rows = df_kpi[df_kpi["Metric Category"] == "Positional Bias Test"]
        match = re.search(r"p\s*=\s*([0-9eE.+\-]+)", str(rows["Value"].iloc[0])) if not rows.empty else None
        return float(match.group(1)) if match else float("nan")
    except Exception:
        return float("nan")


def _load_model_entry(
    spec: ModelSpec,
    paths: ExperimentPaths,
    cfg: ExperimentConfig,
    df_eval: Optional[pd.DataFrame],
    expected_total: Optional[int],
    recompute: bool,
    logger: logging.Logger,
) -> Optional[Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    art = cfg.artifacts
    model_dir = paths.model_results_dir(spec.key)
    jsonl_path, metrics_csv = model_dir / art.output_jsonl, model_dir / art.metrics_summary_csv
    if not jsonl_path.exists() and not metrics_csv.exists():
        logger.info("agg_skip | model=%s | reason=no_output | path=%s", spec.key, model_dir)
        return None

    from_jsonl = recompute and jsonl_path.exists()
    if from_jsonl:
        res = evaluate_single_model(spec=spec, paths=paths, logger=logger, df_eval=df_eval, config=cfg)
        df_m, df_stab, df_dom = res.df_metrics_table.copy(), res.df_prompt_stability_summary.copy(), res.df_domain_accuracy.copy()
        total_eval = len(res.df_valid) + res.invalid_count
        valid_rate = (len(res.df_valid) / total_eval * 100.0) if total_eval > 0 else 0.0
        pos_cons, pos_flip, mc_p, first_pos, cons_3 = (
            res.position_consistency, res.position_flip, res.mcnemar_pvalue, res.first_pos_rate, res.prompt_consistency_3way
        )
    else:
        df_m = pd.read_csv(metrics_csv)
        df_stab = pd.read_csv(model_dir / art.prompt_stability_csv) if (model_dir / art.prompt_stability_csv).exists() else pd.DataFrame()
        df_dom = pd.read_csv(model_dir / art.domain_accuracy_csv) if (model_dir / art.domain_accuracy_csv).exists() else pd.DataFrame()
        det_path = model_dir / art.detailed_results_parquet
        total_eval, valid_rate, first_pos = 0, float("nan"), float("nan")
        if det_path.exists():
            df_det = pd.read_parquet(det_path)
            total_eval = len(df_det)
            if not df_det.empty and "parsed_decision" in df_det.columns:
                first_pos = float((df_det["parsed_decision"] == "A").mean() * 100.0)
            if expected_total:
                valid_rate = min((len(df_det) / expected_total) * 100.0, 100.0)
        pos_cons = float(df_m["Consistency (%)"].mean())
        pos_flip = 100.0 - pos_cons
        mc_p = _parse_mcnemar_p(model_dir / art.executive_summary_csv)
        r3 = df_stab[df_stab["Comparison"].str.contains("3-Way", na=False)] if not df_stab.empty else pd.DataFrame()
        cons_3 = float(r3["Consistency (%)"].iloc[0]) if not r3.empty else float("nan")

    if from_jsonl and expected_total is not None and total_eval < expected_total:
        logger.warning("agg_incomplete | model=%s | evals=%d/%d", spec.key, total_eval, expected_total)

    p1 = cfg.evaluation.prompts[0]
    p1_row = df_m[df_m["Prompt"] == p1].iloc[0] if not df_m[df_m["Prompt"] == p1].empty else None
    summary = {
        "Model Key": spec.key, "Model": spec.display_name, "HF Repo": spec.hf_id, "Is Main": spec.is_primary,
        "Primary Strength": spec.primary_strength, "Precision": "BF16", "Evaluations (N)": int(total_eval),
        "Valid Rate (%)": round(valid_rate, 2) if pd.notna(valid_rate) else float("nan"),
        "Mean Accuracy (%)": round(float(df_m["Accuracy (%)"].mean()), 2),
        "P1 Accuracy (%)": float(p1_row["Accuracy (%)"]) if p1_row is not None else float("nan"),
        "Mean Cohen Kappa": round(float(df_m["Cohen Kappa"].mean()), 3),
        "P1 Cohen Kappa": float(p1_row["Cohen Kappa"]) if p1_row is not None else float("nan"),
        "Overall Consistency (%)": round(pos_cons, 2), "Overall Flip Rate (%)": round(pos_flip, 2),
        "First Slot Rate (%)": round(first_pos, 2) if pd.notna(first_pos) else float("nan"),
        "McNemar p-value": mc_p,
        "3-Way Prompt Consistency (%)": round(cons_3, 2) if pd.notna(cons_3) else float("nan"),
    }
    return summary, df_m, df_stab, df_dom


def aggregate_all_models(
    paths: ExperimentPaths,
    logger: logging.Logger,
    config: Optional[ExperimentConfig] = None,
    recompute_from_jsonl: bool = True,
) -> CrossModelComparisonTables:
    cfg = config or paths.config
    art, ev = cfg.artifacts, cfg.evaluation
    df_eval = pd.read_parquet(paths.eval_parquet_path) if paths.eval_parquet_path.exists() else None
    expected = len(df_eval) * len(ev.prompts) * len(ev.positions) if df_eval is not None else None

    summary_rows: List[Dict[str, object]] = []
    prompt_rows, stab_rows, domain_frames = [], [], {}
    for key, spec in load_model_registry(cfg.models_config_path).items():
        loaded = _load_model_entry(spec, paths, cfg, df_eval, expected, recompute_from_jsonl, logger)
        if loaded is None:
            continue
        summary, df_m, df_stab, df_dom = loaded
        df_m["Model Key"], df_m["Model"] = key, spec.display_name
        prompt_rows.append(df_m)
        if not df_stab.empty:
            df_stab["Model Key"], df_stab["Model"] = key, spec.display_name
            stab_rows.append(df_stab)
        if not df_dom.empty:
            domain_frames[spec.display_name] = df_dom.set_index("domain")["is_correct"]
        summary_rows.append(summary)

    if not summary_rows:
        raise RuntimeError(f"No completed model results found under '{paths.results_root_dir}'.")

    comp_dir = paths.comparison_dir
    comp_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    prompt_df = pd.concat(prompt_rows, ignore_index=True)
    stab_df = pd.concat(stab_rows, ignore_index=True) if stab_rows else pd.DataFrame()
    dom_df = pd.DataFrame(domain_frames).round(2) if domain_frames else pd.DataFrame()

    summary_df.to_csv(comp_dir / art.comparison_summary_csv, index=False)
    prompt_df.to_csv(comp_dir / art.comparison_prompt_breakdown_csv, index=False)
    if not stab_df.empty:
        stab_df.to_csv(comp_dir / art.comparison_prompt_stability_csv, index=False)
    if not dom_df.empty:
        dom_df.to_csv(comp_dir / art.comparison_domain_accuracy_csv, index=True)
    logger.info("agg_done | models=%d | path=%s", len(summary_df), comp_dir)
    return CrossModelComparisonTables(summary_df, prompt_df, dom_df, stab_df, comp_dir)
