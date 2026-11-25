from typing import Dict, List, Protocol

import pandas as pd

from qwen3_eval.metrics.contingency import (
    build_paired_positional_frame,
    summarize_positional_contingency,
)


class ArtifactsProtocol(Protocol):
    @property
    def df_all(self) -> pd.DataFrame: ...

    @property
    def df_valid(self) -> pd.DataFrame: ...

    @property
    def invalid_count(self) -> int: ...

    @property
    def position_consistency(self) -> float: ...

    @property
    def position_flip(self) -> float: ...

    @property
    def first_pos_rate(self) -> float: ...

    @property
    def mcnemar_stat(self) -> float: ...

    @property
    def mcnemar_pvalue(self) -> float: ...

    @property
    def is_significant(self) -> bool: ...


def build_parsed_predictions_preview(
    artifacts: ArtifactsProtocol,
    n_rows: int = 6,
) -> pd.DataFrame:
    source_df = artifacts.df_valid if not artifacts.df_valid.empty else artifacts.df_all
    if source_df.empty:
        return pd.DataFrame()

    ordered = source_df.sort_values(["sample_id", "prompt_id", "position"]).head(max(n_rows, 1)).copy()
    ordered["mapped_choice"] = ordered["logical_decision"].map(
        lambda v: f"Resp. {v}" if v in (1, 2) else str(v)
    )
    ordered["mapped_human"] = ordered["logical_human"].map(
        lambda v: f"Resp. {v}" if v in (1, 2) else str(v)
    )
    preferred_cols = [
        c
        for c in (
            "sample_id",
            "domain",
            "prompt_id",
            "position",
            "raw_output",
            "parsed_decision",
            "mapped_choice",
            "human_preference",
            "mapped_human",
            "is_correct",
            "output_tokens",
            "runtime_sec",
            "tps",
        )
        if c in ordered.columns
    ]
    return ordered[preferred_cols].reset_index(drop=True)


def build_positional_bias_diagnostics(
    artifacts: ArtifactsProtocol,
    alpha: float = 0.05,
) -> pd.DataFrame:
    total_all = len(artifacts.df_all)
    valid_n = len(artifacts.df_valid)
    invalid_pct = (artifacts.invalid_count / total_all * 100.0) if total_all > 0 else 0.0
    contingency = summarize_positional_contingency(build_paired_positional_frame(artifacts.df_valid))

    conclusion = (
        "Significant Position Bias (p < alpha)"
        if artifacts.is_significant
        else "No Significant Bias (p >= alpha)"
    )

    rows: List[Dict[str, str]] = [
        {
            "Diagnostic Metric": "Format Parsing Validity",
            "Value": f"{valid_n:,} valid / {total_all:,} total (invalid={artifacts.invalid_count}, {invalid_pct:.2f}%)",
        },
        {
            "Diagnostic Metric": "Paired AB-BA Comparisons",
            "Value": f"{contingency.n_pairs:,} pairs (Consistent Resp. 1={contingency.consistent_1:,}, Consistent Resp. 2={contingency.consistent_2:,})",
        },
        {
            "Diagnostic Metric": "Discordant Pairs (Position Flip)",
            "Value": f"Slot-A Bias b (AB=1, BA=2): {contingency.slot_a_bias_b:,} | Slot-B Bias c (AB=2, BA=1): {contingency.slot_b_bias_c:,}",
        },
        {
            "Diagnostic Metric": "First-Position Preference Rate",
            "Value": f"{artifacts.first_pos_rate:.2f}% memilih kandidat A (unbiased = 50.00%)",
        },
        {
            "Diagnostic Metric": "Position Consistency and Flip Rate",
            "Value": f"Consistency = {artifacts.position_consistency:.2f}% | Flip Rate = {artifacts.position_flip:.2f}%",
        },
        {
            "Diagnostic Metric": f"McNemar Continuity-Corrected (alpha={alpha})",
            "Value": f"chi2 = {artifacts.mcnemar_stat:.4f} | p = {artifacts.mcnemar_pvalue:.5e} ({conclusion})",
        },
    ]

    if not artifacts.df_all.empty and {"runtime_sec", "tps"}.issubset(artifacts.df_all.columns):
        mean_rt = float(pd.to_numeric(artifacts.df_all["runtime_sec"], errors="coerce").mean())
        total_min = float(pd.to_numeric(artifacts.df_all["runtime_sec"], errors="coerce").sum() / 60.0)
        mean_tps = float(pd.to_numeric(artifacts.df_all["tps"], errors="coerce").mean())
        rows.append(
            {
                "Diagnostic Metric": "Inference Runtime Profile",
                "Value": f"Total = {total_min:.1f} min | Mean Latency = {mean_rt:.2f}s | Mean TPS = {mean_tps:.2f} tok/s",
            }
        )

    return pd.DataFrame(rows)
