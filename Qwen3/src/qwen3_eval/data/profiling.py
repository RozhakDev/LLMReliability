from typing import Dict, List, Optional, Tuple

import pandas as pd

from qwen3_eval.config import ExperimentConfig, ExperimentPaths
from qwen3_eval.data.labels import resolve_human_winner


def _resolve_winner_series(df_eval: pd.DataFrame) -> pd.Series:
    if df_eval.empty:
        return pd.Series(dtype=str)
    return df_eval.apply(resolve_human_winner, axis=1)


def build_preprocessing_summary(
    df_eval: pd.DataFrame,
    paths: ExperimentPaths,
    config: Optional[ExperimentConfig] = None,
) -> pd.DataFrame:
    cfg = config or paths.config
    req_cols = list(cfg.dataset.required_columns)
    txt_cols = [c for c in cfg.dataset.text_columns if c in df_eval.columns]

    raw_rows: Optional[int] = None
    if paths.raw_parquet_path.is_file():
        try:
            raw_rows = len(pd.read_parquet(paths.raw_parquet_path, columns=["original_id"]))
        except (OSError, ValueError):
            raw_rows = None

    eval_rows = len(df_eval)
    removed_dups = (raw_rows - eval_rows) if (raw_rows is not None and raw_rows >= eval_rows) else None
    total_tasks = eval_rows * len(cfg.evaluation.prompts) * len(cfg.evaluation.positions)
    present_req = [c for c in req_cols if c in df_eval.columns]
    null_count = int(df_eval[present_req].isnull().sum().sum()) if present_req else 0
    empty_text_count = int(sum((df_eval[col].astype(str).str.strip() == "").sum() for col in txt_cols))
    mem_kb = float(df_eval.memory_usage(deep=True).sum() / 1024.0)

    sides = _resolve_winner_series(df_eval)
    count_a = int((sides == "A").sum())
    count_b = int((sides == "B").sum())
    pct_a = (count_a / eval_rows * 100.0) if eval_rows > 0 else 0.0
    pct_b = (count_b / eval_rows * 100.0) if eval_rows > 0 else 0.0

    rows: List[Dict[str, str]] = []
    if raw_rows is not None and removed_dups is not None:
        rows.append({"Stage / Check": "Bronze Raw Records", "Value": f"{raw_rows:,} rows"})
        rows.append({"Stage / Check": "Duplicate original_id Removed", "Value": f"{removed_dups:,} rows"})
    rows.extend(
        [
            {"Stage / Check": "Silver Unique Prompts (Cohort)", "Value": f"{eval_rows:,} samples"},
            {
                "Stage / Check": "Evaluation Task Matrix",
                "Value": f"{total_tasks:,} inferences ({len(cfg.evaluation.prompts)} prompts x {len(cfg.evaluation.positions)} positions)",
            },
            {"Stage / Check": "Schema Integrity", "Value": f"{len(present_req)}/{len(req_cols)} required columns | nulls={null_count} | empty_text={empty_text_count}"},
            {"Stage / Check": "Human Preference Balance", "Value": f"Assistant A: {count_a} ({pct_a:.2f}%) | Assistant B: {count_b} ({pct_b:.2f}%)"},
            {"Stage / Check": "DataFrame Memory Footprint", "Value": f"{mem_kb:.1f} KB"},
        ]
    )
    return pd.DataFrame(rows)


def build_domain_distribution(df_eval: pd.DataFrame) -> pd.DataFrame:
    if df_eval.empty or "domain" not in df_eval.columns:
        return pd.DataFrame(columns=["Domain", "Jumlah Sampel", "Persentase (%)", "Assistant A (%)", "Assistant B (%)"])

    total = len(df_eval)
    working = df_eval.assign(_side=_resolve_winner_series(df_eval))
    grouped = (
        working.groupby("domain", as_index=False)
        .agg(
            samples=("original_id", "count"),
            a_rate=("_side", lambda s: float((s == "A").mean() * 100.0)),
            b_rate=("_side", lambda s: float((s == "B").mean() * 100.0)),
        )
        .sort_values(["samples", "domain"], ascending=[False, True])
        .reset_index(drop=True)
    )
    grouped["pct"] = (grouped["samples"] / total * 100.0).round(2)
    grouped["a_rate"] = grouped["a_rate"].round(2)
    grouped["b_rate"] = grouped["b_rate"].round(2)
    return grouped.rename(
        columns={
            "domain": "Domain",
            "samples": "Jumlah Sampel",
            "pct": "Persentase (%)",
            "a_rate": "Assistant A (%)",
            "b_rate": "Assistant B (%)",
        }
    )[["Domain", "Jumlah Sampel", "Persentase (%)", "Assistant A (%)", "Assistant B (%)"]]


def build_sample_overview_table(
    df_eval: pd.DataFrame,
    n_rows: int = 3,
    max_chars: int = 110,
) -> pd.DataFrame:
    cols = [c for c in ("original_id", "domain", "input", "response_1", "response_2", "actual_score") if c in df_eval.columns]
    preview = df_eval[cols].head(max(n_rows, 1)).copy()
    for text_col in ("input", "response_1", "response_2"):
        if text_col in preview.columns:
            preview[text_col] = preview[text_col].astype(str).map(
                lambda s: s.replace("\n", " ")[:max_chars] + ("..." if len(s) > max_chars else "")
            )
    return preview.reset_index(drop=True)


def build_sample_detail_table(df_eval: pd.DataFrame, row_index: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df_eval.empty:
        raise ValueError("df_eval must not be empty.")
    safe_idx = min(max(row_index, 0), len(df_eval) - 1)
    row = df_eval.iloc[safe_idx]
    input_txt = str(row["input"])
    resp1_txt = str(row["response_1"])
    resp2_txt = str(row["response_2"])
    raw_score = str(row["actual_score"])
    winner = resolve_human_winner(row)
    winner_desc = "Jawaban Pertama (response_1)" if winner == "A" else "Jawaban Kedua (response_2)"

    attr_df = pd.DataFrame(
        [
            {"Atribut": "original_id", "Deskripsi": "Kode pengenal instruksi asal", "Nilai": str(row["original_id"])},
            {"Atribut": "domain", "Deskripsi": "Kategori domain topik", "Nilai": str(row["domain"])},
            {"Atribut": "actual_score", "Deskripsi": "Label acuan preferensi manusia", "Nilai": f"{raw_score} | {winner} ({winner_desc})"},
            {"Atribut": "input", "Deskripsi": "Panjang teks instruksi masukan", "Nilai": f"{len(input_txt):,} karakter"},
            {"Atribut": "response_1", "Deskripsi": "Panjang teks jawaban pertama", "Nilai": f"{len(resp1_txt):,} karakter"},
            {"Atribut": "response_2", "Deskripsi": "Panjang teks jawaban kedua", "Nilai": f"{len(resp2_txt):,} karakter"},
        ]
    )
    content_df = pd.DataFrame(
        [
            {"Bagian": "INPUT", "Panjang Karakter": len(input_txt), "Teks": input_txt},
            {"Bagian": "RESPONSE_1 (Kandidat Pertama)", "Panjang Karakter": len(resp1_txt), "Teks": resp1_txt},
            {"Bagian": "RESPONSE_2 (Kandidat Kedua)", "Panjang Karakter": len(resp2_txt), "Teks": resp2_txt},
        ]
    )
    return attr_df, content_df
