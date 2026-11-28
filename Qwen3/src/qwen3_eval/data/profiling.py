from typing import Dict, List, Optional, Tuple

import pandas as pd

from qwen3_eval.config import DatasetConfig, ExperimentConfig, ExperimentPaths, load_experiment_config
from qwen3_eval.data.labels import resolve_human_winner

_DOMAIN_DIST_COLUMNS = [
    "Domain",
    "Jumlah Sampel",
    "Persentase (%)",
    "Assistant A (%)",
    "Assistant B (%)",
]


def _resolve_winner_series(df_eval: pd.DataFrame) -> pd.Series:
    if df_eval.empty:
        return pd.Series(dtype=str)
    return df_eval.apply(resolve_human_winner, axis=1)


def compute_domain_stats(
    df_eval: pd.DataFrame,
    ascending: bool = False,
) -> pd.DataFrame:
    if df_eval.empty or "domain" not in df_eval.columns:
        return pd.DataFrame(columns=["domain", "count", "share_pct", "a_pct", "b_pct"])

    total = len(df_eval)
    working = df_eval.assign(_side=_resolve_winner_series(df_eval))
    stats = (
        working.groupby("domain", as_index=False)
        .agg(
            count=("_side", "size"),
            a_pct=("_side", lambda s: float((s == "A").mean() * 100.0)),
            b_pct=("_side", lambda s: float((s == "B").mean() * 100.0)),
        )
        .sort_values(["count", "domain"], ascending=[ascending, not ascending])
        .reset_index(drop=True)
    )
    stats["share_pct"] = (stats["count"] / total) * 100.0
    return stats[["domain", "count", "share_pct", "a_pct", "b_pct"]]


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
    n_prompts = len(cfg.evaluation.prompts)
    n_positions = len(cfg.evaluation.positions)
    total_tasks = eval_rows * n_prompts * n_positions
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
                "Value": f"{total_tasks:,} inferences ({n_prompts} prompts x {n_positions} positions)",
            },
            {
                "Stage / Check": "Schema Integrity",
                "Value": (
                    f"{len(present_req)}/{len(req_cols)} required columns"
                    f" | nulls={null_count} | empty_text={empty_text_count}"
                ),
            },
            {
                "Stage / Check": "Human Preference Balance",
                "Value": f"Assistant A: {count_a} ({pct_a:.2f}%) | Assistant B: {count_b} ({pct_b:.2f}%)",
            },
            {"Stage / Check": "DataFrame Memory Footprint", "Value": f"{mem_kb:.1f} KB"},
        ]
    )
    return pd.DataFrame(rows)


def build_domain_distribution(df_eval: pd.DataFrame) -> pd.DataFrame:
    stats = compute_domain_stats(df_eval, ascending=False)
    if stats.empty:
        return pd.DataFrame(columns=_DOMAIN_DIST_COLUMNS)

    formatted = stats.copy()
    formatted["share_pct"] = formatted["share_pct"].round(2)
    formatted["a_pct"] = formatted["a_pct"].round(2)
    formatted["b_pct"] = formatted["b_pct"].round(2)
    return formatted.rename(
        columns={
            "domain": "Domain",
            "count": "Jumlah Sampel",
            "share_pct": "Persentase (%)",
            "a_pct": "Assistant A (%)",
            "b_pct": "Assistant B (%)",
        }
    )[_DOMAIN_DIST_COLUMNS]


def build_sample_overview_table(
    df_eval: pd.DataFrame,
    n_rows: int = 3,
    max_chars: int = 110,
    dataset_config: Optional[DatasetConfig] = None,
) -> pd.DataFrame:
    ds_cfg = dataset_config or load_experiment_config().dataset
    cols = [c for c in ds_cfg.required_columns if c in df_eval.columns]
    preview = df_eval[cols].head(max(n_rows, 1)).copy()
    for text_col in ds_cfg.text_columns:
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
            {
                "Atribut": "actual_score",
                "Deskripsi": "Label acuan preferensi manusia",
                "Nilai": f"{raw_score} | {winner} ({winner_desc})",
            },
            {
                "Atribut": "input",
                "Deskripsi": "Panjang teks instruksi masukan",
                "Nilai": f"{len(input_txt):,} karakter",
            },
            {
                "Atribut": "response_1",
                "Deskripsi": "Panjang teks jawaban pertama",
                "Nilai": f"{len(resp1_txt):,} karakter",
            },
            {
                "Atribut": "response_2",
                "Deskripsi": "Panjang teks jawaban kedua",
                "Nilai": f"{len(resp2_txt):,} karakter",
            },
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
