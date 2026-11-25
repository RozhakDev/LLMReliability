from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PositionalContingency:
    n_pairs: int
    consistent_1: int
    consistent_2: int
    slot_a_bias_b: int
    slot_b_bias_c: int


def build_paired_positional_frame(df_valid: pd.DataFrame) -> pd.DataFrame:
    if df_valid.empty:
        return pd.DataFrame(columns=["AB", "BA"])
    return (
        df_valid.pivot_table(
            index=["sample_id", "prompt_id"],
            columns="position",
            values="logical_decision",
            aggfunc="first",
        )
        .reindex(columns=["AB", "BA"])
        .dropna()
    )


def summarize_positional_contingency(df_pos: pd.DataFrame) -> PositionalContingency:
    if df_pos.empty:
        return PositionalContingency(
            n_pairs=0,
            consistent_1=0,
            consistent_2=0,
            slot_a_bias_b=0,
            slot_b_bias_c=0,
        )
    return PositionalContingency(
        n_pairs=len(df_pos),
        consistent_1=int(((df_pos["AB"] == 1) & (df_pos["BA"] == 1)).sum()),
        consistent_2=int(((df_pos["AB"] == 2) & (df_pos["BA"] == 2)).sum()),
        slot_a_bias_b=int(((df_pos["AB"] == 1) & (df_pos["BA"] == 2)).sum()),
        slot_b_bias_c=int(((df_pos["AB"] == 2) & (df_pos["BA"] == 1)).sum()),
    )
