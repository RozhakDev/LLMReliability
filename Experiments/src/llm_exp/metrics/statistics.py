from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.contingency_tables import mcnemar

from llm_exp.config.settings import load_experiment_config


def bootstrap_kappa(
    y_true: List[int],
    y_pred: List[int],
    n_iterations: Optional[int] = None,
    alpha: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    if n_iterations is None or alpha is None or seed is None:
        ev = load_experiment_config().evaluation
        n_iterations = n_iterations if n_iterations is not None else ev.bootstrap_iterations
        alpha = alpha if alpha is not None else ev.alpha
        seed = seed if seed is not None else ev.random_seed

    rng = np.random.default_rng(seed)
    yt, yp = np.array(y_true), np.array(y_pred)
    if len(yt) == 0:
        return 0.0, 0.0

    stats = []
    for _ in range(n_iterations):
        idx = rng.integers(0, len(yt), len(yt))
        score = float(cohen_kappa_score(yt[idx], yp[idx]))
        if not np.isnan(score):
            stats.append(score)

    if not stats:
        return 0.0, 0.0
    return float(np.percentile(stats, (alpha / 2) * 100)), float(np.percentile(stats, (1 - alpha / 2) * 100))


def compute_positional_metrics(
    df_valid: pd.DataFrame,
    alpha: Optional[float] = None,
) -> Tuple[pd.DataFrame, float, float, float, float, float, bool]:
    resolved_alpha = alpha if alpha is not None else load_experiment_config().evaluation.alpha
    if df_valid.empty:
        df_pos = pd.DataFrame(columns=["AB", "BA"])
    else:
        df_pos = (
            df_valid.pivot_table(index=["sample_id", "prompt_id"], columns="position", values="logical_decision", aggfunc="first")
            .reindex(columns=["AB", "BA"])
            .dropna()
        )

    pos_cons = float((df_pos["AB"] == df_pos["BA"]).mean() * 100) if not df_pos.empty else 0.0
    first_pos_rate = float((df_valid["parsed_decision"] == "A").mean() * 100) if not df_valid.empty else 0.0
    b = int(sum((df_pos["AB"] == 1) & (df_pos["BA"] == 2))) if not df_pos.empty else 0
    c = int(sum((df_pos["AB"] == 2) & (df_pos["BA"] == 1))) if not df_pos.empty else 0

    if b + c == 0:
        stat, pval, sig = 0.0, 1.0, False
    else:
        res = mcnemar([[0, b], [c, 0]], exact=False, correction=True)
        stat, pval, sig = float(res.statistic), float(res.pvalue), bool(res.pvalue < resolved_alpha)
    return df_pos, pos_cons, 100.0 - pos_cons, first_pos_rate, stat, pval, sig


def compute_prompt_metrics_table(
    df_valid: pd.DataFrame,
    df_pos: pd.DataFrame,
    prompts: Optional[Sequence[str]] = None,
    n_iterations: Optional[int] = None,
    alpha: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, Optional[float], Optional[float]]:
    ev = load_experiment_config().evaluation
    p_list = tuple(prompts) if prompts is not None else ev.prompts
    iters = n_iterations if n_iterations is not None else ev.bootstrap_iterations
    resolved_alpha = alpha if alpha is not None else ev.alpha
    resolved_seed = seed if seed is not None else ev.random_seed

    rows: List[Dict[str, object]] = []
    for p_id in p_list:
        sub = df_valid[df_valid["prompt_id"] == p_id]
        if sub.empty:
            rows.append({"Prompt": p_id, "Accuracy (%)": 0.0, "Consistency (%)": 0.0, "Flip Rate (%)": 100.0, "Cohen Kappa": 0.0, "Kappa 95% CI": "[0.000, 0.000]"})
            continue

        acc = float((sub["logical_decision"] == sub["logical_human"]).mean() * 100)
        sub_pos = df_pos.xs(p_id, level="prompt_id") if (not df_pos.empty and p_id in df_pos.index.levels[1]) else pd.DataFrame()
        cons = float((sub_pos["AB"] == sub_pos["BA"]).mean() * 100) if not sub_pos.empty else 0.0

        y_dec, y_hum = [int(v) for v in sub["logical_decision"]], [int(v) for v in sub["logical_human"]]
        kappa = float(cohen_kappa_score(y_dec, y_hum))
        kappa = 0.0 if np.isnan(kappa) else kappa
        ci_low, ci_high = bootstrap_kappa(y_hum, y_dec, n_iterations=iters, alpha=resolved_alpha, seed=resolved_seed)

        rows.append(
            {
                "Prompt": p_id,
                "Accuracy (%)": round(acc, 2),
                "Consistency (%)": round(cons, 2),
                "Flip Rate (%)": round(100.0 - cons, 2),
                "Cohen Kappa": round(kappa, 3),
                "Kappa 95% CI": f"[{ci_low:.3f}, {ci_high:.3f}]",
            }
        )

    base_acc, base_kappa = None, None
    sub_base = df_valid[(df_valid["prompt_id"] == p_list[0]) & (df_valid["position"] == "AB")]
    if not sub_base.empty:
        base_acc = float((sub_base["logical_decision"] == sub_base["logical_human"]).mean() * 100)
        raw_k = float(cohen_kappa_score([int(v) for v in sub_base["logical_decision"]], [int(v) for v in sub_base["logical_human"]]))
        base_kappa = 0.0 if np.isnan(raw_k) else raw_k
    return pd.DataFrame(rows), base_acc, base_kappa


def compute_prompt_stability(
    df_valid: pd.DataFrame,
    prompts: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, float, float]:
    p_list = tuple(prompts) if prompts is not None else load_experiment_config().evaluation.prompts
    df_stab = (
        df_valid[df_valid["position"] == "AB"]
        .pivot_table(index="sample_id", columns="prompt_id", values="logical_decision", aggfunc="first")
        .dropna()
    )
    if df_stab.empty or len(p_list) < 3 or not set(p_list[:3]).issubset(set(df_stab.columns)):
        return df_stab, pd.DataFrame(), 0.0, 0.0

    p1, p2, p3 = p_list[:3]
    cons_3 = float(((df_stab[p1] == df_stab[p2]) & (df_stab[p2] == df_stab[p3])).mean() * 100)
    flip_3 = 100.0 - cons_3
    flips = [(f"{a} vs {b} (Pairwise)", float((df_stab[a] != df_stab[b]).mean() * 100)) for a, b in ((p1, p2), (p1, p3), (p2, p3))]
    records = [{"Comparison": f"{p1}={p2}={p3} (Simultaneous 3-Way)", "Consistency (%)": round(cons_3, 2), "Flip Rate (%)": round(flip_3, 2)}]
    records.extend({"Comparison": lbl, "Consistency (%)": round(100.0 - flp, 2), "Flip Rate (%)": round(flp, 2)} for lbl, flp in flips)
    return df_stab, pd.DataFrame(records), cons_3, flip_3
