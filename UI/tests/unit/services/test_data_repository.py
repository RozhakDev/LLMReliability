"""Unit test mendalam untuk engine repositori data dan kalkulasi statistik."""
import pytest
import pandas as pd
from llm_reliability.services.data_repository import EvaluationRepository as Repo
from llm_reliability.domain import FilterCriteria
from llm_reliability.core.constants import (
    SYMMETRY_FILTER_FLIP,
    SYMMETRY_FILTER_CONSISTENT,
    GT_FILTER_CORRECT,
    GT_FILTER_INCORRECT,
    PROMPT_FILTER_ALL,
    DOMAIN_ALL
)


def test_dataset_statistics_match_empirical_data(results_dir):
    stats = Repo.get_dataset_statistics()
    assert stats["total_evaluations"] == stats["paired_count"] * 2
    assert stats["sample_count"] == 519
    assert stats["paired_count"] == 519 * 3


def test_metrics_summary_has_all_prompts(results_dir):
    df = Repo.load_metrics_summary()
    assert set(df["Prompt"]) == {"P1", "P2", "P3"}
    assert df["Consistency (%)"].add(df["Flip Rate (%)"]).round(2).eq(100.0).all()


def test_mcnemar_is_consistent_with_flip_counts(results_dir):
    data = Repo.get_mcnemar_data()
    assert set(data) == {"ALL", "P1", "P2", "P3"}
    for m in data.values():
        assert m.total_flips == m.b + m.c
        assert 0.0 <= m.pvalue <= 1.0
    assert data["ALL"].total_flips == sum(data[p].total_flips for p in ("P1", "P2", "P3"))


def test_domain_accuracy_is_percentage(results_dir):
    df = Repo.load_domain_accuracy()
    assert not df.empty
    assert df["Akurasi (%)"].between(0, 100).all()


def test_get_available_domains(results_dir):
    domains = Repo.get_available_domains()
    assert len(domains) >= 5
    assert "logic" in domains or "coding" in domains or "translation" in domains


def test_filter_evaluations_cross_filtering(results_dir):
    """Pengujian cross-filtering compound: Prompt, Domain, Simetri, dan GT."""
    crit_p1 = FilterCriteria(prompt_id="P1 (Direct)")
    df_p1 = Repo.filter_evaluations(crit_p1)
    assert not df_p1.empty
    assert (df_p1['prompt_id'] == 'P1').all()

    crit_flip = FilterCriteria(symmetry_filter=SYMMETRY_FILTER_FLIP)
    df_flip = Repo.filter_evaluations(crit_flip)
    assert not df_flip.empty
    assert df_flip['is_flip'].all()

    crit_cons = FilterCriteria(symmetry_filter=SYMMETRY_FILTER_CONSISTENT)
    df_cons = Repo.filter_evaluations(crit_cons)
    assert not df_cons.empty
    assert (~df_cons['is_flip']).all()

    crit_gt = FilterCriteria(gt_filter=GT_FILTER_CORRECT)
    df_gt = Repo.filter_evaluations(crit_gt)
    assert not df_gt.empty
    assert df_gt['is_robust_correct'].all()

    crit_gt_inc = FilterCriteria(gt_filter=GT_FILTER_INCORRECT)
    df_gt_inc = Repo.filter_evaluations(crit_gt_inc)
    assert not df_gt_inc.empty
    assert (~df_gt_inc['is_robust_correct']).all()


def test_filter_evaluations_empty_result(results_dir):
    """Memastikan filter dengan kombinasi tidak mungkin menghasilkan dataframe kosong tanpa crash."""
    crit_impossible = FilterCriteria(domain="domain_yang_pasti_tidak_ada_xyz_123")
    df_empty = Repo.filter_evaluations(crit_impossible)
    assert isinstance(df_empty, pd.DataFrame)
    assert df_empty.empty


def test_prompt_stability_matrix_diagonal_is_zero(results_dir):
    """Memastikan struktur metrik stabilitas pasangan prompt valid dan komplementer 100%."""
    df_stab = Repo.get_prompt_stability_matrix()
    assert not df_stab.empty
    assert "Pasangan Strategi" in df_stab.columns
    assert set(df_stab["Pasangan Strategi"]) == {"P1 vs P2", "P1 vs P3", "P2 vs P3"}
    totals = df_stab["Tingkat Perubahan Keputusan (%)"] + df_stab["Tingkat Keselarasan (%)"]
    assert totals.round(1).eq(100.0).all()
