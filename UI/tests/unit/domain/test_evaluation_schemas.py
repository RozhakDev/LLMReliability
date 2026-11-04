"""Unit test untuk domain dataclasses dan validasi kontrak data."""
import pytest
from llm_reliability.domain import DataPaths, FilterCriteria, DualInferenceResult, McNemarMetrics


def test_datapaths_to_dict():
    paths = DataPaths(
        results_dir="UI/data/results",
        detailed_parquet="UI/data/results/detailed.parquet",
        indopref_parquet="UI/data/processed/indopref.parquet",
        metrics_csv="UI/data/results/metrics.csv",
        plot_accuracy="UI/data/results/plot1.png",
        plot_consistency=None,
        plot_parsing_domain=None,
        plot_heatmaps=None
    )
    d = paths.to_dict()
    assert isinstance(d, dict)
    assert d["results_dir"] == "UI/data/results"
    assert d["detailed_parquet"] == "UI/data/results/detailed.parquet"
    assert d["plot_consistency"] is None


def test_datapaths_immutability():
    paths = DataPaths()
    with pytest.raises(Exception):
        paths.results_dir = "new/path"


def test_filter_criteria_defaults_and_custom():
    default_crit = FilterCriteria()
    assert default_crit.prompt_id == "Semua Prompt"
    assert default_crit.domain == "Semua Domain"
    assert default_crit.symmetry_filter == "Semua Sampel"
    assert default_crit.gt_filter == "Semua"

    custom_crit = FilterCriteria(
        prompt_id="P1 (Direct)",
        domain="penalaran_logis",
        symmetry_filter="Hanya Bias Posisi (Flip)",
        gt_filter="Sesuai Ground Truth"
    )
    assert custom_crit.prompt_id == "P1 (Direct)"
    assert custom_crit.domain == "penalaran_logis"


def test_dual_inference_result_properties():
    res = DualInferenceResult(
        parsed_AB="A",
        raw_output_AB="A",
        logical_decision_AB=1,
        runtime_AB=0.45,
        parsed_BA="B",
        raw_output_BA="B",
        logical_decision_BA=1,
        runtime_BA=0.55,
        is_consistent=True,
        is_flip=False,
        mode="CPU_LLAMA",
        device_name="CPU",
        ram_usage_mb=1200.0,
        peak_ram_mb=1400.0
    )
    assert res.logical_decision_AB == 1
    assert res.logical_decision_BA == 1
    assert res.is_consistent is True
    assert res.is_flip is False
    assert res.runtime_AB + res.runtime_BA == 1.0


def test_mcnemar_metrics_consistency():
    m = McNemarMetrics(statistic=3.84, pvalue=0.05, b=25, c=10, total_flips=35)
    assert m.total_flips == m.b + m.c
    assert 0.0 <= m.pvalue <= 1.0
    d = m.to_dict()
    assert d["statistic"] == 3.84
    assert d["total_flips"] == 35
