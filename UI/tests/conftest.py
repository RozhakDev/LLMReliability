"""Fixture bersama untuk seluruh test suite (Unit, Integration, dan E2E)."""
import pytest
import pandas as pd
from typing import Dict, Any

from llm_reliability.core.paths import DATA_DIR
from llm_reliability.domain import DualInferenceResult, McNemarMetrics


@pytest.fixture(scope="session")
def results_dir():
    path = DATA_DIR / "results"
    if not (path / "evaluation_results_detailed.parquet").exists():
        pytest.skip("Data evaluasi riil tidak ditemukan di UI/data/results")
    return path


@pytest.fixture()
def fresh_settings():
    """Reset cache settings agar override environment terbaca, lalu pulihkan."""
    from llm_reliability.core.settings import clear_settings_cache

    clear_settings_cache()
    yield
    clear_settings_cache()


@pytest.fixture()
def mock_paired_df() -> pd.DataFrame:
    """Menyediakan DataFrame evaluasi paired terstruktur untuk pengujian view dan repository."""
    return pd.DataFrame([
        {
            "sample_id": 1,
            "prompt_id": "P1",
            "domain": "penalaran_logis",
            "input": "Jelaskan hukum Newton pertama.",
            "response_1": "Benda diam akan tetap diam.",
            "response_2": "Semua benda bergerak tanpa henti.",
            "chosen_model": "Qwen3-4B",
            "rejected_model": "Llama-3",
            "ground_truth": 1,
            "is_flip": False,
            "is_consistent": True,
            "is_correct_AB": True,
            "is_correct_BA": True,
            "is_robust_correct": True,
            "parsed_decision_AB": "A",
            "parsed_decision_BA": "B",
            "logical_decision_AB": "1",
            "logical_decision_BA": "1",
            "raw_output_AB": "A",
            "raw_output_BA": "B",
            "runtime_sec_AB": 0.45,
            "runtime_sec_BA": 0.42
        },
        {
            "sample_id": 2,
            "prompt_id": "P1",
            "domain": "bahasa_sastra",
            "input": "Buat puisi senja di dermaga.",
            "response_1": "Senja temaram di ufuk barat.",
            "response_2": "Matahari terbenam indah sekali.",
            "chosen_model": "Mistral-7B",
            "rejected_model": "Gemma-2",
            "ground_truth": 2,
            "is_flip": True,
            "is_consistent": False,
            "is_correct_AB": False,
            "is_correct_BA": True,
            "is_robust_correct": False,
            "parsed_decision_AB": "A",
            "parsed_decision_BA": "A",
            "logical_decision_AB": "1",
            "logical_decision_BA": "2",
            "raw_output_AB": "A",
            "raw_output_BA": "A",
            "runtime_sec_AB": 0.50,
            "runtime_sec_BA": 0.48
        }
    ])


@pytest.fixture()
def mock_metrics_summary_df() -> pd.DataFrame:
    """Menyediakan DataFrame ringkasan metrik 3 strategi prompt."""
    return pd.DataFrame([
        {
            "Prompt": "P1",
            "Accuracy (%)": 65.5,
            "Consistency (%)": 78.0,
            "Flip Rate (%)": 22.0,
            "Cohen Kappa": 0.560,
            "Kappa 95% CI": "[0.51, 0.61]"
        },
        {
            "Prompt": "P2",
            "Accuracy (%)": 68.2,
            "Consistency (%)": 81.5,
            "Flip Rate (%)": 18.5,
            "Cohen Kappa": 0.630,
            "Kappa 95% CI": "[0.58, 0.68]"
        },
        {
            "Prompt": "P3",
            "Accuracy (%)": 64.0,
            "Consistency (%)": 85.0,
            "Flip Rate (%)": 15.0,
            "Cohen Kappa": 0.700,
            "Kappa 95% CI": "[0.65, 0.75]"
        }
    ])


@pytest.fixture()
def mock_mcnemar_dict() -> Dict[str, Any]:
    """Menyediakan dictionary hasil uji McNemar mock."""
    return {
        "ALL": McNemarMetrics(statistic=4.5, pvalue=0.034, b=50, c=30, total_flips=80, is_significant=True).to_dict(),
        "P1": McNemarMetrics(statistic=2.1, pvalue=0.147, b=20, c=12, total_flips=32, is_significant=False).to_dict(),
        "P2": McNemarMetrics(statistic=1.5, pvalue=0.220, b=15, c=10, total_flips=25, is_significant=False).to_dict(),
        "P3": McNemarMetrics(statistic=0.9, pvalue=0.342, b=15, c=8, total_flips=23, is_significant=False).to_dict()
    }
