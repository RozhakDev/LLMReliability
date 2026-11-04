"""Integration test untuk pipeline inferensi ganda dan evaluasi reliabilitas."""
from unittest.mock import MagicMock
from llm_reliability.services.evaluator import DualPositionEvaluator
from llm_reliability.services.backends.factory import BackendFactory
from llm_reliability.services.backends.base import InferenceBackend


def test_full_evaluation_pipeline_symmetry(monkeypatch):
    """Menguji integrasi aliran evaluasi dari input instruksi hingga kesimpulan flip."""
    mock_backend = MagicMock(spec=InferenceBackend)
    mock_backend.run_single_inference.side_effect = [
        ("A", "A", 0.35, {"ram_usage_mb": 1200.0, "mode": "CPU_LLAMA"}),
        ("A", "A", 0.38, {"ram_usage_mb": 1210.0, "mode": "CPU_LLAMA"})
    ]
    mock_backend.get_device_name.return_value = "CPU (llama.cpp)"

    monkeypatch.setattr(BackendFactory, "get_backend", lambda: mock_backend)

    res = DualPositionEvaluator.evaluate_symmetry(
        p_id="P1",
        input_text="Jelaskan konsep OOP.",
        cand_1="OOP berfokus pada objek dan kelas.",
        cand_2="OOP adalah singkatan dari Out of Place."
    )

    assert res.logical_decision_AB == 1
    assert res.logical_decision_BA == 2
    assert res.is_flip is True
    assert res.is_consistent is False
    assert res.runtime_AB == 0.35
    assert res.runtime_BA == 0.38
    assert res.ram_usage_mb == 1210.0
