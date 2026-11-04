"""Unit test untuk SimulationBackend (memastikan strict anti-dummy policy)."""
import pytest
from llm_reliability.core.runtime import detect_environment
from llm_reliability.services.backends.simulation import SimulationBackend


def test_simulation_backend_strict_policy():
    cfg = detect_environment()
    backend = SimulationBackend(cfg)
    assert backend.is_ready() is False
    assert backend.load_model() is False
    assert "Nonaktif" in backend.get_device_name()

    with pytest.raises(RuntimeError) as exc_info:
        backend.run_single_inference("P1", "q", "a", "b")
    assert "simulasi tidak aktif" in str(exc_info.value).lower()
