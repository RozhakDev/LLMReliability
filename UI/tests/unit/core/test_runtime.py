"""Unit test untuk deteksi perangkat keras dan runtime environment."""
import sys
from unittest.mock import MagicMock
from llm_reliability.core.runtime import detect_environment, ComputeConfig
from llm_reliability.core.settings import RuntimeEnvironment


def test_detect_environment_cuda_available(monkeypatch, fresh_settings):
    monkeypatch.setenv("LLM_COMPUTE_ENV", "COLAB_GPU")
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    monkeypatch.setitem(sys.modules, "torch", mock_torch)

    cfg = detect_environment()
    assert isinstance(cfg, ComputeConfig)
    assert cfg.env_mode == RuntimeEnvironment.COLAB_GPU
    assert cfg.temperature == 0.0
    assert cfg.context_size == 2048


def test_detect_environment_cpu_fallback(monkeypatch, fresh_settings):
    monkeypatch.setenv("LLM_COMPUTE_ENV", "LOCAL_CPU")
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    monkeypatch.setitem(sys.modules, "torch", mock_torch)

    cfg = detect_environment()
    assert cfg.env_mode == RuntimeEnvironment.LOCAL_CPU
    assert "cpu" in cfg.device.lower()
