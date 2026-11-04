"""Unit test untuk BackendFactory dan resolusi hardware backend."""
import pytest
from unittest.mock import patch, MagicMock
from llm_reliability.services.backends.factory import BackendFactory
from llm_reliability.services.backends.base import InferenceBackend


def test_backend_factory_status_diagnostics():
    status = BackendFactory.get_backend_status()
    assert isinstance(status, dict)
    assert "is_ready" in status
    assert "device_name" in status
    assert "model_ready" in status
    assert "runtime_ready" in status


def test_backend_factory_singleton_and_reset():
    BackendFactory._cached_backend = MagicMock(spec=InferenceBackend)
    assert BackendFactory.get_backend() is not None

    BackendFactory.reset_cache()
    assert BackendFactory._cached_backend is None


def test_backend_factory_colab_gpu_branch(monkeypatch):
    BackendFactory.reset_cache()
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    import sys
    monkeypatch.setitem(sys.modules, "torch", mock_torch)

    from llm_reliability.core.runtime import ComputeConfig, RuntimeEnvironment
    cfg_colab = ComputeConfig(
        env_mode=RuntimeEnvironment.COLAB_GPU,
        device="cuda",
        model_identifier="Qwen/Qwen3-4B-Instruct-2507",
        quantization="4bit_nf4",
        cpu_threads=2,
        data_root="UI/data",
        model_dir="UI/data/models",
        track_memory=False
    )
    status = BackendFactory.get_backend_status(cfg_colab)
    assert status["backend_type"] == "COLAB_GPU"
    assert status["is_ready"] is True

    backend = BackendFactory.get_backend(cfg_colab)
    assert isinstance(backend, InferenceBackend)

    BackendFactory.reset_cache()
    mock_torch.cuda.is_available.return_value = False
    with pytest.raises(RuntimeError) as exc_info:
        BackendFactory.get_backend(cfg_colab)
    assert "akselerasi gpu cuda tidak tersedia" in str(exc_info.value).lower()


def test_backend_factory_local_cpu_runtime_missing(monkeypatch, fresh_settings):
    BackendFactory.reset_cache()
    monkeypatch.setenv("LLM_COMPUTE_ENV", "LOCAL_CPU")
    with patch('llm_reliability.services.backends.factory.BackendFactory.get_backend_status') as mock_status:
        mock_status.return_value = {
            "is_ready": False,
            "model_ready": True,
            "runtime_ready": False,
            "device_name": "CPU"
        }
        with pytest.raises(RuntimeError) as exc_info:
            BackendFactory.get_backend()
        assert "llama-cpp-python" in str(exc_info.value).lower()
