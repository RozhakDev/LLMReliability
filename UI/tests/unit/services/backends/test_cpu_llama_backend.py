"""Unit test untuk CPULlamaCppBackend (eksekusi Qwen3 GGUF via engine llama_cpp)."""
import pytest
from unittest.mock import MagicMock, patch
from llm_reliability.core.runtime import detect_environment
from llm_reliability.services.backends.cpu_llama import CPULlamaCppBackend


def test_cpu_backend_instantiation_and_device():
    cfg = detect_environment()
    backend = CPULlamaCppBackend(cfg)
    assert "CPU" in backend.get_device_name()


def test_cpu_backend_run_inference_mock():
    cfg = detect_environment()
    backend = CPULlamaCppBackend(cfg)

    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        'choices': [{'message': {'content': 'A'}}]
    }
    backend._model = mock_llm

    with patch('psutil.Process') as mock_proc:
        mock_p_inst = MagicMock()
        mock_p_inst.memory_info.return_value = MagicMock(rss=1024 * 1024 * 1500)
        mock_proc.return_value = mock_p_inst

        raw, parsed, runtime, telem = backend.run_single_inference("P1", "Tanya", "A", "B")
        assert raw == "A"
        assert parsed == "A"
        assert runtime >= 0.0
        assert "ram_usage_mb" in telem
