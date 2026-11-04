"""Unit test untuk CUDATransformersBackend."""
import sys
import pytest
from unittest.mock import MagicMock
from llm_reliability.core.runtime import detect_environment
from llm_reliability.services.backends.cuda_transformers import CUDATransformersBackend


def test_cuda_backend_raises_when_no_cuda(monkeypatch):
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    monkeypatch.setitem(sys.modules, "torch", mock_torch)

    cfg = detect_environment()
    backend = CUDATransformersBackend(cfg)
    assert backend.load_model() is False
    with pytest.raises(RuntimeError) as exc_info:
        backend.run_single_inference("P1", "test", "cand A", "cand B")
    assert "gagal memuat" in str(exc_info.value).lower()


def test_cuda_backend_format_prompt_and_parsing():
    cfg = detect_environment()
    backend = CUDATransformersBackend(cfg)

    p1 = backend._format_prompt("P1", "Instr", "RespA", "RespB")
    p2 = backend._format_prompt("P2", "Instr", "RespA", "RespB")
    p3 = backend._format_prompt("P3", "Instr", "RespA", "RespB")
    p_unk = backend._format_prompt("UNKNOWN", "Instr", "RespA", "RespB")
    assert "### TUGAS" in p1
    assert "Akurasi" in p2
    assert "ATURAN KETAT" in p3
    assert p_unk == ""

    assert "<system>" in backend._apply_chat_template("Halo")

    assert backend._parse_decision("Pilihan: A") == "A"
    assert backend._parse_decision("B.") == "B"
    assert backend._parse_decision("Tidak Valid") == "INVALID"


def test_cuda_backend_mock_run_single_inference(monkeypatch):
    mock_torch = MagicMock()
    monkeypatch.setitem(sys.modules, "torch", mock_torch)

    cfg = detect_environment()
    backend = CUDATransformersBackend(cfg)

    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.return_value = "A"
    mock_tokenizer.return_value = MagicMock(to=lambda dev: {'input_ids': MagicMock(shape=[1, 10])})

    mock_model = MagicMock()
    mock_model.generate.return_value = [MagicMock()]
    mock_model.device = "cuda:0"

    backend.tokenizer = mock_tokenizer
    backend.model = mock_model
    backend.device_name = "CUDA (NVIDIA A100)"

    assert backend.is_ready() is True
    assert "NVIDIA" in backend.get_device_name()

    parsed, raw, runtime, telem = backend.run_single_inference("P1", "Instruksi", "RespA", "RespB")
    assert parsed == "A"
    assert raw == "A"
    assert runtime >= 0.0
    assert telem["mode"] == "COLAB_GPU (4-bit NF4)"
