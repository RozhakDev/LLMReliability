"""Unit test untuk DualPositionEvaluator, formatting prompt, dan regex parser."""
from unittest.mock import MagicMock
from llm_reliability.services.evaluator import DualPositionEvaluator, format_prompt, parse_decision


def test_format_prompt_all_strategies():
    p1 = format_prompt("P1", "Pertanyaan", "A", "B")
    assert "secara keseluruhan lebih baik" in p1
    assert "### RESPONS A" in p1
    assert "### RESPONS B" in p1

    p2 = format_prompt("P2", "Pertanyaan", "A", "B")
    assert "Akurasi" in p2 and "Relevansi" in p2 and "Kejelasan" in p2

    p3 = format_prompt("P3", "Pertanyaan", "A", "B")
    assert "Abaikan urutan kemunculan respons" in p3


def test_format_prompt_unknown_id():
    assert format_prompt("UNKNOWN", "q", "a", "b") == ""


def test_apply_chat_template():
    mock_tok = MagicMock()
    mock_tok.apply_chat_template.return_value = "<chat>formatted</chat>"
    res = DualPositionEvaluator.apply_chat_template(mock_tok, "Hello")
    assert res == "<chat>formatted</chat>"

    # Fallback saat tokenizer None
    res_fallback = DualPositionEvaluator.apply_chat_template(None, "Hello")
    assert "<system>" in res_fallback and "Hello" in res_fallback


def test_parse_decision_standard_letters():
    assert parse_decision("A") == "A"
    assert parse_decision("B") == "B"
    assert parse_decision("a") == "A"
    assert parse_decision("b") == "B"


def test_parse_decision_noisy_and_adversarial():
    assert parse_decision("Jawaban: A") == "A"
    assert parse_decision("Kandidat B adalah pilihan saya") == "B"
    assert parse_decision("(A).") == "A"
    assert parse_decision("**B**") == "B"
    assert parse_decision("\n  A  \n") == "A"
    assert parse_decision("tidak tahu") == "INVALID"
    assert parse_decision("C") == "INVALID"
    assert parse_decision("") == "INVALID"
    assert parse_decision("AB") == "INVALID"


def test_evaluate_symmetry_logic(monkeypatch):
    """Memastikan logika pertukaran posisi AB dan BA bekerja secara presisi."""
    mock_backend = MagicMock()
    mock_backend.run_single_inference.side_effect = [
        ("A", "A", 0.2, {"ram_usage_mb": 1000}),
        ("B", "B", 0.2, {"ram_usage_mb": 1000})
    ]
    mock_backend.get_device_name.return_value = "CPU"

    monkeypatch.setattr("llm_reliability.services.backends.factory.BackendFactory.get_backend", lambda: mock_backend)

    res = DualPositionEvaluator.evaluate_symmetry("P1", "Instruksi", "Kandidat 1", "Kandidat 2")
    assert res.logical_decision_AB == 1
    assert res.logical_decision_BA == 1
    assert res.is_consistent is True
    assert res.is_flip is False


def test_evaluator_run_single_inference_mock(monkeypatch):
    import sys
    mock_torch = MagicMock()
    monkeypatch.setitem(sys.modules, "torch", mock_torch)

    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.return_value = "A"
    mock_tokenizer.return_value = MagicMock(to=lambda dev: {'input_ids': MagicMock(shape=[1, 10])})

    mock_model = MagicMock()
    mock_model.generate.return_value = [MagicMock()]
    mock_model.device = "cuda:0"

    parsed, raw, runtime = DualPositionEvaluator.run_single_inference(
        mock_tokenizer, mock_model, "P1", "Instr", "Resp A", "Resp B"
    )
    assert parsed == "A"
    assert raw == "A"
    assert runtime >= 0.0
