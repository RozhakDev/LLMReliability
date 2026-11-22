import logging

import pytest
import torch

from llm_exp.config import get_model_spec
from llm_exp.prompts import format_prompt, get_system_prompt, prepare_model_inputs

pytestmark = pytest.mark.unit


def test_system_prompt_matches_notebook_08() -> None:
    assert get_system_prompt() == (
        "Anda adalah evaluator AI yang objektif. "
        "Anda tidak memiliki bias posisi. "
        "Jawab HANYA dengan pilihan kandidat (A atau B)."
    )


@pytest.mark.parametrize("prompt_id", ["P1", "P2", "P3"])
def test_format_prompt_renders_and_preserves_braces(prompt_id: str) -> None:
    text = format_prompt('Contoh {"k": 1}', "Jawaban {a}", "Jawaban {b}", prompt_id)
    assert '### INSTRUKSI\nContoh {"k": 1}' in text
    assert "### RESPONS A\nJawaban {a}" in text
    assert "### RESPONS B\nJawaban {b}" in text


def test_format_prompt_rejects_unknown_id() -> None:
    with pytest.raises(ValueError, match="Unsupported prompt_id"):
        format_prompt("q", "a", "b", "P9")


@pytest.mark.parametrize("model_key", ["qwen3_32b", "aya_expanse_32b", "glm_4_32b"])
def test_prepare_model_inputs_causal_prevents_double_bos_and_strips_token_type_ids(
    fake_tokenizer,
    model_key: str,
) -> None:
    inputs = prepare_model_inputs(fake_tokenizer, "Halo", get_model_spec(model_key), torch.device("cpu"))
    assert inputs["input_ids"].shape == (1, 3)
    assert fake_tokenizer.last_add_special_tokens is False
    assert "token_type_ids" not in inputs


def test_prepare_model_inputs_multimodal_and_fallback(
    fake_multimodal_processor,
    test_logger: logging.Logger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    spec = get_model_spec("gemma3_27b_goto")
    vlm_inputs = prepare_model_inputs(fake_multimodal_processor(False), "Halo", spec, torch.device("cpu"))
    assert vlm_inputs["input_ids"].shape == (1, 2)
    assert "token_type_ids" in vlm_inputs

    with caplog.at_level(logging.WARNING, logger=test_logger.name):
        fallback_inputs = prepare_model_inputs(
            fake_multimodal_processor(True),
            "Halo",
            spec,
            torch.device("cpu"),
            logger=test_logger,
        )
    assert fallback_inputs["input_ids"].shape == (1, 3)
    assert "chat_template_fallback" in caplog.text
