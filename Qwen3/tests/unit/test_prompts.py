from typing import Callable

import pytest
import torch

from qwen3_eval.config import load_model_spec
from qwen3_eval.prompts import format_chat_text, format_prompt, get_system_prompt, prepare_model_inputs

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


def test_format_chat_text_applies_non_thinking_and_handles_type_error(
    fake_tokenizer,
    fake_tokenizer_factory: Callable[..., object],
) -> None:
    spec = load_model_spec()
    text_with_flag = format_chat_text(fake_tokenizer, "Halo", spec)
    assert "|THINK:False" in text_with_flag

    legacy_tokenizer = fake_tokenizer_factory(reject_thinking_kwarg=True)
    text_fallback = format_chat_text(legacy_tokenizer, "Halo", spec)
    assert "|THINK:None" in text_fallback


def test_prepare_model_inputs_prevents_double_bos_and_strips_token_type_ids(
    fake_tokenizer,
) -> None:
    inputs = prepare_model_inputs(fake_tokenizer, "Halo", load_model_spec(), torch.device("cpu"))
    assert inputs["input_ids"].shape == (1, 3)
    assert fake_tokenizer.last_add_special_tokens is False
    assert "token_type_ids" not in inputs
