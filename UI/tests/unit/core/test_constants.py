"""Unit test untuk konstanta riset, metadata model, dan opsi filter."""
from llm_reliability.core.constants import (
    APP_TITLE,
    APP_SUBTITLE,
    APP_PROTOCOL_LABEL,
    APP_YEAR,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REPO,
    DEFAULT_GGUF_FILENAME,
    DATASET_NAME,
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_PAIRED_COUNT,
    DEFAULT_TOTAL_EVALUATION_COUNT,
    PROMPT_CODES,
    PROMPT_FILTER_ALL,
    PROMPT_DISPLAY_NAMES,
    PROMPT_FILTER_OPTIONS,
    SYMMETRY_FILTER_OPTIONS,
    GT_FILTER_OPTIONS,
    DEFAULT_DOWNLOAD_MIRRORS,
    PLAYGROUND_SAMPLE_CASES
)


def test_constants_metadata():
    assert APP_TITLE == "LLM-as-a-Judge Reliability Dashboard"
    assert "LLM-as-a-Judge" in APP_SUBTITLE
    assert "AB vs BA" in APP_PROTOCOL_LABEL
    assert APP_YEAR == "2026"


def test_constants_model_specification_strictly_qwen3():
    """Memastikan kepatuhan mutlak PKB: Model evaluator tunggal adalah Qwen3-4B-Instruct-2507."""
    assert DEFAULT_MODEL_NAME == "Qwen3-4B-Instruct-2507"
    assert DEFAULT_MODEL_REPO == "Qwen/Qwen3-4B-Instruct-2507"
    assert DEFAULT_GGUF_FILENAME == "qwen3-4b-instruct-2507-q4_k_m.gguf"


def test_constants_dataset_metadata():
    assert DATASET_NAME == "IndoPref"
    assert DEFAULT_SAMPLE_COUNT == 519
    assert DEFAULT_PAIRED_COUNT == 519 * 3
    assert DEFAULT_TOTAL_EVALUATION_COUNT == DEFAULT_PAIRED_COUNT * 2


def test_constants_prompt_codes_and_displays():
    assert PROMPT_CODES == ("P1", "P2", "P3")
    assert PROMPT_FILTER_ALL in PROMPT_DISPLAY_NAMES
    for code in PROMPT_CODES:
        assert code in PROMPT_DISPLAY_NAMES
        assert any(code in opt for opt in PROMPT_FILTER_OPTIONS)


def test_constants_filters():
    assert len(SYMMETRY_FILTER_OPTIONS) == 3
    assert len(GT_FILTER_OPTIONS) == 3


def test_constants_download_mirrors():
    assert len(DEFAULT_DOWNLOAD_MIRRORS) >= 2
    for url in DEFAULT_DOWNLOAD_MIRRORS:
        assert url.startswith("https://")
        assert "Qwen3-4B-Instruct-2507" in url or "qwen3-4b-instruct-2507" in url.lower()


def test_constants_playground_sample_cases():
    assert len(PLAYGROUND_SAMPLE_CASES) >= 3
    for case in PLAYGROUND_SAMPLE_CASES:
        assert "label" in case
        assert "prompt" in case
        assert "cand_1" in case
        assert "cand_2" in case
        assert len(case["prompt"]) > 0
        assert len(case["cand_1"]) > 0
        assert len(case["cand_2"]) > 0
