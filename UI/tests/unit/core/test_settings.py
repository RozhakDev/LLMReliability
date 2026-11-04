"""Unit test untuk pengaturan 12-Factor AppSettings."""
from llm_reliability.core.settings import get_settings, clear_settings_cache


def test_settings_defaults(fresh_settings, monkeypatch):
    for key in ("CPU_THREADS", "LLM_CONTEXT_SIZE", "LLM_TEMPERATURE", "LLM_MAX_TOKENS",
                "STATISTICAL_ALPHA", "LLM_COMPUTE_ENV", "COMPUTE_ENV"):
        monkeypatch.delenv(key, raising=False)
    s = get_settings()
    assert s.statistical_alpha == 0.05
    assert s.temperature == 0.0
    assert s.max_tokens == 16
    assert s.context_size == 2048
    assert s.cpu_threads >= 1


def test_settings_environment_overrides(fresh_settings, monkeypatch):
    monkeypatch.setenv("STATISTICAL_ALPHA", "0.01")
    monkeypatch.setenv("LLM_MAX_TOKENS", "32")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.7")
    monkeypatch.setenv("CPU_THREADS", "8")
    s = get_settings()
    assert s.statistical_alpha == 0.01
    assert s.max_tokens == 32
    assert s.temperature == 0.7
    assert s.cpu_threads == 8


def test_settings_invalid_numeric_fallback(fresh_settings, monkeypatch):
    monkeypatch.setenv("LLM_TEMPERATURE", "bukan_angka")
    monkeypatch.setenv("LLM_MAX_TOKENS", "invalid")
    s = get_settings()
    assert s.temperature == 0.0
    assert s.max_tokens == 16
