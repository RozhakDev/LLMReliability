"""
Pengaturan Aplikasi Berbasis 12-Factor App (Environment & Settings).
Mendukung pemuatan dinamis dari variabel lingkungan (.env) dengan fallback bawaan yang aman.
"""

import os
from enum import Enum
from dataclasses import dataclass
from typing import List
from functools import lru_cache

from llm_reliability.core.paths import PROJECT_ROOT
from llm_reliability.core.constants import (
    DEFAULT_MODEL_REPO,
    DEFAULT_GGUF_FILENAME,
    DEFAULT_STATISTICAL_ALPHA,
    DEFAULT_FLIP_ALERT_THRESHOLD,
    DEFAULT_CONSISTENCY_HIGH_THRESHOLD,
    DEFAULT_DOWNLOAD_MIRRORS,
    EXPECTED_MIN_MODEL_BYTES,
)

class RuntimeEnvironment(str, Enum):
    """Profil lingkungan komputasi yang didukung platform."""
    COLAB_GPU = "COLAB_GPU"
    LOCAL_CPU = "LOCAL_CPU"
    VPS_CPU = "VPS_CPU"

@dataclass(frozen=True)
class AppSettings:
    """Kontrak konfigurasi lengkap aplikasi, perangkat keras, dan data empiris."""
    env_mode: RuntimeEnvironment
    device: str
    model_identifier: str
    quantization: str
    cpu_threads: int
    context_size: int
    temperature: float
    max_tokens: int
    data_root: str
    model_dir: str
    drive_root: str
    track_memory: bool
    statistical_alpha: float
    flip_alert_threshold: float
    consistency_high_threshold: float
    download_mirrors: List[str]
    expected_min_bytes: int

def _get_int_env(key: str, default: int) -> int:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _get_float_env(key: str, default: float) -> float:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """
    Memuat dan memvalidasi konfigurasi aplikasi secara terpusat (Singleton via lru_cache).
    Prioritas: Variabel Lingkungan (.env) > Deteksi Otomatis Sistem > Nilai Baku.
    """
    env_override = os.environ.get("LLM_COMPUTE_ENV") or os.environ.get("COMPUTE_ENV")
    drive_root = os.environ.get("DRIVE_ROOT", "/content/drive/MyDrive/LLMReliability")
    is_colab = os.path.exists(drive_root)

    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    ui_dir = os.path.abspath(str(PROJECT_ROOT))
    default_local_data = os.path.join(ui_dir, "data")
    default_local_models = os.path.join(default_local_data, "models")

    if env_override == "COLAB_GPU" or (is_colab and has_cuda):
        drive_model_dir = os.path.join(drive_root, "models")
        os.environ["HF_HOME"] = drive_model_dir

        return AppSettings(
            env_mode=RuntimeEnvironment.COLAB_GPU,
            device="cuda",
            model_identifier=os.environ.get("MODEL_REPO", DEFAULT_MODEL_REPO),
            quantization="4bit_nf4",
            cpu_threads=_get_int_env("CPU_THREADS", 2),
            context_size=_get_int_env("LLM_CONTEXT_SIZE", 2048),
            temperature=_get_float_env("LLM_TEMPERATURE", 0.0),
            max_tokens=_get_int_env("LLM_MAX_TOKENS", 16),
            data_root=os.environ.get("DATA_ROOT", drive_root),
            model_dir=os.environ.get("MODEL_DIR", drive_model_dir),
            drive_root=drive_root,
            track_memory=False,
            statistical_alpha=_get_float_env("STATISTICAL_ALPHA", DEFAULT_STATISTICAL_ALPHA),
            flip_alert_threshold=_get_float_env("FLIP_ALERT_THRESHOLD", DEFAULT_FLIP_ALERT_THRESHOLD),
            consistency_high_threshold=_get_float_env("CONSISTENCY_HIGH_THRESHOLD", DEFAULT_CONSISTENCY_HIGH_THRESHOLD),
            download_mirrors=DEFAULT_DOWNLOAD_MIRRORS,
            expected_min_bytes=EXPECTED_MIN_MODEL_BYTES,
        )

    is_vps = env_override == "VPS_CPU" or (os.name != "nt" and not is_colab)
    mode = RuntimeEnvironment.VPS_CPU if is_vps else RuntimeEnvironment.LOCAL_CPU

    data_root = os.environ.get("DATA_ROOT", default_local_data)
    model_dir = os.environ.get("MODEL_DIR", default_local_models)
    local_gguf_path = os.environ.get(
        "LOCAL_GGUF_PATH",
        os.path.join(model_dir, DEFAULT_GGUF_FILENAME)
    )

    return AppSettings(
        env_mode=mode,
        device="cpu",
        model_identifier=local_gguf_path,
        quantization="gguf_q4",
        cpu_threads=_get_int_env("CPU_THREADS", 4),
        context_size=_get_int_env("LLM_CONTEXT_SIZE", 2048),
        temperature=_get_float_env("LLM_TEMPERATURE", 0.0),
        max_tokens=_get_int_env("LLM_MAX_TOKENS", 16),
        data_root=data_root,
        model_dir=model_dir,
        drive_root=drive_root,
        track_memory=True,
        statistical_alpha=_get_float_env("STATISTICAL_ALPHA", DEFAULT_STATISTICAL_ALPHA),
        flip_alert_threshold=_get_float_env("FLIP_ALERT_THRESHOLD", DEFAULT_FLIP_ALERT_THRESHOLD),
        consistency_high_threshold=_get_float_env("CONSISTENCY_HIGH_THRESHOLD", DEFAULT_CONSISTENCY_HIGH_THRESHOLD),
        download_mirrors=DEFAULT_DOWNLOAD_MIRRORS,
        expected_min_bytes=EXPECTED_MIN_MODEL_BYTES,
    )

def clear_settings_cache() -> None:
    """Mereset cache konfigurasi (berguna saat pengujian atau perubahan env runtime)."""
    get_settings.cache_clear()
