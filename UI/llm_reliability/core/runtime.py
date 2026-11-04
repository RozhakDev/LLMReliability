from dataclasses import dataclass
from llm_reliability.core.settings import RuntimeEnvironment, AppSettings, get_settings

@dataclass(frozen=True)
class ComputeConfig:
    """Konfigurasi perangkat keras, bobot model, dan sink data (Adapter ke AppSettings)."""
    env_mode: RuntimeEnvironment
    device: str
    model_identifier: str
    quantization: str
    cpu_threads: int
    data_root: str
    model_dir: str
    track_memory: bool
    context_size: int = 2048
    temperature: float = 0.0
    max_tokens: int = 16

def detect_environment() -> ComputeConfig:
    """
    Mendeteksi profil perangkat keras dan lingkungan eksekusi secara otomatis.
    Mendelegasikan ke modul AppSettings terpusat (12-Factor App).
    """
    s: AppSettings = get_settings()
    return ComputeConfig(
        env_mode=s.env_mode,
        device=s.device,
        model_identifier=s.model_identifier,
        quantization=s.quantization,
        cpu_threads=s.cpu_threads,
        data_root=s.data_root,
        model_dir=s.model_dir,
        track_memory=s.track_memory,
        context_size=s.context_size,
        temperature=s.temperature,
        max_tokens=s.max_tokens,
    )
