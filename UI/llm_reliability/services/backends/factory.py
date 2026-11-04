import os
from typing import Optional, Dict, Any

from llm_reliability.core.runtime import ComputeConfig, RuntimeEnvironment, detect_environment
from llm_reliability.services.backends.base import InferenceBackend
from llm_reliability.services.backends.cuda_transformers import CUDATransformersBackend
from llm_reliability.services.backends.cpu_llama import CPULlamaCppBackend
from llm_reliability.services.model_downloader import is_model_available, get_model_size_mb

class BackendFactory:
    """Pabrik penyedia backend inferensi berbasis profil lingkungan (Factory Pattern)."""

    _cached_backend: Optional[InferenceBackend] = None

    @classmethod
    def get_backend_status(cls, config: Optional[ComputeConfig] = None) -> Dict[str, Any]:
        """Memeriksa status kesiapan perangkat keras, dependensi, dan bobot model."""
        cfg = config or detect_environment()

        if cfg.env_mode == RuntimeEnvironment.COLAB_GPU:
            has_cuda = False
            try:
                import torch
                has_cuda = torch.cuda.is_available()
            except ImportError:
                has_cuda = False

            return {
                "is_ready": has_cuda,
                "backend_type": "COLAB_GPU",
                "device_name": "GPU CUDA",
                "model_identifier": cfg.model_identifier,
                "model_ready": True,
                "runtime_ready": has_cuda,
                "message": "GPU CUDA siap digunakan." if has_cuda else "CUDA tidak terdeteksi pada PyTorch."
            }

        model_path = cfg.model_identifier
        model_dir = os.path.dirname(model_path)
        model_ready = is_model_available(model_dir) or (os.path.exists(model_path) and os.path.getsize(model_path) >= 2_000_000_000)

        has_llama_cpp = False
        try:
            import llama_cpp
            has_llama_cpp = True
        except ImportError:
            has_llama_cpp = False

        is_ready = model_ready and has_llama_cpp

        missing = []
        if not model_ready:
            missing.append("Bobot model GGUF belum diunduh")
        if not has_llama_cpp:
            missing.append("Pustaka llama-cpp-python belum terpasang di venv")

        msg = "CPU Llama.cpp siap digunakan." if is_ready else f"Belum siap: {', '.join(missing)}."

        return {
            "is_ready": is_ready,
            "backend_type": "CPU_LLAMA",
            "device_name": f"CPU ({cfg.cpu_threads} Threads)",
            "model_identifier": model_path,
            "model_ready": model_ready,
            "model_size_mb": get_model_size_mb(model_dir) if model_ready else 0.0,
            "runtime_ready": has_llama_cpp,
            "message": msg
        }

    @classmethod
    def get_backend(cls, config: Optional[ComputeConfig] = None) -> InferenceBackend:
        """Mengembalikan backend inferensi aktif. Tidak ada fallback ke simulasi."""
        if cls._cached_backend is not None:
            return cls._cached_backend

        cfg = config or detect_environment()

        if cfg.env_mode == RuntimeEnvironment.COLAB_GPU:
            try:
                import torch
                if torch.cuda.is_available():
                    backend = CUDATransformersBackend(cfg)
                    cls._cached_backend = backend
                    return backend
            except Exception:
                pass
            raise RuntimeError(
                "Akselerasi GPU CUDA tidak tersedia pada sesi ini. "
                "Silakan periksa ketersediaan driver dan CUDA toolkit."
            )

        status = cls.get_backend_status(cfg)
        if not status["model_ready"]:
            raise RuntimeError(
                f"Bobot model Qwen3-4B-Instruct-2507 belum tersedia di: {cfg.model_identifier}. "
                "Silakan unduh bobot model asli terlebih dahulu. "
                "Gunakan fungsi unduh model atau tentukan path lokal yang valid."
            )

        if not status["runtime_ready"]:
            raise RuntimeError(
                "Pustaka 'llama-cpp-python' belum terpasang di virtual environment aktif. "
                "Jalankan: pip install llama-cpp-python"
            )

        backend = CPULlamaCppBackend(cfg)
        cls._cached_backend = backend
        return backend

    @classmethod
    def reset_cache(cls) -> None:
        """Mereset cache backend."""
        cls._cached_backend = None
