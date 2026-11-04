from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
from llm_reliability.core.runtime import ComputeConfig

class InferenceBackend(ABC):
    """Antarmuka abstrak mesin inferensi evaluasi (Strategy Pattern & DIP)."""

    def __init__(self, config: ComputeConfig):
        self.config = config

    @abstractmethod
    def load_model(self) -> bool:
        """Memuat bobot model ke memori/perangkat."""
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Memeriksa apakah model siap dieksekusi."""
        pass

    @abstractmethod
    def get_device_name(self) -> str:
        """Mengembalikan deskripsi perangkat keras yang aktif."""
        pass

    @abstractmethod
    def run_single_inference(
        self,
        p_id: str,
        input_text: str,
        resp_a: str,
        resp_b: str
    ) -> Tuple[str, str, float, Dict[str, Any]]:
        """
        Mengeksekusi satu kali inferensi evaluasi.
        Mengembalikan tuple: (parsed_decision, raw_output, runtime_sec, telemetry_dict).
        """
        pass
