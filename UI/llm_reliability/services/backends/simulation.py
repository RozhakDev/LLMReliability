from typing import Tuple, Dict, Any
from llm_reliability.core.runtime import ComputeConfig
from llm_reliability.services.backends.base import InferenceBackend

class SimulationBackend(InferenceBackend):
    """
    [DINONAKTIFKAN] Backend simulasi dinonaktifkan.
    Sistem hanya mendukung eksekusi inferensi berbasis bobot model nyata.
    """

    def __init__(self, config: ComputeConfig):
        super().__init__(config)
        self.device_name = "CPU (Nonaktif)"

    def load_model(self) -> bool:
        return False

    def is_ready(self) -> bool:
        return False

    def get_device_name(self) -> str:
        return self.device_name

    def run_single_inference(
        self,
        p_id: str,
        input_text: str,
        resp_a: str,
        resp_b: str
    ) -> Tuple[str, str, float, Dict[str, Any]]:
        raise RuntimeError(
            "Backend simulasi tidak aktif. "
            "Gunakan bobot model Qwen3-4B-Instruct-2507 untuk inferensi nyata."
        )
