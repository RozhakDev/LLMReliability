import os
import re
import time
from typing import Tuple, Dict, Any

from llm_reliability.core.runtime import ComputeConfig
from llm_reliability.services.backends.base import InferenceBackend

class CPULlamaCppBackend(InferenceBackend):
    """
    Mesin inferensi Local CPU (Ryzen 3, 8GB RAM) & VPS CPU menggunakan format GGUF Q4_K_M (llama-cpp-python).
    Mengonsumsi hanya ~2.4 GB RAM fisik dan dilengkapi pengukuran lonjakan RAM aktual via psutil.
    """

    def __init__(self, config: ComputeConfig):
        super().__init__(config)
        self.llm = None
        self.device_name = f"CPU ({config.cpu_threads} Threads)"

    def load_model(self) -> bool:
        if self.llm is not None:
            return True

        model_path = self.config.model_identifier
        if not os.path.exists(model_path) or not os.path.isfile(model_path):
            return False

        try:
            from llama_cpp import Llama

            self.llm = Llama(
                model_path=model_path,
                n_threads=self.config.cpu_threads,
                n_ctx=self.config.context_size,
                verbose=False
            )
            return True
        except Exception:
            self.llm = None
            return False

    def is_ready(self) -> bool:
        return self.llm is not None

    def get_device_name(self) -> str:
        return self.device_name

    @staticmethod
    def _format_prompt(p_id: str, input_text: str, resp_a: str, resp_b: str) -> str:
        if p_id == 'P1':
            return (
                "### TUGAS\n"
                "Bandingkan dua respons AI di bawah ini untuk instruksi yang sama. Pilih mana yang secara keseluruhan lebih baik.\n\n"
                f"### INSTRUKSI\n{input_text}\n\n"
                f"### RESPONS A\n{resp_a}\n\n"
                f"### RESPONS B\n{resp_b}\n\n"
                "### FORMAT JAWABAN\n"
                "Keluarkan HANYA satu huruf: A atau B. Dilarang menulis alasan atau teks lain."
            )
        elif p_id == 'P2':
            return (
                "### TUGAS\n"
                "Nilai dua respons AI berikut berdasarkan: Akurasi (kebenaran), Relevansi (menjawab instruksi), dan Kejelasan (tidak bertele-tele).\n\n"
                f"### INSTRUKSI\n{input_text}\n\n"
                f"### RESPONS A\n{resp_a}\n\n"
                f"### RESPONS B\n{resp_b}\n\n"
                "### FORMAT JAWABAN\n"
                "Berdasarkan kriteria di atas, keluarkan HANYA satu huruf: A atau B. Dilarang memberikan penjelasan."
            )
        elif p_id == 'P3':
            return (
                "### TUGAS\n"
                "Tentukan respons terbaik untuk instruksi di bawah ini.\n\n"
                f"### INSTRUKSI\n{input_text}\n\n"
                f"### RESPONS A\n{resp_a}\n\n"
                f"### RESPONS B\n{resp_b}\n\n"
                "### ATURAN KETAT\n"
                "- Abaikan urutan kemunculan respons (jangan bias posisi).\n"
                "- Abaikan panjang respons (jangan bias verbosity).\n"
                "- FORMAT JAWABAN WAJIB: Keluarkan hanya karakter A atau B."
            )
        return ""

    @staticmethod
    def _parse_decision(raw_output: str) -> str:
        clean = str(raw_output).strip().upper()
        match = re.search(r'\b([AB])\b', clean)
        return match.group(1) if match else 'INVALID'

    def run_single_inference(
        self,
        p_id: str,
        input_text: str,
        resp_a: str,
        resp_b: str
    ) -> Tuple[str, str, float, Dict[str, Any]]:
        if not self.is_ready():
            if not self.load_model():
                raise RuntimeError(
                    f"File bobot model GGUF tidak ditemukan di: {self.config.model_identifier}. "
                    "Silakan unduh bobot model menggunakan UI/services/model_downloader.py terlebih dahulu."
                )

        import psutil
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)

        start_time = time.time()
        user_prompt = self._format_prompt(p_id, input_text, resp_a, resp_b)

        messages = [
            {'role': 'system', 'content': 'Anda adalah evaluator AI yang objektif. Anda tidak memiliki bias posisi. Jawab HANYA dengan pilihan kandidat (A atau B).'},
            {'role': 'user', 'content': user_prompt}
        ]

        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )

        raw_output = response["choices"][0]["message"]["content"].strip()
        runtime = time.time() - start_time

        mem_after = process.memory_info().rss / (1024 * 1024)
        peak_mem = max(mem_before, mem_after)

        parsed = self._parse_decision(raw_output)
        telemetry = {
            "mode": f"{self.config.env_mode.value} (GGUF Q4)",
            "device_name": self.device_name,
            "ram_usage_mb": round(mem_after, 1),
            "peak_ram_mb": round(peak_mem, 1)
        }
        return parsed, raw_output, runtime, telemetry
