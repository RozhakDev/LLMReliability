import re
import time
import gc
from typing import Tuple, Dict, Any

from llm_reliability.core.runtime import ComputeConfig
from llm_reliability.services.backends.base import InferenceBackend

class CUDATransformersBackend(InferenceBackend):
    """
    Mesin inferensi Colab GPU (CUDA) menggunakan Hugging Face Transformers & BitsAndBytes 4-bit NF4.
    Bobot model dicache permanen di Google Drive untuk mencegah pengunduhan berulang.
    """

    def __init__(self, config: ComputeConfig):
        super().__init__(config)
        self.tokenizer = None
        self.model = None
        self.device_name = "CUDA"

    def load_model(self) -> bool:
        if self.model is not None and self.tokenizer is not None:
            return True

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            if not torch.cuda.is_available():
                return False

            self.device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
            model_id = self.config.model_identifier
            cache_dir = self.config.model_dir

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                cache_dir=cache_dir
            )
            return True
        except Exception:
            self.tokenizer = None
            self.model = None
            return False

    def is_ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None

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

    def _apply_chat_template(self, text: str) -> str:
        messages = [
            {'role': 'system', 'content': 'Anda adalah evaluator AI yang objektif. Anda tidak memiliki bias posisi. Jawab HANYA dengan pilihan kandidat (A atau B).'},
            {'role': 'user', 'content': text}
        ]
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"<system>Anda adalah evaluator AI yang objektif.</system>\n<user>{text}</user>\n<assistant>"

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
        import torch

        if not self.is_ready():
            if not self.load_model():
                raise RuntimeError("Gagal memuat model CUDA Transformers.")

        start_time = time.time()
        raw_prompt = self._format_prompt(p_id, input_text, resp_a, resp_b)
        formatted_chat = self._apply_chat_template(raw_prompt)

        inputs = self.tokenizer([formatted_chat], return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        raw_output = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        runtime = time.time() - start_time

        del inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()

        parsed = self._parse_decision(raw_output)
        telemetry = {
            "mode": "COLAB_GPU (4-bit NF4)",
            "device_name": self.device_name,
            "ram_usage_mb": None,
            "peak_ram_mb": None
        }
        return parsed, raw_output, runtime, telemetry
