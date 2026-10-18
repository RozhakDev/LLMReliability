import os
import re
import time
import gc
from typing import Tuple, Dict, Any, Optional
import streamlit as st

from schemas import DualInferenceResult

DRIVE_MODELS = "/content/drive/MyDrive/LLMReliability/models"
if os.path.exists("/content/drive/MyDrive/LLMReliability"):
    os.environ["HF_HOME"] = DRIVE_MODELS

class DualPositionEvaluator:
    """Service inferensi dan evaluasi simetri ganda (AB/BA) untuk LLM-as-a-Judge."""

    @classmethod
    @st.cache_resource(show_spinner=False)
    def load_model(cls):
        """Memuat model Qwen3-4B-Instruct dan tokenizer satu kali dengan kuantisasi 4-bit NF4."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            return None, None

        model_id = "Qwen/Qwen3-4B-Instruct-2507"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=os.environ.get("HF_HOME"))
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=os.environ.get("HF_HOME")
        )
        return tokenizer, model

    @staticmethod
    def format_prompt(p_id: str, input_text: str, resp_a: str, resp_b: str) -> str:
        """Menyusun teks prompt masukan sesuai strategi P1, P2, atau P3 (Notebook 08)."""
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
    def apply_chat_template(tokenizer, text: str) -> str:
        """Membungkus prompt ke format template percakapan model Qwen3."""
        messages = [
            {'role': 'system', 'content': 'Anda adalah evaluator AI yang objektif. Anda tidak memiliki bias posisi. Jawab HANYA dengan pilihan kandidat (A atau B).'},
            {'role': 'user', 'content': text}
        ]
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"<system>Anda adalah evaluator AI yang objektif.</system>\n<user>{text}</user>\n<assistant>"

    @staticmethod
    def parse_decision(raw_output: str) -> str:
        """Mengekstrak karakter pilihan A atau B menggunakan batas kata regex."""
        clean = str(raw_output).strip().upper()
        match = re.search(r'\b([AB])\b', clean)
        return match.group(1) if match else 'INVALID'

    @classmethod
    def run_single_inference(cls, tokenizer, model, p_id: str, input_text: str, resp_a: str, resp_b: str) -> Tuple[str, str, float]:
        """Menjalankan inferensi satu kali untuk pasangan posisi tertentu."""
        import torch
        start_time = time.time()
        raw_prompt = cls.format_prompt(p_id, input_text, resp_a, resp_b)
        formatted_chat = cls.apply_chat_template(tokenizer, raw_prompt)

        inputs = tokenizer([formatted_chat], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        raw_output = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        runtime = time.time() - start_time

        del inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()

        parsed = cls.parse_decision(raw_output)
        return parsed, raw_output, runtime

    @classmethod
    def evaluate_symmetry(cls, p_id: str, input_text: str, cand_1: str, cand_2: str) -> DualInferenceResult:
        """
        Menjalankan pengujian simetri posisi ganda (Dual-Position Symmetry Evaluation):
        1. Posisi AB: Respons A = Kandidat 1, Respons B = Kandidat 2
        2. Posisi BA: Respons A = Kandidat 2, Respons B = Kandidat 1
        """
        tokenizer, model = cls.load_model()

        if tokenizer is None or model is None:
            time.sleep(1.2)
            parsed_ab = 'A' if len(cand_1) >= len(cand_2) else 'B'
            parsed_ba = 'A'
            logical_ab = 1 if parsed_ab == 'A' else 2
            logical_ba = 2 if parsed_ba == 'A' else 1

            return DualInferenceResult(
                parsed_AB=parsed_ab,
                raw_output_AB=parsed_ab,
                logical_decision_AB=logical_ab,
                runtime_AB=0.45,
                parsed_BA=parsed_ba,
                raw_output_BA=parsed_ba,
                logical_decision_BA=logical_ba,
                runtime_BA=0.42,
                is_consistent=(logical_ab == logical_ba),
                is_flip=(logical_ab != logical_ba),
                mode="SIMULATION_FALLBACK (No GPU)"
            )

        parsed_ab, raw_ab, time_ab = cls.run_single_inference(tokenizer, model, p_id, input_text, cand_1, cand_2)
        logical_ab = 1 if parsed_ab == 'A' else (2 if parsed_ab == 'B' else 0)

        parsed_ba, raw_ba, time_ba = cls.run_single_inference(tokenizer, model, p_id, input_text, cand_2, cand_1)
        logical_ba = 2 if parsed_ba == 'A' else (1 if parsed_ba == 'B' else 0)

        is_consistent = (logical_ab == logical_ba) and (logical_ab != 0)
        is_flip = not is_consistent

        return DualInferenceResult(
            parsed_AB=parsed_ab,
            raw_output_AB=raw_ab,
            logical_decision_AB=logical_ab,
            runtime_AB=round(time_ab, 2),
            parsed_BA=parsed_ba,
            raw_output_BA=raw_ba,
            logical_decision_BA=logical_ba,
            runtime_BA=round(time_ba, 2),
            is_consistent=is_consistent,
            is_flip=is_flip,
            mode="LIVE_GPU"
        )

def format_prompt(p_id: str, input_text: str, resp_a: str, resp_b: str) -> str:
    return DualPositionEvaluator.format_prompt(p_id, input_text, resp_a, resp_b)

def parse_decision(raw_output: str) -> str:
    return DualPositionEvaluator.parse_decision(raw_output)

def run_live_dual_inference(p_id: str, input_text: str, cand_1: str, cand_2: str) -> DualInferenceResult:
    return DualPositionEvaluator.evaluate_symmetry(p_id, input_text, cand_1, cand_2)
