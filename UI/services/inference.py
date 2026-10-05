import time
import gc
from typing import Tuple, Dict, Any, Optional
import streamlit as st
from utils.json_parser import extract_json

@st.cache_resource
def load_model():
    """Load LLM and tokenizer strictly once."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    model_id = "Qwen/Qwen3-4B-Instruct-2507"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    return tokenizer, model

def format_json_prompt(p_id: str, user_input: str, resp_a: str, resp_b: str) -> str:
    """Constructs the structured CoT prompt based on the strict RCAF Framework."""
    return (
        "### PERAN\n"
        "Anda adalah Juri AI (LLM-as-a-Judge) tingkat lanjut yang sangat objektif dan analitis.\n\n"
        "### KONTEKS\n"
        "Anda harus mengevaluasi dua respons AI berdasarkan kriteria berikut:\n"
        "1. Accuracy: Apakah respons memuat informasi yang faktual dan benar?\n"
        "2. Relevance: Apakah respons menjawab instruksi secara langsung?\n"
        "3. Clarity: Apakah respons terstruktur dengan baik dan tidak bertele-tele?\n\n"
        "### INSTRUKSI\n" + user_input + "\n\n"
        "### RESPONS A\n" + resp_a + "\n\n"
        "### RESPONS B\n" + resp_b + "\n\n"
        "### AKSI\n"
        "1. Ekstrak bukti spesifik (evidence) dari kedua respons untuk setiap kriteria.\n"
        "2. Berikan skor 1-10 untuk masing-masing kriteria.\n"
        "3. Tuliskan penjelasan analitis (Chain-of-Thought).\n"
        "4. Tentukan pemenang akhir (A atau B).\n\n"
        "### FORMAT JAWABAN\n"
        "Keluarkan hasil evaluasi Anda HANYA dalam format JSON valid tanpa awalan/akhiran markdown backticks (```json ... ```).\n"
        "PENTING: JSON harus dieksekusi secara terurut. Atribut 'explanation' HARUS berada di atas sebelum 'decision'.\n"
        "Gunakan skema persis seperti berikut:\n"
        "{\n"
        "  \"explanation\": {\n"
        "    \"accuracy\": \"Penjelasan singkat\",\n"
        "    \"relevance\": \"Penjelasan singkat\",\n"
        "    \"clarity\": \"Penjelasan singkat\"\n"
        "  },\n"
        "  \"evidence\": [\n"
        "    {\n"
        "      \"criterion\": \"accuracy/relevance/clarity\",\n"
        "      \"candidate\": \"A atau B\",\n"
        "      \"evidence\": \"kutipan teks respons\",\n"
        "      \"assessment\": \"supported atau unsupported\"\n"
        "    }\n"
        "  ],\n"
        "  \"criteria\": {\n"
        "    \"accuracy\": { \"A\": 9, \"B\": 8 },\n"
        "    \"relevance\": { \"A\": 8, \"B\": 9 },\n"
        "    \"clarity\": { \"A\": 9, \"B\": 9 }\n"
        "  },\n"
        "  \"decision\": \"A\"\n"
        "}"
    )

def run_live_inference(p_id: str, user_input: str, cand_1: str, cand_2: str, pos: str) -> Tuple[Optional[Dict[str, Any]], str, float]:
    """Execute live model inference and memory cleanup."""
    import torch
    tokenizer, model = load_model()
    
    start_time = time.time()
    resp_a, resp_b = (cand_2, cand_1) if pos == "BA" else (cand_1, cand_2)
    raw_prompt = format_json_prompt(p_id, user_input, resp_a, resp_b)
    
    messages = [
        {"role": "system", "content": "You are an impartial and expert AI evaluator. Always output valid raw JSON without markdown formatting."},
        {"role": "user", "content": raw_prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=2048, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        
    raw_output = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    parsed_json = extract_json(raw_output)
    runtime = time.time() - start_time
    
    # OOM Shield
    del inputs, outputs
    torch.cuda.empty_cache()
    gc.collect()
    
    return parsed_json, raw_output, runtime
