import streamlit as st
import json
import time
import os
import re

st.set_page_config(page_title="LLM Judge Demo", layout="wide", initial_sidebar_state="expanded")

# --- CSS Styling (Glassmorphism & Premium UI) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
        text-align: center;
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: #58a6ff;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #58a6ff, #79c0ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 10px;
    }
    .metric-label {
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        color: #8b949e;
    }
    .stButton>button {
        background: linear-gradient(90deg, #238636, #2ea043);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2ea043, #3fb950);
        box-shadow: 0 4px 12px rgba(46, 160, 67, 0.4);
        border: none;
    }
    .stRadio > div {
        background: rgba(255, 255, 255, 0.05);
        padding: 10px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to extract JSON
def extract_json(raw_text):
    if isinstance(raw_text, dict):
        return raw_text
    try:
        # Bersihkan tag markdown
        cleaned_text = raw_text.replace("```json", "").replace("```", "").strip()
        json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
        text_to_parse = json_match.group(0) if json_match else cleaned_text
        
        try:
            return json.loads(text_to_parse)
        except json.JSONDecodeError:
            # Fallback 1: Coba ubah single quotes ke double quotes (khas error LLM kecil)
            text_to_parse_fixed = text_to_parse.replace("'", '"')
            return json.loads(text_to_parse_fixed)
    except Exception as e:
        return None

# ================= UI LAYOUT =================

st.markdown("<h1 style='text-align: center; margin-bottom: 5px;'>⚖️ LLM-as-a-Judge</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #8b949e; font-weight: 400; margin-bottom: 30px;'>Explainability Demonstrator with Structured JSON Output</h4>", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=50)
    st.header("⚙️ Configuration")
    
    op_mode = st.radio("🔄 Operating Mode", ["Mode 1: Replay (Fast/Cache)", "Mode 2: Live Inference (GPU)"])
    position_sel = st.radio("Candidate Position (Bias Test)", ["AB (Normal)", "BA (Swapped)"])
    pos = position_sel.split(" ")[0]
    
    st.markdown("---")
    st.markdown("💡 **Tip**: Ubah posisi ke `BA` untuk melihat apakah model mengganti pilihannya secara rasional atau hanya terpengaruh bias posisi.")

def render_results(parsed_json, pos, raw_text=""):
    if not parsed_json:
        st.error("JSON Parsing Failed. The model did not strictly follow the JSON schema.")
        with st.expander("Show Raw Output"):
            st.code(raw_text)
        return
        
    winner = parsed_json.get("decision", "Unknown")
    
    # Symmetrical Logic Mapping
    if pos == "BA":
        if winner == "A":
            logical_winner = "Candidate 2 (diposisikan sbg A)"
            color = "#d2a8ff"
        elif winner == "B":
            logical_winner = "Candidate 1 (diposisikan sbg B)"
            color = "#79c0ff"
        else:
            logical_winner = "INVALID"
            color = "#ff7b72"
    else:
        if winner == "A":
            logical_winner = "Candidate 1 (A)"
            color = "#79c0ff"
        elif winner == "B":
            logical_winner = "Candidate 2 (B)"
            color = "#d2a8ff"
        else:
            logical_winner = "INVALID"
            color = "#ff7b72"
    
    st.markdown(f"<h2 style='text-align: center; color: {color}; margin-bottom: 30px;'>🏆 Final Decision: {logical_winner}</h2>", unsafe_allow_html=True)
    
    c_acc = parsed_json.get("criteria", {}).get("accuracy", {})
    c_rel = parsed_json.get("criteria", {}).get("relevance", {})
    c_clar = parsed_json.get("criteria", {}).get("clarity", {})
    
    c1, c2, c3 = st.columns(3)
    for col, title, data in zip([c1, c2, c3], ["Accuracy", "Relevance", "Clarity"], [c_acc, c_rel, c_clar]):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{title}</div>
                <div class="metric-value">A: {data.get('A','-')} <span style='font-weight: 300; font-size: 1.5rem; color:#8b949e;'>|</span> B: {data.get('B','-')}</div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    exp = parsed_json.get("explanation", {})
    
    with st.container():
        st.markdown("### 📝 Detailed Explanations")
        t1, t2, t3 = st.tabs(["🎯 Accuracy", "🔗 Relevance", "📖 Clarity"])
        with t1:
            st.info(exp.get("accuracy", "Tidak ada penjelasan akurasi."))
        with t2:
            st.info(exp.get("relevance", "Tidak ada penjelasan relevansi."))
        with t3:
            st.info(exp.get("clarity", "Tidak ada penjelasan kejelasan."))
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📑 Traceable Evidence")
    evidences = parsed_json.get("evidence", [])
    if evidences:
        for ev in evidences:
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.05); border-left: 4px solid #3fb950; padding: 15px; margin-bottom: 10px; border-radius: 4px;">
                <strong>Criterion:</strong> <span style="color:#79c0ff;">{str(ev.get('criterion')).upper()}</span> &nbsp;|&nbsp; 
                <strong>Candidate:</strong> {ev.get('candidate')} &nbsp;|&nbsp; 
                <strong>Assessment:</strong> {ev.get('assessment')}
                <br><br>
                <em>"{ev.get('evidence')}"</em>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.write("Tidak ada bukti kutipan yang diekstrak.")

if "Replay" in op_mode:
    # --- MODE 1: REPLAY ---
    st.info("🔄 **Mode Replay Aktif**: Memuat precomputed explainability (Sangat Cepat & Tanpa GPU Memory).")
    
    subset_path = "data/processed/explainability_subset.jsonl"
    if not os.path.exists(subset_path):
        st.error(f"File {subset_path} belum ada! Silakan jalankan Notebook 06_Explainability_Generator terlebih dahulu.")
    else:
        # Load JSONL
        records = []
        with open(subset_path, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
                
        sample_options = {f"Sample #{r['sample_id']} - {r['prompt'][:40]}...": r for r in records}
        selected_sample_key = st.selectbox("📂 Pilih Sample IndoPref", list(sample_options.keys()))
        selected = sample_options[selected_sample_key]
        
        colA, colB = st.columns(2)
        with colA:
            st.text_area("👤 User Prompt", value=selected["prompt"], height=100, disabled=True)
            st.text_area("🤖 Candidate 1", value=selected["cand_1"], height=200, disabled=True)
        with colB:
            st.text_area("👥 Human Preference Ground Truth", value=selected["human_preference"], height=100, disabled=True)
            st.text_area("🤖 Candidate 2", value=selected["cand_2"], height=200, disabled=True)
            
        st.markdown("---")
        
        # Decide which precomputed output to show based on position
        raw_output = selected["json_output_ba"] if pos == "BA" else selected["json_output_ab"]
        
        if type(raw_output) is str:
            parsed = extract_json(raw_output)
        else:
            parsed = extract_json(raw_output) if raw_output else None
            
        render_results(parsed, pos, str(raw_output))
        st.success("✅ Rendered from cache instaneously.")

else:
    # --- MODE 2: LIVE INFERENCE ---
    st.warning("⚡ **Mode Live Inference Aktif**: Menggunakan memori GPU untuk evaluasi teks kustom.")
    
    @st.cache_resource
    def load_model():
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
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
        return tokenizer, model

    with st.spinner("Memuat model Qwen3-4B... (bisa memakan waktu 1 menit)"):
        tokenizer, model = load_model()
        
    p_id = st.sidebar.selectbox("Prompt Template", ["P1 (Direct)", "P2 (Criteria-based)", "P3 (Structured)"]).split(" ")[0]

    col1, col2 = st.columns([1, 1])
    with col1:
        user_input = st.text_area("👤 User Prompt (Input Question)", height=120, value="Jelaskan perbedaan antara mitosis dan meiosis.")
    with col2:
        st.info("Ketik respons kustom Anda di bawah. Model akan mengevaluasi secara real-time.")

    colA, colB = st.columns(2)
    with colA:
        cand_1 = st.text_area("🤖 Candidate 1", height=200, value="Mitosis terjadi pada sel tubuh untuk pertumbuhan, menghasilkan 2 sel anak identik. Meiosis terjadi pada sel kelamin, menghasilkan 4 sel berbeda.")
    with colB:
        cand_2 = st.text_area("🤖 Candidate 2", height=200, value="Mitosis membelah 1 kali, meiosis 2 kali. Mitosis hasilnya sama persis, kalau meiosis beda-beda sedikit.")

    if st.button("🚀 Execute Live Inference", use_container_width=True):
        if not user_input or not cand_1 or not cand_2:
            st.warning("Mohon isi form teks!")
        else:
            def format_json_prompt(p_id, user_input, resp_A, resp_B):
                base = "Bandingkan Response A dan Response B. Pilih respons yang lebih baik."
                if p_id == "P2": base = "Bandingkan berdasarkan relevansi dan kejelasan."
                return f"{base}\nAnda HARUS mengevaluasi berdasarkan tiga kriteria: Accuracy/Correctness, Relevance, dan Clarity. Keluarkan hasil HANYA dalam format JSON valid dengan struktur: {{\"decision\": \"A atau B\", \"criteria\": {{\"accuracy\": {{\"A\": 9, \"B\": 8}}, \"relevance\": {{\"A\": 8, \"B\": 7}}, \"clarity\": {{\"A\": 7, \"B\": 7}}}}, \"explanation\": {{\"accuracy\": \"...\", \"relevance\": \"...\", \"clarity\": \"...\"}}, \"evidence\": [{{\"criterion\": \"...\", \"candidate\": \"A\", \"evidence\": \"...\", \"assessment\": \"supported\"}}]}}\n[INPUT CONTEXT]\nInput: {user_input}\nResponse A:\n{resp_A}\nResponse B:\n{resp_B}"

            with st.spinner("🧠 Qwen3-4B is judging..."):
                start_time = time.time()
                
                resp_A, resp_B = (cand_2, cand_1) if pos == "BA" else (cand_1, cand_2)
                raw_prompt = format_json_prompt(p_id, user_input, resp_A, resp_B)
                
                messages = [
                    {"role": "system", "content": "You are a rigid AI evaluator. Always output valid raw JSON."},
                    {"role": "user", "content": raw_prompt}
                ]
                
                import torch
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer([text], return_tensors="pt").to(model.device)
                
                with torch.inference_mode():
                    outputs = model.generate(**inputs, max_new_tokens=2048, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                    
                raw_output = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                parsed_json = extract_json(raw_output)
                runtime = time.time() - start_time
                
                st.markdown("---")
                render_results(parsed_json, pos, raw_output)
                if parsed_json:
                    st.success(f"Inference completed in {runtime:.2f} seconds.")
