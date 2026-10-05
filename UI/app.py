import os
import json
import streamlit as st

os.environ["HF_HOME"] = "/content/drive/MyDrive/LLMReliability/models"

from config.theme import apply_enterprise_theme
from components.metrics_view import render_results
from services.inference import run_live_inference, load_model
from utils.json_parser import extract_json

def main():
    apply_enterprise_theme()
    
    st.markdown("## LLM-as-a-Judge")
    st.markdown("Explainability Demonstrator with Structured JSON Output")
    st.markdown("---")
    
    with st.sidebar:
        # 1. Branding & Identity
        st.markdown("## LLM Reliability")
        st.markdown("<p style='color: #8b949e; font-size: 0.85rem; margin-top: -15px; margin-bottom: 20px;'>v1.0.0 | Evaluation Engine</p>", unsafe_allow_html=True)
        st.divider()
        
        st.markdown("#### Execution Context")
        op_mode = st.radio(
            "Operating Mode", 
            ["Replay (Fast/Cache)", "Live Inference (GPU)"], 
            help="Replay reads precomputed evaluations from disk. Live executes real-time inference on Qwen3-4B."
        )
        st.divider()
        
        st.markdown("#### Bias Analytics")
        position_sel = st.radio(
            "Candidate Position", 
            ["AB (Normal)", "BA (Swapped)"], 
            help="Swap the position of Candidate 1 and 2 in the prompt to evaluate the model's position bias."
        )
        pos = position_sel.split(" ")[0]
        st.divider()
        
        st.markdown(
            "<div style='margin-top: 50px; color: #8b949e; font-size: 0.8rem;'>"
            "<strong>System Status</strong>: Online<br>"
            "<strong>Environment</strong>: Secure Colab<br>"
            "<strong>Access</strong>: Authorized"
            "</div>", 
            unsafe_allow_html=True
        )

    if "Replay" in op_mode:
        st.info("Replay Mode Active: Loading precomputed explainability logs.")
        
        subset_path = "data/processed/explainability_subset.jsonl"
        if not os.path.exists(subset_path):
            st.error(f"File {subset_path} not found. Run Explainability Generator first.")
        else:
            records = []
            with open(subset_path, "r", encoding="utf-8") as f:
                for line in f:
                    records.append(json.loads(line))
                    
            sample_options = {f"Sample #{r['sample_id']} - {r['prompt'][:40]}...": r for r in records}
            selected_sample_key = st.selectbox("Select Sample", list(sample_options.keys()))
            selected = sample_options[selected_sample_key]
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.text_area("User Prompt", value=selected["prompt"], height=100, disabled=True)
                st.text_area("Candidate 1", value=selected["cand_1"], height=200, disabled=True)
            with col_b:
                st.text_area("Human Preference Ground Truth", value=selected["human_preference"], height=100, disabled=True)
                st.text_area("Candidate 2", value=selected["cand_2"], height=200, disabled=True)
                
            st.markdown("---")
            
            raw_output = selected["json_output_ba"] if pos == "BA" else selected["json_output_ab"]
            
            if isinstance(raw_output, str):
                parsed = extract_json(raw_output)
            else:
                parsed = extract_json(raw_output) if raw_output else None
                
            render_results(parsed, pos, str(raw_output))
            st.success("Rendered from cache.")

    else:
        st.warning("Live Inference Mode Active: GPU resources required.")
        
        with st.spinner("Loading Qwen3-4B model..."):
            load_model()
            
        p_id = st.sidebar.selectbox("Prompt Template", ["P1 (Direct)", "P2 (Criteria-based)", "P3 (Structured)"]).split(" ")[0]

        col1, col2 = st.columns([1, 1])
        with col1:
            user_input = st.text_area("User Prompt (Input Question)", height=120, value="Jelaskan perbedaan antara mitosis dan meiosis.")
        with col2:
            st.info("Input custom responses below for real-time evaluation.")

        col_a, col_b = st.columns(2)
        with col_a:
            cand_1 = st.text_area("Candidate 1", height=200, value="Mitosis terjadi pada sel tubuh untuk pertumbuhan, menghasilkan 2 sel anak identik. Meiosis terjadi pada sel kelamin, menghasilkan 4 sel berbeda.")
        with col_b:
            cand_2 = st.text_area("Candidate 2", height=200, value="Mitosis membelah 1 kali, meiosis 2 kali. Mitosis hasilnya sama persis, kalau meiosis beda-beda sedikit.")

        if st.button("Execute Live Inference", use_container_width=True):
            if not user_input or not cand_1 or not cand_2:
                st.warning("All text fields are required.")
            else:
                with st.spinner("Model is evaluating..."):
                    parsed_json, raw_output, runtime = run_live_inference(p_id, user_input, cand_1, cand_2, pos)
                    
                    st.markdown("---")
                    render_results(parsed_json, pos, raw_output)
                    if parsed_json:
                        st.success(f"Inference completed in {runtime:.2f} seconds.")

if __name__ == "__main__":
    main()
