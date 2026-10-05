import streamlit as st

def apply_enterprise_theme():
    st.set_page_config(page_title="LLM Judge Demo", layout="wide", initial_sidebar_state="expanded")
    
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        .stApp {
            font-family: 'Inter', sans-serif;
            background-color: #0e1117;
            color: #c9d1d9;
        }
        .metric-card {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 16px;
            text-align: left;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 600;
            color: #c9d1d9;
            margin-top: 8px;
        }
        .metric-label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 500;
            color: #8b949e;
        }
        /* Buttons */
        .stButton>button {
            background-color: #238636;
            color: #ffffff;
            border: 1px solid rgba(240, 246, 252, 0.1);
            border-radius: 6px;
            font-weight: 500;
        }
        .stButton>button:hover {
            background-color: #2ea043;
            border-color: rgba(240, 246, 252, 0.1);
        }
        /* Radio */
        .stRadio > div {
            padding: 8px 0;
        }
        /* Evidences block */
        .evidence-block {
            background-color: #161b22;
            border-left: 3px solid #58a6ff;
            padding: 12px 16px;
            margin-bottom: 8px;
            border-radius: 0 6px 6px 0;
            font-size: 0.9rem;
        }
        .evidence-meta {
            color: #8b949e;
            font-size: 0.8rem;
            margin-bottom: 6px;
        }
    </style>
    """, unsafe_allow_html=True)
