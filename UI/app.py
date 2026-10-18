import os
import sys

UI_DIR = os.path.dirname(os.path.abspath(__file__))
if UI_DIR not in sys.path:
    sys.path.insert(0, UI_DIR)

import streamlit as st
from config.theme import apply_enterprise_theme, render_html
from services.data_loader import EvaluationRepository
from views.sidebar import render_sidebar
from views.tab_executive import render_executive_tab
from views.tab_flip_explorer import render_flip_explorer_tab
from views.tab_prompt_sensitivity import render_prompt_sensitivity_tab
from views.tab_playground import render_playground_tab

def render_app_header() -> None:
    """Merender header aplikasi."""
    st.markdown("## LLM-as-a-Judge Reliability Dashboard")
    render_html(
        "<p style='color: var(--text-secondary); font-size: 0.92rem; margin-top: -10px; margin-bottom: 22px; letter-spacing: -0.01em;'>"
        "Evaluating Prompt Sensitivity, Position Bias, and Inter-Judge Agreement in Indonesian"
        "</p>"
    )

def render_app_footer() -> None:
    """Merender footer di bagian bawah aplikasi."""
    st.markdown("<br><br>", unsafe_allow_html=True)
    render_html(
        "<div style='border-top: 1px solid var(--border-contrast); padding-top: 16px; display: flex; justify-content: space-between; align-items: center; font-size: 0.75rem; color: var(--text-muted); font-family: \"JetBrains Mono\", monospace;'>"
        "<div>Sumber Data: <strong>Google Drive (/content/drive/MyDrive/LLMReliability)</strong></div>"
        "<div>Protokol: <strong>Permutasi Posisi (AB vs BA)</strong></div>"
        "<div>Tahun: <strong>2026</strong></div>"
        "</div>"
    )

def main() -> None:
    """Main application orchestrator."""
    apply_enterprise_theme()
    render_app_header()

    repo = EvaluationRepository()
    paths = repo.get_paths()

    criteria = render_sidebar(repo, paths)
    filtered_df = repo.filter_evaluations(criteria)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Ringkasan Evaluasi",
        "Analisis Bias Posisi (Flip)",
        "Sensitivitas Prompt & Domain",
        "Uji Simetri Interaktif"
    ])

    with tab1:
        render_executive_tab(repo, paths, criteria)

    with tab2:
        render_flip_explorer_tab(filtered_df)

    with tab3:
        render_prompt_sensitivity_tab(repo)

    with tab4:
        render_playground_tab()

    render_app_footer()

if __name__ == "__main__":
    main()
