from typing import Optional

import streamlit as st
from llm_reliability.domain import DataPaths
from llm_reliability.ui.theme import apply_enterprise_theme, render_html
from llm_reliability.services.data_repository import EvaluationRepository
from llm_reliability.ui.views.sidebar import render_sidebar
from llm_reliability.ui.views.executive_summary import render_executive_tab
from llm_reliability.ui.views.position_bias_explorer import render_flip_explorer_tab
from llm_reliability.ui.views.prompt_sensitivity import render_prompt_sensitivity_tab
from llm_reliability.ui.views.interactive_playground import render_playground_tab

from llm_reliability.core.constants import APP_TITLE, APP_SUBTITLE, APP_PROTOCOL_LABEL, APP_YEAR

def render_app_header() -> None:
    """Merender header aplikasi."""
    st.markdown(f"## {APP_TITLE}")
    render_html(
        f"<p style='color: var(--text-secondary); font-size: 0.92rem; margin-top: -10px; margin-bottom: 22px; letter-spacing: -0.01em;'>"
        f"{APP_SUBTITLE}"
        f"</p>"
    )

def render_app_footer(paths: Optional[DataPaths] = None) -> None:
    """Merender footer di bagian bawah aplikasi."""
    st.markdown("<br><br>", unsafe_allow_html=True)
    is_gdrive = bool(paths and paths.results_dir and "drive" in paths.results_dir.lower())
    source_label = "Google Drive (/content/drive/MyDrive/LLMReliability)" if is_gdrive else "Data Lokal (UI/data)"
    render_html(
        f"<div style='border-top: 1px solid var(--border-contrast); padding-top: 16px; display: flex; justify-content: space-between; align-items: center; font-size: 0.75rem; color: var(--text-muted); font-family: \"JetBrains Mono\", monospace;'>"
        f"<div>Sumber Data: <strong>{source_label}</strong></div>"
        f"<div>Protokol: <strong>{APP_PROTOCOL_LABEL}</strong></div>"
        f"<div>Tahun: <strong>{APP_YEAR}</strong></div>"
        f"</div>"
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
        "Analisis Bias Posisi",
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

    render_app_footer(paths)

if __name__ == "__main__":
    main()
