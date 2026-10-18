import streamlit as st
import pandas as pd

from config.theme import render_html
from services.data_loader import EvaluationRepository
from components.metrics_view import render_prompt_template_display

def render_prompt_sensitivity_tab(repo: EvaluationRepository) -> None:
    """Merender Tab 3: Sensitivitas Strategi Prompt & Analisis Domain."""
    st.markdown("### Sensitivitas Strategi Prompt & Analisis Domain")
    render_html(
        "<p style='color: var(--text-secondary); font-size: 0.88rem; margin-top: -8px; margin-bottom: 16px;'>"
        "Perbandingan stabilitas keputusan model pada 3 strategi prompt: P1 (Baseline Direct), P2 (Multi-Criteria), dan P3 (Debiasing RCAF)."
        "</p>"
    )

    df_stability = repo.get_prompt_stability_matrix()
    df_domain = repo.load_domain_accuracy()

    c_left, c_right = st.columns([1, 1])
    with c_left:
        st.markdown("#### Perbedaan Keputusan Antar-Strategi Prompt")
        render_html(
            "<p style='color: var(--text-muted); font-size: 0.8rem; margin-top: -6px;'>"
            "Persentase perubahan keputusan model saat mengevaluasi instruksi yang sama dengan template prompt berbeda:"
            "</p>"
        )
        if not df_stability.empty:
            st.dataframe(df_stability, use_container_width=True, hide_index=True)
        else:
            st.info("Matriks stabilitas sedang dikalkulasi.")

    with c_right:
        st.markdown("#### Akurasi Evaluator per Domain Topik")
        render_html(
            "<p style='color: var(--text-muted); font-size: 0.8rem; margin-top: -6px;'>"
            "Tingkat kecocokan pilihan model terhadap anotasi preferensi manusia pada 10 domain:"
            "</p>"
        )
        if not df_domain.empty:
            st.dataframe(df_domain, use_container_width=True, hide_index=True)
        else:
            st.info("Tabel akurasi domain belum tersedia.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Template Prompt")
    render_html(
        "<p style='color: var(--text-muted); font-size: 0.8rem; margin-top: -6px;'>"
        "Format prompt yang dikirimkan ke model evaluator:"
        "</p>"
    )
    p_inspect = st.radio("Pilih Strategi Prompt:", ["P1", "P2", "P3"], horizontal=True)
    render_prompt_template_display(p_inspect)
