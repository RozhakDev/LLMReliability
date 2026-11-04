import streamlit as st

from llm_reliability.ui.theme import render_html
from llm_reliability.services.data_repository import EvaluationRepository
from llm_reliability.ui.components.metrics_view import render_prompt_template_display

from llm_reliability.core.constants import PROMPT_CODES

def render_prompt_sensitivity_tab(repo: EvaluationRepository) -> None:
    """Merender Tab 3: Sensitivitas Strategi Prompt & Analisis Domain."""
    st.markdown("### Sensitivitas Strategi Prompt & Analisis Domain")
    render_html(
        "<p style='color: var(--text-secondary); font-size: 0.88rem; margin-top: -8px; margin-bottom: 16px;'>"
        "Perbandingan konsistensi keputusan model pada strategi prompt P1, P2, dan P3."
        "</p>"
    )

    df_stability = repo.get_prompt_stability_matrix()
    df_domain = repo.load_domain_accuracy()

    c_left, c_right = st.columns([1, 1])
    with c_left:
        st.markdown("#### Perubahan Keputusan Antar-Prompt")
        render_html(
            "<p style='color: var(--text-muted); font-size: 0.8rem; margin-top: -6px;'>"
            "Persentase perbedaan pilihan model saat prompt diubah:"
            "</p>"
        )
        if not df_stability.empty:
            st.dataframe(df_stability, width="stretch", hide_index=True)
        else:
            st.info("Matriks stabilitas sedang dikalkulasi.")

    with c_right:
        st.markdown("#### Akurasi per Domain")
        render_html(
            "<p style='color: var(--text-muted); font-size: 0.8rem; margin-top: -6px;'>"
            "Persentase kesesuaian pilihan model dengan ground truth per domain:"
            "</p>"
        )
        if not df_domain.empty:
            st.dataframe(df_domain, width="stretch", hide_index=True)
        else:
            st.info("Tabel akurasi domain belum tersedia.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Template Prompt")
    render_html(
        "<p style='color: var(--text-muted); font-size: 0.8rem; margin-top: -6px;'>"
        "Format prompt yang dikirimkan ke model evaluator:"
        "</p>"
    )
    p_inspect = st.radio("Pilih Strategi Prompt:", PROMPT_CODES, horizontal=True)
    render_prompt_template_display(p_inspect)
