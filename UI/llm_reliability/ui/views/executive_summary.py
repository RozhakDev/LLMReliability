import streamlit as st

from llm_reliability.ui.theme import render_html
from llm_reliability.domain import DataPaths, FilterCriteria
from llm_reliability.services.data_repository import EvaluationRepository
from llm_reliability.ui.components.metrics_view import render_kpi_cards, render_plots_section

from llm_reliability.core.constants import DEFAULT_MODEL_NAME, DATASET_NAME

def render_executive_tab(repo: EvaluationRepository, paths: DataPaths, criteria: FilterCriteria) -> None:
    """Merender Tab 1: Ringkasan Metrik Evaluasi."""
    st.markdown("### Ringkasan Metrik Evaluasi")
    stats = repo.get_dataset_statistics()
    sample_count = stats.get("sample_count", 0)
    render_html(
        f"<p style='color: var(--text-secondary); font-size: 0.88rem; margin-top: -8px; margin-bottom: 16px;'>"
        f"Hasil evaluasi model {DEFAULT_MODEL_NAME} pada {sample_count:,} sampel pengujian {DATASET_NAME}."
        f"</p>"
    )

    df_metrics = repo.load_metrics_summary()
    mcnemar_data = repo.get_mcnemar_data()

    mcnemar_dict = {k: v.to_dict() for k, v in mcnemar_data.items()}

    render_kpi_cards(df_metrics, mcnemar_dict, criteria.prompt_id)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Tabel Perbandingan Metrik")
    if not df_metrics.empty:
        st.dataframe(
            df_metrics,
            width="stretch",
            hide_index=True
        )
    else:
        st.info("Data metrik evaluasi belum tersedia.")

    st.markdown("<br>", unsafe_allow_html=True)
    render_plots_section(paths.to_dict())
