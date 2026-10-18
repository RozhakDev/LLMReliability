import streamlit as st
import pandas as pd
from typing import Dict, Any

from config.theme import render_html
from schemas import DataPaths, FilterCriteria
from services.data_loader import EvaluationRepository
from components.metrics_view import render_kpi_cards, render_plots_section

def render_executive_tab(repo: EvaluationRepository, paths: DataPaths, criteria: FilterCriteria) -> None:
    """Merender Tab 1: Ringkasan Metrik Evaluasi."""
    st.markdown("### Ringkasan Metrik Evaluasi")
    render_html(
        "<p style='color: var(--text-secondary); font-size: 0.88rem; margin-top: -8px; margin-bottom: 16px;'>"
        "Hasil evaluasi model Qwen3-4B-Instruct pada 519 sampel pengujian IndoPref."
        "</p>"
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
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Data metrik evaluasi belum tersedia.")

    st.markdown("<br>", unsafe_allow_html=True)
    render_plots_section(paths.to_dict())
