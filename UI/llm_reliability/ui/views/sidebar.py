import streamlit as st
from llm_reliability.ui.theme import render_html
from llm_reliability.core.constants import (
    DEFAULT_MODEL_NAME,
    DATASET_NAME,
    PROMPT_FILTER_OPTIONS,
    SYMMETRY_FILTER_OPTIONS,
    GT_FILTER_OPTIONS,
    DOMAIN_ALL,
    APP_PROTOCOL_LABEL
)
from llm_reliability.domain import FilterCriteria, DataPaths
from llm_reliability.services.data_repository import EvaluationRepository

def render_sidebar(repo: EvaluationRepository, paths: DataPaths) -> FilterCriteria:
    """Merender bilah samping (sidebar) dan mengembalikan objek typed FilterCriteria."""
    with st.sidebar:
        st.markdown("### Filter Evaluasi")
        render_html(
            f"<p style='color: var(--text-muted); font-size: 0.8rem; margin-top: -10px; margin-bottom: 14px;'>"
            f"Model: {DEFAULT_MODEL_NAME}"
            f"</p>"
        )

        is_gdrive = bool(paths.results_dir and "drive" in paths.results_dir.lower())
        drive_badge_class = "success" if is_gdrive else "neutral"
        drive_status_label = "Google Drive" if is_gdrive else "Data Lokal (UI/data)"

        render_html(f"""
        <div style="margin-bottom: 16px;">
            <span class="status-pill {drive_badge_class}">{drive_status_label}</span>
        </div>
        """)

        stats = repo.get_dataset_statistics()
        sample_count = stats.get("sample_count", 0)
        total_evals = stats.get("total_evaluations", 0)

        render_html(f"""
        <div style="background-color: var(--bg-surface); border: 1px solid var(--border-contrast); border-radius: 4px; padding: 12px; font-size: 0.8rem; margin-bottom: 18px;">
            <div style="color: var(--text-muted); font-size: 0.72rem; text-transform: uppercase; margin-bottom: 4px;">Dataset</div>
            <div style="font-weight: 600; color: var(--text-primary); font-size: 0.9rem; font-family: 'JetBrains Mono', monospace;">{DATASET_NAME} ({sample_count:,} Sampel)</div>
            <div style="color: var(--text-secondary); font-size: 0.75rem; margin-top: 4px;">Total Evaluasi: <span class="tabular-num">{total_evals:,}</span></div>
        </div>
        """)

        st.markdown("#### Filter Data")
        prompt_filter = st.selectbox(
            "Strategi Prompt:",
            PROMPT_FILTER_OPTIONS,
            help="Pilih strategi prompt yang dievaluasi."
        )

        all_domains = [DOMAIN_ALL]
        all_domains.extend(repo.get_available_domains())
        domain_filter = st.selectbox("Domain:", all_domains)

        symmetry_filter = st.selectbox(
            "Status Simetri:",
            SYMMETRY_FILTER_OPTIONS,
            help="Filter berdasarkan konsistensi pilihan saat posisi respons ditukar."
        )

        gt_filter = st.selectbox(
            "Kesesuaian Ground Truth:",
            GT_FILTER_OPTIONS,
            help="Filter berdasarkan kesesuaian dengan anotasi preferensi manusia."
        )

        st.divider()
        render_html(f"""
        <div style='color: var(--text-muted); font-size: 0.75rem; line-height: 1.6; font-family: "JetBrains Mono", monospace;'>
            Model: {DEFAULT_MODEL_NAME}<br>
            Metode: {APP_PROTOCOL_LABEL}<br>
            Dataset: {DATASET_NAME}
        </div>
        """)

    return FilterCriteria(
        prompt_id=prompt_filter,
        domain=domain_filter,
        symmetry_filter=symmetry_filter,
        gt_filter=gt_filter
    )
