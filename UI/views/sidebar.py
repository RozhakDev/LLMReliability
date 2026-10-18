import streamlit as st
from config.theme import render_html
from schemas import FilterCriteria, DataPaths
from services.data_loader import EvaluationRepository

def render_sidebar(repo: EvaluationRepository, paths: DataPaths) -> FilterCriteria:
    """Merender bilah samping (sidebar) dan mengembalikan objek typed FilterCriteria."""
    with st.sidebar:
        st.markdown("### Evaluasi & Filter")
        render_html(
            "<p style='color: var(--text-muted); font-size: 0.8rem; margin-top: -10px; margin-bottom: 14px;'>"
            "Model: Qwen3-4B-Instruct"
            "</p>"
        )

        is_gdrive = bool(paths.results_dir and "drive" in paths.results_dir.lower())
        drive_badge_class = "success" if is_gdrive else "neutral"
        drive_status_label = "Google Drive" if is_gdrive else "Lokal"

        render_html(f"""
        <div style="margin-bottom: 16px;">
            <span class="status-pill {drive_badge_class}">{drive_status_label}</span>
        </div>
        """)

        render_html("""
        <div style="background-color: var(--bg-surface); border: 1px solid var(--border-contrast); border-radius: 4px; padding: 12px; font-size: 0.8rem; margin-bottom: 18px;">
            <div style="color: var(--text-muted); font-size: 0.72rem; text-transform: uppercase; margin-bottom: 4px;">Dataset</div>
            <div style="font-weight: 600; color: var(--text-primary); font-size: 0.9rem; font-family: 'JetBrains Mono', monospace;">IndoPref (519 Sampel)</div>
            <div style="color: var(--text-secondary); font-size: 0.75rem; margin-top: 4px;">Total Evaluasi: <span class="tabular-num">3,114</span></div>
        </div>
        """)

        st.markdown("#### Parameter Filter")
        prompt_filter = st.selectbox(
            "Strategi Prompt:",
            ["Semua (Overall)", "P1 (Baseline Direct)", "P2 (Multi-Criteria)", "P3 (Debiasing RCAF)"],
            help="Pilih strategi prompt yang dievaluasi."
        )

        all_domains = ["Semua Domain"]
        all_domains.extend(repo.get_available_domains())
        domain_filter = st.selectbox("Domain Topik:", all_domains)

        symmetry_filter = st.selectbox(
            "Status Simetri:",
            ["Semua Kasus", "Hanya Kasus Flip (Bias Posisi)", "Hanya Kasus Konsisten"],
            help="Filter berdasarkan konsistensi pilihan saat posisi respons ditukar."
        )

        gt_filter = st.selectbox(
            "Kesesuaian Ground Truth:",
            ["Semua Hasil", "Sesuai Ground Truth (Kedua Posisi)", "Tidak Sesuai Ground Truth"],
            help="Filter berdasarkan kesesuaian dengan anotasi preferensi manusia."
        )

        st.divider()
        render_html(
            "<div style='color: var(--text-muted); font-size: 0.75rem; line-height: 1.6; font-family: \"JetBrains Mono\", monospace;'>"
            "Model: Qwen3-4B-Instruct<br>"
            "Metode: Permutasi AB/BA<br>"
            "Dataset: IndoPref"
            "</div>"
        )

    return FilterCriteria(
        prompt_id=prompt_filter,
        domain=domain_filter,
        symmetry_filter=symmetry_filter,
        gt_filter=gt_filter
    )
