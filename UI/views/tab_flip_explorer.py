import streamlit as st
import pandas as pd
from config.theme import render_html
from components.metrics_view import render_position_bias_case

def _format_sample_label(row: pd.Series) -> str:
    """Helper internal murni untuk menyusun label dropdown sampel secara terstruktur dan terukur."""
    flip_tag = "[FLIP]" if row['is_flip'] else "[KONSISTEN]"
    gt_tag = "[BENAR]" if row['is_robust_correct'] else "[MISMATCH]"
    input_short = str(row['input'])[:50].replace("\n", " ").strip()
    return f"#{row['sample_id']} | {row['prompt_id']} | {row['domain']} | {flip_tag} {gt_tag} - {input_short}..."

def render_flip_explorer_tab(filtered_df: pd.DataFrame) -> None:
    """Merender Tab 2: Analisis Bias Posisi (Flip)."""
    st.markdown("### Analisis Bias Posisi (Flip)")
    render_html(
        "<p style='color: var(--text-secondary); font-size: 0.88rem; margin-top: -8px; margin-bottom: 16px;'>"
        "Analisis sampel di mana keputusan model berubah ketika urutan teks kandidat ditukar (AB vs BA)."
        "</p>"
    )

    total_matched = len(filtered_df)
    total_flips = int(filtered_df['is_flip'].sum()) if not filtered_df.empty else 0
    flip_percentage = (total_flips / total_matched * 100) if total_matched > 0 else 0.0

    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        render_html(f"""
        <div class="metric-card">
            <div class="metric-label">Sampel Tersaring</div>
            <div class="metric-value">{total_matched}</div>
            <div class="metric-sub">Sampel sesuai kriteria filter</div>
        </div>
        """)
    with col_stat2:
        flip_num_color = "var(--signal-danger)" if total_flips > 0 else "var(--text-primary)"
        render_html(f"""
        <div class="metric-card">
            <div class="metric-label">Kasus Flip</div>
            <div class="metric-value" style="color: {flip_num_color};">{total_flips}</div>
            <div class="metric-sub">Perubahan keputusan saat posisi ditukar</div>
        </div>
        """)
    with col_stat3:
        flip_color = "var(--signal-danger)" if flip_percentage > 20 else "var(--signal-success)"
        render_html(f"""
        <div class="metric-card">
            <div class="metric-label">Rasio Flip</div>
            <div class="metric-value" style="color: {flip_color};">{flip_percentage:.2f}%</div>
            <div class="metric-sub">Proporsi sampel yang flip</div>
        </div>
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("Tidak ada sampel yang memenuhi kriteria filter saat ini. Silakan sesuaikan opsi filter pada bilah samping.")
        return

    sample_labels = {_format_sample_label(row): idx for idx, row in filtered_df.iterrows()}
    selected_label = st.selectbox(
        "Pilih Sampel:",
        list(sample_labels.keys()),
        help="Pilih sampel berdasarkan ID, domain, atau status simetri."
    )

    selected_idx = sample_labels[selected_label]
    selected_row = filtered_df.loc[selected_idx]

    render_position_bias_case(selected_row)
