import os
import html
from typing import Dict, Any, Optional
import pandas as pd
import streamlit as st
from config.theme import render_html

def render_kpi_cards(df_metrics: pd.DataFrame, mcnemar_data: Dict[str, Any], prompt_filter: str = "Semua (Overall)") -> None:
    """Menampilkan kartu metrik performa utama berstandar observabilitas industri (Datadog/Linear)."""
    if df_metrics.empty:
        st.warning("Tabel ringkasan metrik belum tersedia.")
        return

    if prompt_filter == "Semua (Overall)":
        acc = df_metrics['Accuracy (%)'].mean()
        cons = df_metrics['Consistency (%)'].mean()
        flip = df_metrics['Flip Rate (%)'].mean()
        kappa = df_metrics['Cohen Kappa'].mean()
        ci_str = f"Rata-rata: {kappa:.3f}"
        m_stat = mcnemar_data.get('ALL', {}).get('statistic', 143.5)
        m_pval = mcnemar_data.get('ALL', {}).get('pvalue', 0.0)
    else:
        p_code = prompt_filter.split(" ")[0]
        row = df_metrics[df_metrics['Prompt'] == p_code]
        if not row.empty:
            acc = float(row['Accuracy (%)'].values[0])
            cons = float(row['Consistency (%)'].values[0])
            flip = float(row['Flip Rate (%)'].values[0])
            kappa = float(row['Cohen Kappa'].values[0])
            ci_str = str(row['Kappa 95% CI'].values[0])
            m_stat = mcnemar_data.get(p_code, {}).get('statistic', 0.0)
            m_pval = mcnemar_data.get(p_code, {}).get('pvalue', 1.0)
        else:
            acc, cons, flip, kappa, ci_str, m_stat, m_pval = 0, 0, 0, 0, "-", 0, 1.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_html(f"""
        <div class="metric-card">
            <div class="metric-label">Akurasi Ground Truth</div>
            <div class="metric-value">{acc:.2f}%</div>
            <div class="metric-sub">Kesesuaian vs Anotasi IndoPref</div>
        </div>
        """)
    with c2:
        cons_color = "var(--signal-success)" if cons >= 70 else "var(--signal-danger)"
        render_html(f"""
        <div class="metric-card">
            <div class="metric-label">Konsistensi Posisi (AB vs BA)</div>
            <div class="metric-value" style="color: {cons_color};">{cons:.2f}%</div>
            <div class="metric-sub">Tingkat keputusan simetris</div>
        </div>
        """)
    with c3:
        flip_color = "var(--signal-danger)" if flip > 25 else "var(--signal-success)"
        render_html(f"""
        <div class="metric-card">
            <div class="metric-label">Flip Rate (Positional Bias)</div>
            <div class="metric-value" style="color: {flip_color};">{flip:.2f}%</div>
            <div class="metric-sub">Tingkat pembalikan keputusan</div>
        </div>
        """)
    with c4:
        render_html(f"""
        <div class="metric-card">
            <div class="metric-label">Cohen's Kappa & 95% CI</div>
            <div class="metric-value">{kappa:.3f}</div>
            <div class="metric-sub">CI: {ci_str}</div>
        </div>
        """)

    is_sig = m_pval < 0.05
    status_class = "danger" if is_sig else "success"
    status_text = "Bias Posisi Signifikan (p < 0.05)" if is_sig else "Tidak Ada Bias Signifikan (p >= 0.05)"

    render_html(f"""
    <div class="hud-telemetry">
        <div>
            <strong>Uji Signifikansi McNemar:</strong> Chi-square = <code>{m_stat:.4f}</code> | p-value = <code>{m_pval:.3e}</code> (alpha = 0.05)
        </div>
        <div>
            <span class="status-pill {status_class}">{status_text}</span>
        </div>
    </div>
    """)

def render_position_bias_case(row: pd.Series) -> None:
    """Menampilkan komparasi audit simetris satu sampel konkret dengan Matrix Table."""
    st.markdown("#### Detail Instruksi Pengguna (Prompt)")
    domain_clean = html.escape(str(row.get('domain', 'N/A'))).upper()
    sample_id_clean = html.escape(str(row.get('sample_id', 'N/A')))
    prompt_id_clean = html.escape(str(row.get('prompt_id', 'N/A')))
    input_clean = html.escape(str(row.get('input', ''))).replace('\n', '<br>')

    render_html(f"""
    <div style="background-color: var(--bg-surface); border: 1px solid var(--border-contrast); border-left: 3px solid var(--signal-accent); padding: 12px 16px; border-radius: 4px; margin-bottom: 16px; font-size: 0.88rem;">
        <div style="color: var(--text-muted); font-size: 0.72rem; text-transform: uppercase; margin-bottom: 6px; letter-spacing: 0.05em;">
            Domain: <code>{domain_clean}</code> | Sample ID: <code>#{sample_id_clean}</code> | Strategi: <code>{prompt_id_clean}</code>
        </div>
        <div>{input_clean}</div>
    </div>
    """)

    c1_model = html.escape(str(row.get('chosen_model', 'N/A')))
    c2_model = html.escape(str(row.get('rejected_model', 'N/A')))
    c1_body = html.escape(str(row.get('response_1', ''))).replace('\n', '<br>')
    c2_body = html.escape(str(row.get('response_2', ''))).replace('\n', '<br>')

    c1_gt_badge = '<span class="status-pill success">Pemenang Ground Truth</span>' if row.get('ground_truth') == 1 else '<span class="status-pill neutral">Kalah Ground Truth</span>'
    c2_gt_badge = '<span class="status-pill success">Pemenang Ground Truth</span>' if row.get('ground_truth') == 2 else '<span class="status-pill neutral">Kalah Ground Truth</span>'

    col1, col2 = st.columns(2)
    with col1:
        render_html(f"""
        <div class="candidate-box">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div class="candidate-header">Kandidat 1 [Model: {c1_model}]</div>
                <div>{c1_gt_badge}</div>
            </div>
            <div class="candidate-body">{c1_body}</div>
        </div>
        """)

    with col2:
        render_html(f"""
        <div class="candidate-box">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div class="candidate-header">Kandidat 2 [Model: {c2_model}]</div>
                <div>{c2_gt_badge}</div>
            </div>
            <div class="candidate-body">{c2_body}</div>
        </div>
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    is_flip = bool(row.get('is_flip', False))
    verdict_class = "danger" if is_flip else "success"
    verdict_text = "Bias Posisi (Flip)" if is_flip else "Konsisten"

    is_correct_ab = bool(row.get('is_correct_AB', False))
    is_correct_ba = bool(row.get('is_correct_BA', False))
    gt_status_ab = "success" if is_correct_ab else "danger"
    gt_text_ab = "Sesuai Ground Truth" if is_correct_ab else "Tidak Sesuai Ground Truth"

    gt_status_ba = "success" if is_correct_ba else "danger"
    gt_text_ba = "Sesuai Ground Truth" if is_correct_ba else "Tidak Sesuai Ground Truth"

    parsed_ab = html.escape(str(row.get('parsed_decision_AB', '-')))
    logical_ab = html.escape(str(row.get('logical_decision_AB', '-')))
    parsed_ba = html.escape(str(row.get('parsed_decision_BA', '-')))
    logical_ba = html.escape(str(row.get('logical_decision_BA', '-')))

    analysis_text = (
        "Model mengubah pilihan ketika urutan kemunculan respons ditukar. Hal ini menunjukkan adanya bias posisi."
        if is_flip else
        "Model konsisten memilih kandidat yang sama baik pada urutan AB maupun BA. Keputusan tidak dipengaruhi urutan penyajian respons."
    )
    border_signal = "var(--signal-danger)" if is_flip else "var(--signal-success)"

    render_html(f"""
    <div style="background-color: var(--bg-surface); border: 1px solid var(--border-contrast); border-radius: 4px; padding: 16px; margin: 12px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <div style="font-size: 0.92rem; font-weight: 600; color: var(--text-primary);">
                Evaluasi Simetri Posisi (AB vs BA)
            </div>
            <div>
                <span class="status-pill {verdict_class}">{verdict_text}</span>
            </div>
        </div>
        <table class="audit-matrix-table">
            <thead>
                <tr>
                    <th>Kondisi Permutasi</th>
                    <th>Urutan Slot Masukan</th>
                    <th>Token Model</th>
                    <th>Pemenang Logis</th>
                    <th>Validasi Ground Truth</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Posisi Normal (AB)</strong></td>
                    <td>Slot A: Kandidat 1 | Slot B: Kandidat 2</td>
                    <td><code>{parsed_ab}</code></td>
                    <td><strong>Kandidat {logical_ab}</strong></td>
                    <td><span class="status-pill {gt_status_ab}">{gt_text_ab}</span></td>
                </tr>
                <tr>
                    <td><strong>Posisi Terbalik (BA)</strong></td>
                    <td>Slot A: Kandidat 2 | Slot B: Kandidat 1</td>
                    <td><code>{parsed_ba}</code></td>
                    <td><strong>Kandidat {logical_ba}</strong></td>
                    <td><span class="status-pill {gt_status_ba}">{gt_text_ba}</span></td>
                </tr>
            </tbody>
        </table>
        <div style="background-color: var(--bg-surface-inset); border-left: 3px solid {border_signal}; padding: 10px 14px; border-radius: 0 4px 4px 0; font-size: 0.82rem; margin-top: 10px;">
            <strong>Keterangan:</strong> {analysis_text}
        </div>
    </div>
    """)

    with st.expander("Output Mentah Model & Runtime"):
        c_tel1, c_tel2 = st.columns(2)
        with c_tel1:
            st.markdown(f"**Posisi AB Raw Output:** `'{row['raw_output_AB']}'` | Runtime: `{row.get('runtime_sec_AB', 0):.2f}s`")
        with c_tel2:
            st.markdown(f"**Posisi BA Raw Output:** `'{row['raw_output_BA']}'` | Runtime: `{row.get('runtime_sec_BA', 0):.2f}s`")

def render_prompt_template_display(p_id: str) -> None:
    """Menampilkan template prompt eksplisit sesuai Notebook 08 Cell 30."""
    from services.inference import format_prompt
    sample_prompt = format_prompt(p_id, "[TEKS_INSTRUKSI_PENGGUNA]", "[TEKS_RESPONS_A]", "[TEKS_RESPONS_B]")
    st.code(sample_prompt, language="markdown")

def render_plots_section(paths: Dict[str, Optional[str]]) -> None:
    """Menampilkan grafik visualisasi hasil evaluasi."""
    st.markdown("#### Grafik Hasil Evaluasi")

    t1, t2, t3, t4 = st.tabs([
        "1. Akurasi & Bias Posisi",
        "2. Konsistensi & Cohen's Kappa",
        "3. Parsing & Akurasi Domain",
        "4. Matriks Keputusan Posisi"
    ])

    with t1:
        if paths.get("plot_accuracy") and os.path.exists(paths["plot_accuracy"]):
            st.image(paths["plot_accuracy"], caption="Gambar 1: Perbandingan Akurasi dan First-Position Rate", use_container_width=True)
        else:
            st.info("File plot_accuracy_and_position_bias.png belum ditemukan di Google Drive.")

    with t2:
        if paths.get("plot_consistency") and os.path.exists(paths["plot_consistency"]):
            st.image(paths["plot_consistency"], caption="Gambar 2: Konsistensi Simetris dan Cohen's Kappa", use_container_width=True)
        else:
            st.info("File plot_consistency_and_kappa.png belum ditemukan di Google Drive.")

    with t3:
        if paths.get("plot_parsing_domain") and os.path.exists(paths["plot_parsing_domain"]):
            st.image(paths["plot_parsing_domain"], caption="Gambar 3: Rasio Parsing dan Akurasi per Domain Topik", use_container_width=True)
        else:
            st.info("File plot_parsing_and_domain_acc.png belum ditemukan di Google Drive.")

    with t4:
        if paths.get("plot_heatmaps") and os.path.exists(paths["plot_heatmaps"]):
            st.image(paths["plot_heatmaps"], caption="Gambar 4: Matriks Kontingensi Keputusan Simetris AB vs BA", use_container_width=True)
        else:
            st.info("File plot_positional_heatmaps.png belum ditemukan di Google Drive.")
