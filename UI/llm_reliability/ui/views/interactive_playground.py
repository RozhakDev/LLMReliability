import os
import html
import streamlit as st
from llm_reliability.core.constants import (
    DEFAULT_MODEL_NAME,
    DEFAULT_GGUF_FILENAME,
    PROMPT_CODES,
    PROMPT_DISPLAY_NAMES,
    PLAYGROUND_SAMPLE_CASES,
)
from llm_reliability.core.settings import get_settings
from llm_reliability.ui.theme import render_html
from llm_reliability.services.evaluator import DualPositionEvaluator
from llm_reliability.services.backends.factory import BackendFactory
from llm_reliability.services.model_downloader import download_qwen3_gguf

def render_playground_tab() -> None:
    """Merender Tab 4: Pengujian Simetri Interaktif."""
    s = get_settings()
    st.markdown("### Pengujian Simetri Interaktif")
    st.markdown(
        "<p style='color: var(--text-secondary); font-size: 0.88rem; margin-top: -8px; margin-bottom: 16px;'>"
        "Uji konsistensi keputusan model saat posisi teks kandidat ditukar (AB vs BA)."
        "</p>",
        unsafe_allow_html=True
    )

    status = BackendFactory.get_backend_status()
    is_ready = status["is_ready"]
    device_name = status["device_name"]

    status_badge_class = "success" if is_ready else "neutral"
    status_badge_label = "Model Siap" if is_ready else "Model Belum Tersedia"

    render_html(f"""
    <div class="hud-telemetry">
        <div style="display: flex; gap: 20px; font-family: 'JetBrains Mono', monospace; font-size: 0.76rem;">
            <span>Model: <strong>{DEFAULT_MODEL_NAME}</strong></span>
            <span>Perangkat: <strong>{device_name}</strong></span>
            <span>Temperature: <strong>{s.temperature:.1f}</strong></span>
            <span>Evaluasi: <strong>AB & BA</strong></span>
        </div>
        <div>
            <span class="status-pill {status_badge_class}">{status_badge_label}</span>
        </div>
    </div>
    """)

    sample_labels = [c["label"] for c in PLAYGROUND_SAMPLE_CASES]
    quick_sample_opt = st.selectbox(
        "Pilih Contoh Soal:",
        sample_labels
    )

    selected_case = next(
        (c for c in PLAYGROUND_SAMPLE_CASES if c["label"] == quick_sample_opt),
        PLAYGROUND_SAMPLE_CASES[0]
    )
    default_prompt = selected_case["prompt"]
    default_c1 = selected_case["cand_1"]
    default_c2 = selected_case["cand_2"]

    p_options = [PROMPT_DISPLAY_NAMES[p] for p in PROMPT_CODES]
    p_interactive = st.selectbox(
        "Strategi Prompt:",
        p_options
    )
    p_id_play = p_interactive.split(" ")[0]

    input_user = st.text_area(
        "Pertanyaan / Instruksi:",
        value=default_prompt,
        height=100
    )

    c_cand1, c_cand2 = st.columns(2)
    with c_cand1:
        cand_1_text = st.text_area("Respons Kandidat 1:", value=default_c1, height=180)
        render_html(
            f"<div style='font-family: \"JetBrains Mono\", monospace; font-size: 0.72rem; color: var(--text-muted); text-align: right;'>"
            f"Karakter: <span class='tabular-num'>{len(cand_1_text)}</span>"
            f"</div>"
        )
    with c_cand2:
        cand_2_text = st.text_area("Respons Kandidat 2:", value=default_c2, height=180)
        render_html(
            f"<div style='font-family: \"JetBrains Mono\", monospace; font-size: 0.72rem; color: var(--text-muted); text-align: right;'>"
            f"Karakter: <span class='tabular-num'>{len(cand_2_text)}</span>"
            f"</div>"
        )

    model_ready = status["model_ready"]
    model_size = status.get("model_size_mb", 0.0)

    if not is_ready:
        render_html(f"""
        <div style="background-color: var(--bg-surface); border: 1px solid var(--border-contrast); border-left: 3px solid var(--signal-danger); border-radius: 4px; padding: 14px 16px; margin-top: 14px; margin-bottom: 16px;">
            <div style="font-weight: 600; color: var(--text-primary); font-size: 0.88rem; margin-bottom: 6px;">
                File Model Belum Tersedia
            </div>
            <div style="color: var(--text-secondary); font-size: 0.82rem; line-height: 1.5;">
                File model <code>{DEFAULT_GGUF_FILENAME}</code> belum ditemukan di folder <code>UI/data/models/</code>.
                Silakan unduh file model untuk menjalankan uji interaktif.

            </div>
        </div>
        """)

        c_dl1, c_dl2 = st.columns([1, 1])
        with c_dl1:
            if st.button("Unduh File Model (~2.33 GB)", key="btn_download_model_ui", width="stretch"):
                p_bar = st.progress(0.0)
                p_text = st.empty()
                def _ui_cb(frac: float, msg: str):
                    p_bar.progress(frac)
                    p_text.text(msg)
                try:
                    with st.spinner("Mengunduh file model..."):
                        saved = download_qwen3_gguf(progress_callback=_ui_cb)
                        BackendFactory.reset_cache()
                        st.success(f"File model berhasil diunduh: {saved}")
                        st.rerun()
                except Exception as err:
                    st.error(f"Gagal mengunduh: {err}")

        with c_dl2:
            with st.expander("Perintah Unduh Manual via Terminal (CLI)"):
                st.code("UI\\venv\\Scripts\\python.exe -m llm_reliability.services.model_downloader", language="bash")
                st.caption("Jalankan perintah ini di terminal jika ingin mengunduh di latar belakang.")

        custom_path = st.text_input(
            "Atau masukkan path file GGUF yang sudah ada di komputer Anda:",
            placeholder=r"C:\Users\USER\Downloads\qwen3-4b-instruct-2507-q4_k_m.gguf",
            key="custom_gguf_path"
        )
        if custom_path and os.path.exists(custom_path) and os.path.getsize(custom_path) >= s.expected_min_bytes:
            os.environ["LOCAL_GGUF_PATH"] = custom_path
            BackendFactory.reset_cache()
            st.success(f"File model ditemukan: {custom_path}")
            st.rerun()

        st.caption("Tombol evaluasi aktif setelah file model tersedia.")
    else:
        render_html(f"""
        <div style="background-color: var(--bg-surface); border: 1px solid var(--border-contrast); border-left: 3px solid var(--signal-success); border-radius: 4px; padding: 10px 14px; margin-top: 10px; margin-bottom: 14px; font-size: 0.8rem; color: var(--text-secondary);">
            Model aktif: <code>{DEFAULT_MODEL_NAME} GGUF Q4_K_M</code> (<span class="tabular-num">{model_size:.1f} MB</span>) | Siap inferensi via CPU.
        </div>
        """)

    if st.button("Jalankan Evaluasi (AB & BA)", width="stretch", disabled=not is_ready):
        if not input_user.strip() or not cand_1_text.strip() or not cand_2_text.strip():
            st.warning("Seluruh kolom teks instruksi dan kandidat respons wajib diisi.")
            return

        with st.spinner("Menjalankan inferensi model (urutan AB dan BA)..."):
            try:
                eval_res = DualPositionEvaluator.evaluate_symmetry(p_id_play, input_user, cand_1_text, cand_2_text)
            except Exception as err:
                st.error(f"Terjadi kesalahan saat inferensi model: {err}")
                return

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Hasil Evaluasi")

            is_flip_res = eval_res.is_flip
            status_pill_class = "danger" if is_flip_res else "success"
            status_pill_label = "Bias Posisi (Flip)" if is_flip_res else "Konsisten"

            total_runtime = eval_res.runtime_AB + eval_res.runtime_BA
            mode_clean = html.escape(str(eval_res.mode))
            device_clean = html.escape(str(eval_res.device_name))
            ram_info = f"RAM: <span class='tabular-num'>{eval_res.ram_usage_mb:.0f} MB</span> | " if eval_res.ram_usage_mb is not None else ""
            parsed_ab_clean = html.escape(str(eval_res.parsed_AB))
            logical_ab_clean = html.escape(str(eval_res.logical_decision_AB))
            parsed_ba_clean = html.escape(str(eval_res.parsed_BA))
            logical_ba_clean = html.escape(str(eval_res.logical_decision_BA))

            render_html(f"""
            <div class="hud-telemetry">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">
                    Waktu: <span class="tabular-num"><strong>{total_runtime:.2f}s</strong></span> |
                    Perangkat: <code>{device_clean}</code> |
                    {ram_info}Mode: <code>{mode_clean}</code>
                </div>
                <div>
                    <span class="status-pill {status_pill_class}">{status_pill_label}</span>
                </div>
            </div>
            """)

            analysis_text = (
                "Keputusan model berubah saat urutan ditukar. Terindikasi bias posisi."
                if is_flip_res else
                "Model konsisten memilih kandidat yang sama pada urutan AB maupun BA."
            )
            border_signal = "var(--signal-danger)" if is_flip_res else "var(--signal-success)"

            render_html(f"""
            <div style="background-color: var(--bg-surface); border: 1px solid var(--border-contrast); border-radius: 4px; padding: 16px; margin-bottom: 16px;">
                <div style="font-size: 0.92rem; font-weight: 600; color: var(--text-primary); margin-bottom: 8px;">
                    Hasil Evaluasi (AB vs BA)
                </div>
                <table class="audit-matrix-table">
                    <thead>
                        <tr>
                            <th>Urutan</th>
                            <th>Posisi Input</th>
                            <th>Pilihan Model</th>
                            <th>Hasil Evaluasi</th>
                            <th>Waktu</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Urutan AB</strong></td>
                            <td>A: Kandidat 1 | B: Kandidat 2</td>
                            <td><code>{parsed_ab_clean}</code></td>
                            <td><strong>Kandidat {logical_ab_clean}</strong></td>
                            <td class="tabular-num">{eval_res.runtime_AB:.2f}s</td>
                        </tr>
                        <tr>
                            <td><strong>Urutan BA</strong></td>
                            <td>A: Kandidat 2 | B: Kandidat 1</td>
                            <td><code>{parsed_ba_clean}</code></td>
                            <td><strong>Kandidat {logical_ba_clean}</strong></td>
                            <td class="tabular-num">{eval_res.runtime_BA:.2f}s</td>
                        </tr>
                    </tbody>
                </table>
                <div style="background-color: var(--bg-surface-inset); border-left: 3px solid {border_signal}; padding: 12px 14px; border-radius: 0 4px 4px 0; font-size: 0.82rem; margin-top: 10px; line-height: 1.5;">
                    <strong>Keterangan:</strong> {analysis_text}
                </div>
            </div>
            """)

            with st.expander("Lihat Output Mentah (Raw Output)"):
                st.code(f"Keluaran Mentah AB:\n{eval_res.raw_output_AB}\n\nKeluaran Mentah BA:\n{eval_res.raw_output_BA}", language="text")
