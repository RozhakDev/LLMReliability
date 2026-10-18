import html
import streamlit as st
from config.theme import render_html
from services.inference import DualPositionEvaluator

def render_playground_tab() -> None:
    """Merender Tab 4: Pengujian Simetri Interaktif."""
    st.markdown("### Pengujian Simetri Interaktif")
    st.markdown(
        "<p style='color: var(--text-secondary); font-size: 0.88rem; margin-top: -8px; margin-bottom: 16px;'>"
        "Uji konsistensi keputusan model saat posisi teks kandidat ditukar (AB vs BA)."
        "</p>",
        unsafe_allow_html=True
    )

    render_html("""
    <div class="hud-telemetry">
        <div style="display: flex; gap: 20px; font-family: 'JetBrains Mono', monospace; font-size: 0.76rem;">
            <span>Model: <strong>Qwen3-4B-Instruct</strong></span>
            <span>Suhu: <strong>0.0 (Greedy)</strong></span>
            <span>Evaluasi: <strong>Posisi AB & BA</strong></span>
        </div>
        <div>
            <span class="status-pill neutral">Siap</span>
        </div>
    </div>
    """)

    quick_sample_opt = st.selectbox(
        "Muat Contoh Sampel Cepat (Opsional):",
        [
            "Ketik Masukan Kustom Mandiri",
            "Kasus Logika & Fisika: Penambahan Kecepatan dan Beban Rem",
            "Kasus Rekayasa Perangkat Lunak: Exception Handling Python",
            "Kasus Sastra: Penulisan Deskripsi Alam Pesisir"
        ]
    )

    default_prompt = "Jelaskan efek penambahan kecepatan kendaraan secara tiba-tiba terhadap kinerja rem."
    default_c1 = "Ketika kendaraan dipercepat tiba-tiba, energi kinetik kendaraan meningkat pesat. Akibatnya, sistem rem harus mengubah energi kinetik yang lebih besar tersebut menjadi energi panas melalui gesekan, yang menyebabkan beban termal dan mekanis pada rem meningkat drastis."
    default_c2 = "Jika kendaraan ditambah kecepatannya secara mendadak, rem akan bekerja lebih keras dan bannya bisa rusak karena panas dan aus."

    if "Kasus Logika & Fisika" in quick_sample_opt:
        default_prompt = "Jika Anda menambah kecepatan kendaraan tiba-tiba, apa efeknya pada rem?"
        default_c1 = "Akselerasi cepat meningkatkan torsi mesin dan energi kinetik. Untuk memperlambat kendaraan, sistem pengereman memerlukan tekanan hidrolik lebih tinggi dan gesekan bantalan rem yang jauh lebih intensif, meningkatkan suhu rem secara signifikan."
        default_c2 = "Efeknya adalah rem jadi lebih panas, ban lebih cepat aus, dan rem bisa mengalami kerusakan mendadak jika dilakukan terus-menerus."
    elif "Kasus Rekayasa Perangkat Lunak" in quick_sample_opt:
        default_prompt = "Bagaimana cara menangani pengecualian (exception) di Python agar program tidak crash?"
        default_c1 = "Gunakan blok try-except terstruktur. Tangkap exception spesifik seperti ValueError atau KeyError, dan hindari penggunaan bare except agar bug kritis tidak tersembunyi tanpa log."
        default_c2 = "Pakai saja try dan except: pass di semua bagian kode supaya program jalan terus tanpa ada error yang muncul di terminal."
    elif "Kasus Sastra" in quick_sample_opt:
        default_prompt = "Buatlah bait puisi singkat tentang senja di pantai."
        default_c1 = "Kala surya rebah di pelukan laut jingga,\nOmbak berbisik menyeka jejak yang fana,\nDalam keheningan semesta merenda jeda,\nMenyambut malam berbalut rahasia buana."
        default_c2 = "Matahari terbenam di pantai sangat indah,\nWarna langit menjadi oranye kemerahan,\nOrang-orang berjalan di pasir pantai,\nMenikmati sore yang tenang."

    p_interactive = st.selectbox(
        "Strategi Prompt Evaluasi:",
        ["P1 (Baseline Direct)", "P2 (Multi-Criteria)", "P3 (Debiasing RCAF)"]
    )
    p_id_play = p_interactive.split(" ")[0]

    input_user = st.text_area(
        "Instruksi / Pertanyaan Pengguna:",
        value=default_prompt,
        height=100
    )

    c_cand1, c_cand2 = st.columns(2)
    with c_cand1:
        cand_1_text = st.text_area("Teks Kandidat 1:", value=default_c1, height=180)
        render_html(
            f"<div style='font-family: \"JetBrains Mono\", monospace; font-size: 0.72rem; color: var(--text-muted); text-align: right;'>"
            f"Karakter: <span class='tabular-num'>{len(cand_1_text)}</span>"
            f"</div>"
        )
    with c_cand2:
        cand_2_text = st.text_area("Teks Kandidat 2:", value=default_c2, height=180)
        render_html(
            f"<div style='font-family: \"JetBrains Mono\", monospace; font-size: 0.72rem; color: var(--text-muted); text-align: right;'>"
            f"Karakter: <span class='tabular-num'>{len(cand_2_text)}</span>"
            f"</div>"
        )

    if st.button("Jalankan Evaluasi Simetri (AB & BA)", use_container_width=True):
        if not input_user.strip() or not cand_1_text.strip() or not cand_2_text.strip():
            st.warning("Seluruh kolom teks instruksi dan kandidat respons wajib diisi.")
            return

        with st.spinner("Mengeksekusi inferensi posisi simetris (AB dan BA)..."):
            eval_res = DualPositionEvaluator.evaluate_symmetry(p_id_play, input_user, cand_1_text, cand_2_text)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Hasil Evaluasi Simetri Posisi")

            is_flip_res = eval_res.is_flip
            status_pill_class = "danger" if is_flip_res else "success"
            status_pill_label = "Bias Posisi (Flip)" if is_flip_res else "Konsisten"

            total_runtime = eval_res.runtime_AB + eval_res.runtime_BA
            mode_clean = html.escape(str(eval_res.mode))
            parsed_ab_clean = html.escape(str(eval_res.parsed_AB))
            logical_ab_clean = html.escape(str(eval_res.logical_decision_AB))
            parsed_ba_clean = html.escape(str(eval_res.parsed_BA))
            logical_ba_clean = html.escape(str(eval_res.logical_decision_BA))

            render_html(f"""
            <div class="hud-telemetry">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">
                    Waktu Eksekusi: <span class="tabular-num"><strong>{total_runtime:.2f}s</strong></span> |
                    AB: <span class="tabular-num">{eval_res.runtime_AB:.2f}s</span> |
                    BA: <span class="tabular-num">{eval_res.runtime_BA:.2f}s</span> |
                    Mode: <code>{mode_clean}</code>
                </div>
                <div>
                    <span class="status-pill {status_pill_class}">{status_pill_label}</span>
                </div>
            </div>
            """)

            analysis_text = (
                "Model mengubah pilihan saat urutan teks kandidat ditukar. Hal ini menunjukkan model memiliki bias posisi terhadap urutan kemunculan respons."
                if is_flip_res else
                "Model konsisten memilih kandidat yang sama baik pada urutan AB maupun BA. Keputusan tidak dipengaruhi urutan penyajian respons."
            )
            border_signal = "var(--signal-danger)" if is_flip_res else "var(--signal-success)"

            render_html(f"""
            <div style="background-color: var(--bg-surface); border: 1px solid var(--border-contrast); border-radius: 4px; padding: 16px; margin-bottom: 16px;">
                <div style="font-size: 0.92rem; font-weight: 600; color: var(--text-primary); margin-bottom: 8px;">
                    Evaluasi Simetri Posisi (AB vs BA)
                </div>
                <table class="audit-matrix-table">
                    <thead>
                        <tr>
                            <th>Kondisi Permutasi</th>
                            <th>Pemetaan Slot Input</th>
                            <th>Token Output Model</th>
                            <th>Keputusan Logis</th>
                            <th>Waktu Eksekusi</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Posisi Normal (AB)</strong></td>
                            <td>Slot A: Kandidat 1 | Slot B: Kandidat 2</td>
                            <td><code>{parsed_ab_clean}</code></td>
                            <td><strong>Kandidat {logical_ab_clean}</strong></td>
                            <td class="tabular-num">{eval_res.runtime_AB:.2f}s</td>
                        </tr>
                        <tr>
                            <td><strong>Posisi Terbalik (BA)</strong></td>
                            <td>Slot A: Kandidat 2 | Slot B: Kandidat 1</td>
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

            with st.expander("Inspeksi Keluaran Mentah (Raw Output) Model"):
                st.code(f"Keluaran Mentah AB:\n{eval_res.raw_output_AB}\n\nKeluaran Mentah BA:\n{eval_res.raw_output_BA}", language="text")
