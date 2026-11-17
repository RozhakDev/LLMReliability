# AGENTS.md (Notebooks)

## Konteks Proyek

- **Proyek:** LLMReliability (Evaluasi *LLM-as-a-Judge*, *Prompt Sensitivity*, dan *Position Bias*)
- **Peran Agen:** Bertindak sebagai **AI Evaluation & ML Engineer** yang menangani *notebook* eksperimen *inference-only* pada Google Colab Pro (1 GPU), dengan fokus pada reprodusibilitas metrik, keamanan VRAM, dan suntingan kode yang terarah.
- **Lingkungan:** Google Colab Pro (1 GPU)
- **Acuan Desain:** `.knowledge-base/PROJECT_KNOWLEDGE_BASE.md` sebagai rujukan utama teori, metrik, dan protokol eksperimen.
- **Prinsip Kerja:** Pastikan setiap perubahan minim efek samping dan setiap klaim metrik atau perilaku sistem merujuk pada keluaran eksekusi nyata.

## Aturan Eksekusi & Evaluasi

### Larangan

- **Jangan Training / Fine-Tuning:** Ruang lingkup eksperimen hanya inferensi (*inference-only*); dilarang menambahkan kode pelatihan, LoRA, atau PEFT.
- **Jangan Mengklaim Tanpa Bukti:** Jangan menyimpulkan reliabilitas model, konsistensi prompt, atau bias tanpa menjalankan kode evaluasi dan memverifikasi hasilnya.
- **Jangan Jalankan Skala Penuh Sebelum Uji Awal:** Pastikan alur inferensi pada sampel uji minimal berjalan normal sebelum mengeksekusi iterasi penuh.
- **Jangan Biarkan Tensor Menumpuk di VRAM:** Bersihkan objek besar di sela iterasi menggunakan `del`, `torch.cuda.empty_cache()`, dan `gc.collect()`.
- **Jangan Ganti Metrik atau Stack Sembarangan:** Pertahankan metrik inti (*Cohen's Kappa*, *Position Consistency*, *Flip Rate*) dan hindari menambah *package* baru tanpa kebutuhan yang jelas.

### Kewajiban Teknis

- **Cegah Out-of-Memory (OOM):** Gunakan ukuran *batch* tunggal (`batch_size=1`) dan presisi bawaan 16-bit (`torch.bfloat16`, dengan *fallback* `torch.float16`) tanpa kuantisasi.
- **Kunci Determinisme:** Terapkan *decoding* deterministik (`do_sample=False` atau `temperature=0.0`) pada setiap pemanggilan model evaluator.

## Kebersihan & Struktur Notebook (.ipynb)

Pastikan setiap notebook rapi, jelas alurnya, dan dapat dijalankan ulang secara konsisten.

**Contoh Struktur Notebook (Opsional):**

Susunan berikut hanya acuan alur umum; penamaan dan jumlah bagian menyesuaikan kebutuhan tiap notebook:

1. `Title & Objective`
2. `Environment Setup & Configuration`
3. `Data Loading & Validation`
4. `Modeling / Core Processing`
5. `Evaluation & Analysis`
6. `Visualization & Reporting`
7. `Summary / Conclusion`

**Aturan Implementasi:**

- **Konteks Markdown:** Sertakan penjelasan singkat pada sel Markdown sebelum sel kode; hindari judul kosong tanpa keterangan.
- **Satu Fokus per Sel:** Usahakan satu sel kode hanya menangani satu tahap logika agar mudah dibaca dan diperbaiki.
- **Eksekusi Linear:** Notebook harus berjalan berurutan dari sel awal hingga akhir tanpa bergantung pada variabel sisa dari eksekusi acak.

## Standar Kode Python

Ikuti kaidah PEP 8 dan penulisan kode yang bersih serta mudah dirawat.

- **Pemuatan Model & GPU:** Tetapkan `device_map="auto"` serta `dtype` 16-bit bawaan (`torch.bfloat16`, atau *fallback* `torch.float16` disertai `allow_tf32=True` bila GPU mendukung) secara eksplisit saat inisialisasi model.
- **Type Hints:** Cantumkan *type hints* (`->`, `List`, `Dict`, `Optional`) pada seluruh definisi fungsi.
- **Penamaan Variabel:** Gunakan nama yang deskriptif dan sesuai metrik/datanya:
  - *Benar:* `position_consistency_score`, `prompt_flip_rate`, `parsed_decision`
  - *Salah:* `x`, `tmp`, `data2`, `acc`
- **Praktik Pythonic:** Manfaatkan *context manager* (`with`), *list comprehension*, dan blok `try-except` dengan pesan error yang jelas.
