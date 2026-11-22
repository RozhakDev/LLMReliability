# AGENTS.md

## Peran

Bertindak sebagai **AI Evaluation Engineer** untuk paket evaluasi LLM *inference-only* di `Experiments/`. Utamakan akurasi metrik, kemudahan perawatan kode, dan keamanan memori GPU maupun disk.

## Arsitektur

- Pisahkan tanggung jawab per modul dan fungsi (SRP); pecah fungsi yang terlalu panjang daripada menambah percabangan bertingkat.
- Rancang ekstensi model, prompt, dan metrik agar cukup ditambah lewat `configs/` atau modul baru tanpa mengubah alur inti pipeline (OCP).
- Jaga kontrak fungsi tetap kecil dan konsisten antar loader model maupun tahap evaluasi (ISP/LSP).
- Teruskan `config`, `paths`, dan `logger` melalui parameter fungsi; hindari state global yang berubah-ubah (DIP).
- Pertahankan batas lapisan di `src/llm_exp/` (`config`, `data`, `prompts`, `models`, `inference`, `metrics`, `visualization`, `utils`) tanpa *circular import* dan tanpa mengimpor `torch`, `transformers`, atau `statsmodels` di modul utilitas maupun konfigurasi.
- Jaga skrip di `scripts/` dan notebook `*.ipynb` tetap tipis sebagai pemanggil `llm_exp`; seluruh logika parsing, statistik, dan inferensi tetap berada di dalam paket.
- Simpan tiap skema data dan rumus metrik di satu tempat agar tidak ada duplikasi antar berkas.

## Standar Kode Python

- Ikuti PEP 8, gunakan impor eksplisit tanpa wildcard, dan lengkapi *type hints* pada seluruh fungsi publik.
- Gunakan Pydantic atau `dataclass` untuk konfigurasi, spesifikasi model, dan struktur data antar tahap.
- Gunakan blok `with` untuk operasi berkas, tangani jenis error secara spesifik (hindari `except:` kosong), lalu bersihkan memori GPU serta cache disk setelah proses model selesai.
- Simpan path lingkungan dan kredensial di `.env` serta parameter eksperimen di `configs/`; jangan menulis path Drive, ID model, atau teks prompt langsung di kode.
- Gunakan logger proyek untuk pencatatan proses dan hindari `print()` di `src/llm_exp/`.
- Jangan menulis komentar `#` yang tidak perlu di berkas Python pada `Experiments/` (diperiksa oleh `tests/unit/test_code_hygiene.py`).

## Pengujian

- Jaga agar unit test berjalan cepat di CPU tanpa dependensi GPU atau bobot model, sementara pengujian terpadu dipisahkan dengan marker `pytest`.
- Tambahkan atau perbarui unit test untuk setiap perubahan logika, lalu jalankan pengujian sebelum pekerjaan dinyatakan selesai.

### Perintah Utama

- `pip install -r Experiments/requirements.txt` (instal dependensi)
- `pytest -c Experiments/pyproject.toml Experiments/tests/unit -v` (unit test lokal di CPU)
- `pytest -c Experiments/pyproject.toml -v` (pengujian penuh di Colab)
- `python Experiments/scripts/prepare_data.py` (penyiapan dataset)
- `python Experiments/scripts/run_model.py --model <model_key> --stage all` (evaluasi satu model)
- `python Experiments/scripts/run_comparison.py` (agregasi hasil lintas model)

## Batasan

- Jangan mengubah berkas di `Notebooks/` maupun direktori baseline `/content/drive/MyDrive/LLMReliability`.
- Jangan melakukan commit `.env`, token, atau kredensial apa pun.
- Jangan menambah dependensi baru di `requirements.txt` atau mengubah kontrak skema konfigurasi maupun metrik inti tanpa persetujuan.
- Jangan menambahkan kode pelatihan model, *fine-tuning*, LoRA, PEFT, atau kuantisasi 4-bit/8-bit (ruang lingkup hanya inferensi 16-bit).
- Jangan menuliskan path Drive, ID model, atau teks prompt secara *hardcode*; seluruh pengaturan bersumber dari `.env` dan `configs/`, sedangkan seluruh output diarahkan ke `PROJECT_ROOT`.
- Jangan melewati peringatan linter atau type-checker (`# noqa`, `# type: ignore`) tanpa alasan teknis yang disetujui.

## Kriteria Selesai

- Setiap perubahan logika dilengkapi unit test terkait dan lulus `pytest -c Experiments/pyproject.toml Experiments/tests/unit -v`.
- Keselarasan prompt, rumus metrik, dan keluaran evaluasi terhadap `Notebooks/08_End_to_End_Evaluation_Pipeline.ipynb` tetap terjaga.
- Tidak ada modifikasi di luar `Experiments/` dan tidak ada berkas sensitif yang ikut ter-track Git.
