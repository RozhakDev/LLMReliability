# AGENTS.md

## Peran

Bertindak sebagai **AI Evaluation Engineer** untuk paket evaluasi single-model `Qwen3-32B` (*inference-only*) di `Qwen3/`. Utamakan akurasi metrik, konsistensi penuh dengan `Experiments/` dan `Notebooks/08_End_to_End_Evaluation_Pipeline.ipynb`, serta keamanan memori GPU A100, disk lokal Colab, dan direktori Google Drive.

## Arsitektur

- Pisahkan tanggung jawab per modul dan fungsi (SRP); pecah fungsi yang terlalu panjang daripada menambah percabangan bertingkat.
- Rancang konfigurasi model, prompt, dan parameter eksperimen agar cukup diubah melalui `configs/` tanpa menyentuh alur inti pipeline (OCP).
- Jaga kontrak fungsi tetap kecil, eksplisit, dan konsisten antar tahap evaluasi (ISP/LSP).
- Teruskan `config`, `paths`, `spec`, dan `logger` melalui parameter fungsi; hindari state global yang berubah-ubah (DIP).
- Pertahankan batas lapisan di `src/qwen3_eval/` (`config`, `data`, `prompts`, `models`, `inference`, `metrics`, `visualization`, `utils`) tanpa *circular import* dan tanpa mengimpor `torch`, `transformers`, atau `statsmodels` di modul konfigurasi maupun logger.
- Jaga skrip di `scripts/` dan notebook `01_Qwen3_Evaluation_Runner.ipynb` tetap tipis sebagai pemanggil `qwen3_eval`; seluruh logika parsing, statistik, inferensi, dan plotting tetap berada di dalam paket.
- Simpan tiap skema data dan rumus metrik di satu tempat agar tidak ada duplikasi antar berkas.

## Standar Kode Python

- Ikuti PEP 8, gunakan impor eksplisit tanpa wildcard, dan lengkapi *type hints* (PEP 484) pada seluruh fungsi publik.
- Gunakan Pydantic v2 (`frozen=True, extra="forbid"`) atau `dataclass` untuk konfigurasi, spesifikasi model, dan struktur data antar tahap.
- Gunakan blok `with` untuk operasi berkas, tangani jenis error secara spesifik (hindari `except:` kosong), lalu bersihkan memori GPU serta cache disk setelah inferensi tuntas 100%.
- Simpan path lingkungan dan kredensial di `.env` serta parameter eksperimen di `configs/`; jangan menulis path Drive, ID model, ID dataset, atau teks prompt langsung di kode sumber `src/qwen3_eval/`.
- Gunakan logger proyek terstruktur (`event_name | key=value`) dan hindari `print()` di `src/qwen3_eval/`.
- Jangan menulis komentar `#` maupun docstring (`"""..."""`) di berkas Python pada `Qwen3/` (diperiksa otomatis oleh `tests/unit/test_code_hygiene.py`).

## Pengujian

- Jaga agar unit test berjalan cepat di CPU tanpa dependensi GPU atau bobot model, sementara pengujian terpadu dipisahkan dengan marker `pytest`.
- Tambahkan atau perbarui unit test untuk setiap perubahan logika, lalu jalankan pengujian sebelum pekerjaan dinyatakan selesai.

### Perintah Utama

- `pip install -r Qwen3/requirements.txt` (instal dependensi)
- `pytest -c Qwen3/pyproject.toml Qwen3/tests/unit -v` (unit test lokal di CPU)
- `pytest -c Qwen3/pyproject.toml -v` (pengujian penuh termasuk integrasi)
- `python Qwen3/scripts/prepare_data.py` (penyiapan dataset IndoPref)
- `python Qwen3/scripts/run_pipeline.py --stage all` (eksekusi end-to-end single-model Qwen3)

## Batasan

- Jangan mengubah berkas di `Notebooks/`, `Experiments/`, maupun direktori Drive yang diproteksi (`/content/drive/MyDrive/LLMReliability` dan `/content/drive/MyDrive/LLMReliabilityEXP`).
- Jangan melakukan commit `.env`, token, atau kredensial apa pun.
- Jangan menambah dependensi baru di `requirements.txt` atau mengubah kontrak skema konfigurasi maupun rumus metrik inti tanpa persetujuan.
- Jangan menambahkan kode pelatihan model, *fine-tuning*, LoRA, PEFT, atau kuantisasi 4-bit/8-bit (ruang lingkup hanya inferensi 16-bit BF16).
- Jangan melewati peringatan linter atau type-checker (`# noqa`, `# type: ignore`) tanpa alasan teknis yang disetujui.

## Kriteria Selesai

- Setiap perubahan logika dilengkapi unit test terkait dan lulus `pytest -c Qwen3/pyproject.toml Qwen3/tests/unit -v`.
- Keselarasan prompt, rumus metrik, 6 artefak tabel, dan 12 pasang plot PNG/SVG terhadap `Experiments/` dan `Notebooks/08_End_to_End_Evaluation_Pipeline.ipynb` tetap terjaga.
- Tidak ada berkas sensitif yang ikut ter-track Git.
