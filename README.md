# LLMReliability

**Analisis Reliabilitas LLM-as-a-Judge pada Variasi Prompt dan Posisi Kandidat Bahasa Indonesia**

LLMReliability adalah *framework* eksperimental yang dirancang untuk mengukur tingkat konsistensi dan keandalan Large Language Model (LLM) ketika difungsikan sebagai penilai otomatis (*judge*) untuk respons berbahasa Indonesia.

## Fokus Penelitian

Fokus utama proyek ini adalah menguji reliabilitas keputusan LLM terhadap referensi manusia (*human preference*) dengan mengevaluasi dua jenis kerentanan utama:

- **Position Bias:** Mengukur konsistensi keputusan *judge* ketika posisi kandidat yang dievaluasi ditukar (A-B menjadi B-A).
- **Prompt Sensitivity:** Mengukur stabilitas keputusan *judge* ketika dievaluasi menggunakan berbagai formulasi instruksi (*prompt*) yang secara semantik ekuivalen.

## Dataset

- **davidanugraha/IndoPref** (HuggingFace)
- Jenis: **Human Preference / Pairwise**
- Subset: Ditarik menggunakan metode *stratified sampling* berdasarkan kategori domain.

## Pipeline & Arsitektur Utama

1. **Dataset Preparation:** *Stratified sampling* dari himpunan data IndoPref.
2. **Experimental Setup:** Pembuatan instrumen variasi *Prompt* (P1/P2/P3) dan pengaturan posisi (AB/BA).
3. **Inference Engine:** Eksekusi deterministik menggunakan LLM `Qwen3-4B-Instruct-2507` berkuantisasi 4-bit.
4. **Statistical Analysis:** Pemrosesan hasil inferensi untuk menghitung tingkat perubahan keputusan (*flip rate*) dan metrik kesepakatan (*agreement*).

## Metrik Evaluasi

Performa keandalan model *judge* diukur menggunakan metrik statistik berikut:

- **Accuracy / Human Agreement**
- **Cohen's Kappa**
- **Position Consistency & Flip Rate**
- **Prompt Consistency & Flip Rate**

Validasi keandalan dilakukan melalui analisis perbandingan antar kondisi modifikasi *prompt* dan pertukaran posisi kandidat.

## Status

**Active Development** — Fokus saat ini pada penyiapan *environment*, penyusunan *notebook boilerplate* (`ML/`), dan validasi sampel dataset awal.

## Lisensi

Repositori ini dikelola untuk keperluan penelitian akademik dan pengembangan Tugas Akhir.
