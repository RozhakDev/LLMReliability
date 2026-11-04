"""
Konstanta Sistem, Metadata Ilmiah, dan Token Konfigurasi Aplikasi.
Seluruh nilai statis, label UI, dan parameter domain dipusatkan di sini (Zero-Hardcode).
"""

from typing import Dict, List, Tuple

APP_TITLE = "LLM-as-a-Judge Reliability Dashboard"
APP_SUBTITLE = "Evaluasi Reliabilitas, Bias Posisi, dan Sensitivitas Prompt pada LLM-as-a-Judge"
APP_PROTOCOL_LABEL = "Permutasi Posisi (AB vs BA)"
APP_YEAR = "2026"

DEFAULT_MODEL_NAME = "Qwen3-4B-Instruct-2507"
DEFAULT_MODEL_REPO = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_GGUF_FILENAME = "qwen3-4b-instruct-2507-q4_k_m.gguf"
QUANTIZATION_LABEL_CPU = "GGUF Q4_K_M"
QUANTIZATION_LABEL_GPU = "4-bit NF4"

DATASET_NAME = "IndoPref"
DEFAULT_SAMPLE_COUNT = 519
DEFAULT_PAIRED_COUNT = 1557
DEFAULT_TOTAL_EVALUATION_COUNT = 3114

PROMPT_CODES: Tuple[str, ...] = ("P1", "P2", "P3")
PROMPT_FILTER_ALL = "Semua Prompt"

PROMPT_DISPLAY_NAMES: Dict[str, str] = {
    "Semua Prompt": "Semua Prompt",
    "P1": "P1 (Direct)",
    "P2": "P2 (Multi-Criteria)",
    "P3": "P3 (Mitigasi Bias)"
}

PROMPT_FILTER_OPTIONS: List[str] = [
    PROMPT_FILTER_ALL,
    "P1 (Direct)",
    "P2 (Multi-Criteria)",
    "P3 (Mitigasi Bias)"
]

SYMMETRY_FILTER_ALL = "Semua Sampel"
SYMMETRY_FILTER_FLIP = "Hanya Bias Posisi (Flip)"
SYMMETRY_FILTER_CONSISTENT = "Hanya Konsisten"

SYMMETRY_FILTER_OPTIONS: List[str] = [
    SYMMETRY_FILTER_ALL,
    SYMMETRY_FILTER_FLIP,
    SYMMETRY_FILTER_CONSISTENT
]

GT_FILTER_ALL = "Semua"
GT_FILTER_CORRECT = "Sesuai Ground Truth"
GT_FILTER_INCORRECT = "Tidak Sesuai Ground Truth"

GT_FILTER_OPTIONS: List[str] = [
    GT_FILTER_ALL,
    GT_FILTER_CORRECT,
    GT_FILTER_INCORRECT
]

DOMAIN_ALL = "Semua Domain"

DEFAULT_STATISTICAL_ALPHA = 0.05
DEFAULT_FLIP_ALERT_THRESHOLD = 20.0
DEFAULT_CONSISTENCY_HIGH_THRESHOLD = 70.0
EXPECTED_MIN_MODEL_BYTES = 2_000_000_000

DEFAULT_DOWNLOAD_MIRRORS: List[str] = [
    "https://hf-mirror.com/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    "https://hf-mirror.com/lmstudio-community/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    "https://huggingface.co/lmstudio-community/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
]

PLAYGROUND_SAMPLE_CASES: List[Dict[str, str]] = [
    {
        "label": "Input Manual (Ketik Sendiri)",
        "prompt": "Jelaskan efek penambahan kecepatan kendaraan secara tiba-tiba terhadap kinerja rem.",
        "cand_1": "Ketika kendaraan dipercepat tiba-tiba, energi kinetik kendaraan meningkat pesat. Akibatnya, sistem rem harus mengubah energi kinetik yang lebih besar tersebut menjadi energi panas melalui gesekan, yang menyebabkan beban termal dan mekanis pada rem meningkat drastis.",
        "cand_2": "Jika kendaraan ditambah kecepatannya secara mendadak, rem akan bekerja lebih keras dan bannya bisa rusak karena panas dan aus."
    },
    {
        "label": "Fisika: Pengereman Kendaraan",
        "prompt": "Jika Anda menambah kecepatan kendaraan tiba-tiba, apa efeknya pada rem?",
        "cand_1": "Akselerasi cepat meningkatkan torsi mesin dan energi kinetik. Untuk memperlambat kendaraan, sistem pengereman memerlukan tekanan hidrolik lebih tinggi dan gesekan bantalan rem yang jauh lebih intensif, meningkatkan suhu rem secara signifikan.",
        "cand_2": "Efeknya adalah rem jadi lebih panas, ban lebih cepat aus, dan rem bisa mengalami kerusakan mendadak jika dilakukan terus-menerus."
    },
    {
        "label": "Python: Exception Handling",
        "prompt": "Bagaimana cara menangani pengecualian (exception) di Python agar program tidak crash?",
        "cand_1": "Gunakan blok try-except terstruktur. Tangkap exception spesifik seperti ValueError atau KeyError, dan hindari penggunaan bare except agar bug kritis tidak tersembunyi tanpa log.",
        "cand_2": "Pakai saja try dan except: pass di semua bagian kode supaya program jalan terus tanpa ada error yang muncul di terminal."
    },
    {
        "label": "Sastra: Puisi Senja",
        "prompt": "Buatlah bait puisi singkat tentang senja di pantai.",
        "cand_1": "Kala surya rebah di pelukan laut jingga,\nOmbak berbisik menyeka jejak yang fana,\nDalam keheningan semesta merenda jeda,\nMenyambut malam berbalut rahasia buana.",
        "cand_2": "Matahari terbenam di pantai sangat indah,\nWarna langit menjadi oranye kemerahan,\nOrang-orang berjalan di pasir pantai,\nMenikmati sore yang tenang."
    }
]
