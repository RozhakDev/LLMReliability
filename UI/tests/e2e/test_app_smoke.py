"""E2E Smoke test untuk validasi kesiapan inisialisasi aplikasi Streamlit."""
from pathlib import Path
from streamlit.testing.v1 import AppTest
from llm_reliability.core.paths import PROJECT_ROOT

APP_FILE = str(Path(PROJECT_ROOT) / "main.py")


def test_app_renders_four_tabs_without_exception(results_dir):
    at = AppTest.from_file(APP_FILE, default_timeout=180).run()
    assert not at.exception
    labels = [t.label for t in at.tabs]
    for expected in (
        "Ringkasan Evaluasi",
        "Analisis Bias Posisi",
        "Sensitivitas Prompt & Domain",
        "Uji Simetri Interaktif",
    ):
        assert expected in labels
