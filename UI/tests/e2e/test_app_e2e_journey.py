"""E2E User Journey Test menggunakan Streamlit AppTest untuk interaksi simulasi nyata."""
from pathlib import Path
from streamlit.testing.v1 import AppTest
from llm_reliability.core.paths import PROJECT_ROOT

APP_FILE = str(Path(PROJECT_ROOT) / "main.py")


def test_e2e_sidebar_filtering_and_tabs(results_dir):
    """Simulasi alur pengguna: Membuka aplikasi, memilih filter prompt P1, dan memeriksa tab."""
    at = AppTest.from_file(APP_FILE, default_timeout=180).run()
    assert not at.exception

    tab_labels = [t.label for t in at.tabs]
    assert "Ringkasan Evaluasi" in tab_labels
    assert "Analisis Bias Posisi" in tab_labels
    assert "Sensitivitas Prompt & Domain" in tab_labels
    assert "Uji Simetri Interaktif" in tab_labels

    if len(at.sidebar.selectbox) >= 4:
        at.sidebar.selectbox[0].set_value("P1 (Direct)").run()
        assert not at.exception

        if len(at.sidebar.selectbox[1].options) > 1:
            domain_choice = at.sidebar.selectbox[1].options[1]
            at.sidebar.selectbox[1].set_value(domain_choice).run()
            assert not at.exception


def test_e2e_prompt_template_radio_switch(results_dir):
    """Simulasi alur pengguna pada Tab 3: Memeriksa radio switcher template prompt."""
    at = AppTest.from_file(APP_FILE, default_timeout=180).run()
    assert not at.exception

    if len(at.radio) >= 1:
        at.radio[0].set_value("P2").run()
        assert not at.exception
