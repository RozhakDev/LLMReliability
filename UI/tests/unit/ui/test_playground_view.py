"""Unit test untuk view Tab 4 (Playground Interaktif)."""
from unittest.mock import patch
from llm_reliability.ui.views.interactive_playground import render_playground_tab


def test_render_playground_tab_status():
    with patch('llm_reliability.services.backends.factory.BackendFactory.get_backend_status') as mock_status, \
         patch('llm_reliability.ui.views.interactive_playground.render_html') as mock_render, \
         patch('streamlit.selectbox') as mock_sb, \
         patch('streamlit.text_area') as mock_ta, \
         patch('streamlit.button') as mock_btn:
        mock_status.return_value = {
            "is_ready": True,
            "model_ready": True,
            "runtime_ready": True,
            "device_name": "CPU (llama.cpp)",
            "model_size_mb": 2330.0
        }
        mock_sb.return_value = "P1 (Direct)"
        mock_ta.return_value = "Sample"
        mock_btn.return_value = False

        render_playground_tab()
        assert mock_render.call_count >= 1


def test_render_playground_tab_model_not_ready():
    with patch('llm_reliability.services.backends.factory.BackendFactory.get_backend_status') as mock_status, \
         patch('llm_reliability.ui.views.interactive_playground.render_html') as mock_render, \
         patch('streamlit.selectbox') as mock_sb, \
         patch('streamlit.text_area') as mock_ta, \
         patch('streamlit.button') as mock_btn, \
         patch('streamlit.text_input') as mock_ti, \
         patch('streamlit.expander'):
        mock_status.return_value = {
            "is_ready": False,
            "model_ready": False,
            "runtime_ready": True,
            "device_name": "CPU (llama.cpp)",
            "model_size_mb": 0.0
        }
        mock_sb.return_value = "P1 (Direct)"
        mock_ta.return_value = "Sample"
        mock_btn.return_value = False
        mock_ti.return_value = ""

        render_playground_tab()
        rendered_texts = [str(call[0][0]) for call in mock_render.call_args_list]
        combined = " ".join(rendered_texts)
        assert "File Model Belum Tersedia" in combined


def test_render_playground_tab_run_evaluation():
    from llm_reliability.domain import DualInferenceResult
    mock_res = DualInferenceResult(
        parsed_AB="A",
        raw_output_AB="A",
        logical_decision_AB=1,
        runtime_AB=0.45,
        parsed_BA="B",
        raw_output_BA="B",
        logical_decision_BA=1,
        runtime_BA=0.45,
        is_consistent=True,
        is_flip=False,
        mode="CPU_LLAMA",
        device_name="CPU",
        ram_usage_mb=1200.0,
        peak_ram_mb=1300.0
    )
    with patch('llm_reliability.services.backends.factory.BackendFactory.get_backend_status') as mock_status, \
         patch('llm_reliability.ui.views.interactive_playground.render_html') as mock_render, \
         patch('llm_reliability.services.evaluator.DualPositionEvaluator.evaluate_symmetry', return_value=mock_res), \
         patch('streamlit.selectbox') as mock_sb, \
         patch('streamlit.text_area') as mock_ta, \
         patch('streamlit.button') as mock_btn:
        mock_status.return_value = {
            "is_ready": True,
            "model_ready": True,
            "runtime_ready": True,
            "device_name": "CPU (llama.cpp)",
            "model_size_mb": 2330.0
        }
        mock_sb.return_value = "P1 (Direct)"
        mock_ta.side_effect = ["Instruksi Pertanyaan", "Respons Kandidat 1", "Respons Kandidat 2"]
        mock_btn.return_value = True

        render_playground_tab()
        rendered_texts = [str(call[0][0]) for call in mock_render.call_args_list]
        combined = " ".join(rendered_texts)
        assert "Hasil Evaluasi" in combined
        assert "Konsisten" in combined
