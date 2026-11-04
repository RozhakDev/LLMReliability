"""Unit test untuk komponen metrics_view."""
from unittest.mock import patch, MagicMock
from llm_reliability.ui.components.metrics_view import (
    render_kpi_cards,
    render_position_bias_case,
    render_prompt_template_display,
    render_plots_section
)


def test_render_kpi_cards_overall(mock_metrics_summary_df, mock_mcnemar_dict):
    with patch('llm_reliability.ui.components.metrics_view.render_html') as mock_render:
        render_kpi_cards(mock_metrics_summary_df, mock_mcnemar_dict, "Semua Prompt")
        assert mock_render.call_count >= 4
        rendered_texts = [str(call[0][0]) for call in mock_render.call_args_list]
        combined = " ".join(rendered_texts)
        assert "Rata-rata 3 prompt" in combined


def test_render_kpi_cards_single_prompt(mock_metrics_summary_df, mock_mcnemar_dict):
    with patch('llm_reliability.ui.components.metrics_view.render_html') as mock_render:
        render_kpi_cards(mock_metrics_summary_df, mock_mcnemar_dict, "P1 (Direct)")
        assert mock_render.call_count >= 4
        rendered_texts = [str(call[0][0]) for call in mock_render.call_args_list]
        combined = " ".join(rendered_texts)
        assert "[0.51, 0.61]" in combined


def test_render_position_bias_case_clean_badges(mock_paired_df):
    row = mock_paired_df.iloc[0]
    with patch('llm_reliability.ui.components.metrics_view.render_html') as mock_render, patch('streamlit.markdown'):
        render_position_bias_case(row)
        rendered_texts = [str(call[0][0]) for call in mock_render.call_args_list]
        combined = " ".join(rendered_texts)
        assert "Pilihan Ground Truth" in combined
        assert "Kalah Ground Truth" not in combined


def test_render_plots_section_fallback():
    empty_paths = {
        "plot_accuracy": None,
        "plot_consistency": None,
        "plot_parsing_domain": None,
        "plot_heatmaps": None
    }
    with patch('streamlit.info') as mock_info, patch('streamlit.tabs') as mock_tabs:
        mock_tabs.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        render_plots_section(empty_paths)
        assert mock_info.call_count >= 1
