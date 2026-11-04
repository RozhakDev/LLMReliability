"""Unit test untuk view Tab 3 (Sensitivitas Prompt)."""
from unittest.mock import patch, MagicMock
from llm_reliability.ui.views.prompt_sensitivity import render_prompt_sensitivity_tab


def test_render_prompt_sensitivity_tab(mock_metrics_summary_df):
    mock_repo = MagicMock()
    mock_repo.get_prompt_stability_matrix.return_value = mock_metrics_summary_df
    mock_repo.load_domain_accuracy.return_value = mock_metrics_summary_df

    with patch('streamlit.dataframe') as mock_df, patch('streamlit.radio') as mock_radio:
        mock_radio.return_value = "P1"
        render_prompt_sensitivity_tab(mock_repo)
        assert mock_df.call_count >= 2
