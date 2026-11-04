"""Unit test untuk view sidebar."""
from unittest.mock import patch, MagicMock
from llm_reliability.ui.views.sidebar import render_sidebar
from llm_reliability.domain import DataPaths, FilterCriteria


def test_render_sidebar_extracts_criteria():
    mock_repo = MagicMock()
    mock_repo.get_dataset_statistics.return_value = {"sample_count": 519, "total_evaluations": 3114}
    mock_repo.get_available_domains.return_value = ["penalaran_logis", "bahasa_sastra"]

    paths = DataPaths(results_dir="UI/data/results")

    with patch('streamlit.sidebar'), patch('streamlit.selectbox') as mock_sb:
        mock_sb.side_effect = ["Semua Prompt", "Semua Domain", "Semua Sampel", "Semua"]
        criteria = render_sidebar(mock_repo, paths)
        assert isinstance(criteria, FilterCriteria)
        assert criteria.prompt_id == "Semua Prompt"
        assert criteria.domain == "Semua Domain"
