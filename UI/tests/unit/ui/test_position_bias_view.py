"""Unit test untuk view Tab 2 (Analisis Bias Posisi)."""
from unittest.mock import patch
import pandas as pd
from llm_reliability.ui.views.position_bias_explorer import _format_sample_label, render_flip_explorer_tab


def test_format_sample_label_clean(mock_paired_df):
    row_flip = mock_paired_df.iloc[1]
    label_flip = _format_sample_label(row_flip)
    assert "Flip" in label_flip
    assert "[FLIP]" not in label_flip
    assert "#2" in label_flip

    row_cons = mock_paired_df.iloc[0]
    label_cons = _format_sample_label(row_cons)
    assert "Konsisten" in label_cons
    assert "Sesuai GT" in label_cons


def test_render_flip_explorer_tab_empty():
    empty_df = pd.DataFrame()
    with patch('streamlit.warning') as mock_warn:
        render_flip_explorer_tab(empty_df)
        mock_warn.assert_called_once()
        msg = str(mock_warn.call_args[0][0])
        assert "sidebar" in msg
        assert "bilah samping" not in msg
