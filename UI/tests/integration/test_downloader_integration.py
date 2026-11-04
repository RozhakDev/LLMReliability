"""Integration test untuk download stream, chunking, dan fallback mirror."""
import os
import tempfile
from unittest.mock import patch, MagicMock
from llm_reliability.services.model_downloader import download_qwen3_gguf


def test_download_qwen3_gguf_mock_streaming():
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_chunks = [b'chunk1_', b'chunk2_', b'chunk3_valid_bytes']
        mock_resp = MagicMock()
        mock_resp.headers = {'Content-Length': str(sum(len(c) for c in mock_chunks))}
        mock_resp.read.side_effect = mock_chunks + [b'']
        mock_resp.__enter__.return_value = mock_resp

        progress_calls = []
        def _cb(fraction, msg):
            progress_calls.append((fraction, msg))

        mock_settings = MagicMock()
        mock_settings.download_mirrors = ["https://fake-mirror.com/model.gguf"]
        mock_settings.expected_min_bytes = 10

        with patch('urllib.request.urlopen', return_value=mock_resp), \
             patch('llm_reliability.services.model_downloader.get_settings', return_value=mock_settings):
            target_path = download_qwen3_gguf(tmp_dir, progress_callback=_cb, force_download=True)
            assert os.path.exists(target_path)
            assert len(progress_calls) >= 3
            assert not os.path.exists(target_path + ".part")
