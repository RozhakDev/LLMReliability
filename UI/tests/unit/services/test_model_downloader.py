"""Unit test untuk modul downloader model GGUF Qwen3-4B-Instruct-2507."""
import os
import tempfile
from unittest.mock import patch, MagicMock
from llm_reliability.services.model_downloader import (
    get_target_model_path,
    is_model_available,
    get_model_size_mb,
    verify_mirror_connection,
    download_qwen3_gguf
)


def test_get_target_model_path():
    path = get_target_model_path("UI/data/models")
    assert path.endswith("qwen3-4b-instruct-2507-q4_k_m.gguf")


def test_is_model_available_thresholds():
    with tempfile.TemporaryDirectory() as tmp_dir:
        fake_file = os.path.join(tmp_dir, "qwen3-4b-instruct-2507-q4_k_m.gguf")
        assert is_model_available(tmp_dir) is False

        with open(fake_file, 'wb') as f:
            f.write(b'x' * 100)
        assert is_model_available(tmp_dir) is False


def test_get_model_size_mb():
    with tempfile.TemporaryDirectory() as tmp_dir:
        fake_file = os.path.join(tmp_dir, "qwen3-4b-instruct-2507-q4_k_m.gguf")
        with open(fake_file, 'wb') as f:
            f.write(b'0' * (1024 * 1024 * 5))  # 5 MB
        size = get_model_size_mb(tmp_dir)
        assert 4.9 <= size <= 5.1


@patch('urllib.request.urlopen')
def test_verify_mirror_connection_success(mock_urlopen):
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    mirror = verify_mirror_connection(timeout=2)
    assert mirror is not None
    assert mirror.startswith("https://")


def test_model_downloader_main_skip():
    from llm_reliability.services.model_downloader import main
    with patch('sys.argv', ['model_downloader', '--target-dir', 'fake/dir']), \
         patch('llm_reliability.services.model_downloader.is_model_available', return_value=True), \
         patch('builtins.print') as mock_print:
        main()
        printed = " ".join(str(c[0][0]) for c in mock_print.call_args_list)
        assert "Pengunduhan dilewati" in printed


def test_model_downloader_main_download_success():
    from llm_reliability.services.model_downloader import main
    with patch('sys.argv', ['model_downloader', '--target-dir', 'fake/dir']), \
         patch('llm_reliability.services.model_downloader.is_model_available', return_value=False), \
         patch('llm_reliability.services.model_downloader.download_qwen3_gguf', return_value='fake/dir/model.gguf'), \
         patch('builtins.print') as mock_print:
        main()
        printed = " ".join(str(c[0][0]) for c in mock_print.call_args_list)
        assert "Sukses" in printed
