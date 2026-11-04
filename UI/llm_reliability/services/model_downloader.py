import os
import sys
import time
import argparse
import urllib.request
import urllib.error
from typing import Optional, Callable

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from llm_reliability.core.constants import (
    DEFAULT_MODEL_REPO,
    DEFAULT_GGUF_FILENAME,
    DEFAULT_DOWNLOAD_MIRRORS,
    EXPECTED_MIN_MODEL_BYTES,
)
from llm_reliability.core.paths import MODELS_DIR
from llm_reliability.core.settings import get_settings

MODEL_REPO_ID = DEFAULT_MODEL_REPO
GGUF_FILENAME = DEFAULT_GGUF_FILENAME
DOWNLOAD_MIRRORS = DEFAULT_DOWNLOAD_MIRRORS
EXPECTED_MIN_BYTES = EXPECTED_MIN_MODEL_BYTES

def get_default_model_dir() -> str:
    """Mengembalikan path direktori penyimpanan model lokal (dari AppSettings atau UI/data/models)."""
    try:
        return get_settings().model_dir
    except Exception:
        return str(MODELS_DIR)

def get_target_model_path(target_dir: Optional[str] = None) -> str:
    """Mengembalikan path lengkap file model GGUF."""
    model_dir = target_dir or get_default_model_dir()
    return os.path.join(model_dir, GGUF_FILENAME)

def is_model_available(target_dir: Optional[str] = None) -> bool:
    """Memeriksa apakah bobot model GGUF telah diunduh di sistem lokal secara utuh."""
    path = get_target_model_path(target_dir)
    try:
        min_bytes = get_settings().expected_min_bytes
    except Exception:
        min_bytes = EXPECTED_MIN_BYTES
    return os.path.exists(path) and os.path.isfile(path) and os.path.getsize(path) >= min_bytes

def get_model_size_mb(target_dir: Optional[str] = None) -> float:
    """Mengembalikan ukuran file model lokal dalam megabyte (MB)."""
    path = get_target_model_path(target_dir)
    if os.path.exists(path):
        return round(os.path.getsize(path) / (1024 * 1024), 1)
    return 0.0

def verify_mirror_connection(timeout: int = 8) -> Optional[str]:
    """Menguji mirror mana yang paling cepat merespons (status HTTP 200)."""
    try:
        mirrors = get_settings().download_mirrors
    except Exception:
        mirrors = DOWNLOAD_MIRRORS

    for url in mirrors:
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    return url
        except Exception:
            continue
    return None

def download_qwen3_gguf(
    target_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    force_download: bool = False
) -> str:
    """
    Mengunduh bobot model Qwen3-4B-Instruct-2507 GGUF Q4_K_M (~2.33 GB).
    Menggunakan chunked stream download dengan mekanisme anti-redownload dan verifikasi integritas.
    """
    model_dir = target_dir or get_default_model_dir()
    os.makedirs(model_dir, exist_ok=True)
    target_path = os.path.join(model_dir, GGUF_FILENAME)
    temp_path = target_path + ".part"

    if not force_download and is_model_available(model_dir):
        if progress_callback:
            progress_callback(1.0, f"Model sudah tersedia: {target_path} ({get_model_size_mb(model_dir):.1f} MB)")
        return target_path

    try:
        settings = get_settings()
        mirrors = settings.download_mirrors
        min_bytes = settings.expected_min_bytes
    except Exception:
        mirrors = DOWNLOAD_MIRRORS
        min_bytes = EXPECTED_MIN_BYTES

    errors = []
    chunk_size = 4 * 1024 * 1024

    for mirror_idx, url in enumerate(mirrors, 1):
        try:
            if progress_callback:
                progress_callback(0.01, f"Menghubungkan ke mirror {mirror_idx}/{len(mirrors)}...")

            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                start_time = time.time()

                with open(temp_path, "wb") as out_file:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        downloaded += len(chunk)

                        if progress_callback and total_size > 0:
                            fraction = min(downloaded / total_size, 0.99)
                            elapsed = time.time() - start_time
                            speed_mb = (downloaded / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                            downloaded_mb = downloaded / (1024 * 1024)
                            total_mb = total_size / (1024 * 1024)
                            progress_callback(
                                fraction,
                                f"{downloaded_mb:.1f} MB / {total_mb:.1f} MB ({speed_mb:.2f} MB/s) - Mirror {mirror_idx}"
                            )

            if os.path.exists(temp_path) and os.path.getsize(temp_path) >= min_bytes:
                if os.path.exists(target_path):
                    os.remove(target_path)
                os.rename(temp_path, target_path)

                if progress_callback:
                    progress_callback(1.0, f"Pengunduhan selesai! Ukuran: {get_model_size_mb(model_dir):.1f} MB")
                return target_path
            else:
                actual_sz = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
                raise RuntimeError(f"Ukuran berkas tidak lengkap: {actual_sz} bytes (diharapkan >= {min_bytes} bytes)")

        except Exception as e:
            errors.append(f"Mirror {mirror_idx} ({url}): {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            continue

    err_summary = "\n".join(errors)
    raise RuntimeError(
        f"Gagal mengunduh {GGUF_FILENAME} dari seluruh mirror yang tersedia.\n"
        f"Riwayat kegagalan:\n{err_summary}\n\n"
        f"Solusi manual: Unduh berkas Qwen3-4B-Instruct-2507-Q4_K_M.gguf dari Hugging Face "
        f"dan simpan di direktori: {target_path}"
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="Unduh Model Qwen3-4B-Instruct-2507 GGUF Q4_K_M (~2.33 GB)")
    parser.add_argument("--target-dir", default=None, help="Direktori target penyimpanan model")
    parser.add_argument("--force", action="store_true", help="Paksa unduh ulang meskipun file sudah ada")
    args = parser.parse_args()

    dest_dir = args.target_dir or get_default_model_dir()
    print(f"Direktori Target: {dest_dir}")
    print(f"Model ID Baku: {MODEL_REPO_ID}")
    print(f"Nama Berkas GGUF: {GGUF_FILENAME}")

    if is_model_available(dest_dir) and not args.force:
        print(f"Model sudah ada di: {get_target_model_path(dest_dir)} ({get_model_size_mb(dest_dir):.1f} MB).")
        print("Pengunduhan dilewati (Anti-Redownload).")
        return

    print("Memulai pengunduhan model Qwen3-4B-Instruct-2507...")
    last_print = 0.0

    def _cli_progress(fraction: float, msg: str):
        nonlocal last_print
        now = time.time()
        if now - last_print > 0.5 or fraction >= 1.0:
            percent = int(fraction * 100)
            bar_len = 30
            filled = int(bar_len * fraction)
            bar = "=" * filled + "-" * (bar_len - filled)
            sys.stdout.write(f"\r[{bar}] {percent:3d}% | {msg}")
            sys.stdout.flush()
            last_print = now

    try:
        saved_path = download_qwen3_gguf(dest_dir, progress_callback=_cli_progress, force_download=args.force)
        print(f"\nSukses! File model tersimpan di: {saved_path}")
    except Exception as err:
        print(f"\nTerjadi kesalahan pengunduhan:\n{err}")
        sys.exit(1)

if __name__ == "__main__":
    main()
