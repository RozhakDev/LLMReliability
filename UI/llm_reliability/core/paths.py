"""Lokasi direktori proyek yang terpusat (satu sumber kebenaran untuk path lokal)."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = DATA_DIR / "models"
