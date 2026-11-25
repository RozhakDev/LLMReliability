import logging
from pathlib import Path
import shutil


def _dir_size_bytes(path: Path) -> int:
    total = 0
    try:
        if path.is_file() or path.is_symlink():
            return path.lstat().st_size if not path.is_symlink() else 0
        for item in path.rglob("*"):
            try:
                if item.is_file() and not item.is_symlink():
                    total += item.stat().st_size
            except OSError:
                continue
    except OSError:
        return 0
    return total


def log_top_disk_consumers(cache_dir: Path, logger: logging.Logger, limit: int = 5) -> None:
    scan_roots = (cache_dir, Path.home() / ".cache", Path("/tmp"))
    sized: list[tuple[int, Path]] = []
    for root in scan_roots:
        if not root.exists() or not root.is_dir():
            continue
        try:
            for entry in root.iterdir():
                if entry.name == "drive":
                    continue
                size_b = _dir_size_bytes(entry)
                if size_b >= 100 * 1024 * 1024:
                    sized.append((size_b, entry))
        except OSError:
            continue
    for size_b, entry in sorted(sized, key=lambda pair: pair[0], reverse=True)[:limit]:
        logger.warning("disk_top_consumer | gb=%.2f | path=%s", size_b / (1024**3), entry)


def _safe_rmtree(target: Path, logger: logging.Logger) -> bool:
    if not target.exists() or target.is_symlink() or not target.is_dir():
        return False
    try:
        shutil.rmtree(target, ignore_errors=False)
        return True
    except OSError as exc:
        logger.warning("hf_cache_remove_failed | path=%s | err=%s", target, exc)
        return False


def purge_other_hf_caches(
    active_hf_id: str,
    models_cache_dir: Path,
    logger: logging.Logger,
) -> int:
    active_folder = f"models--{active_hf_id.replace('/', '--')}"
    home_hf_dir = Path.home() / ".cache" / "huggingface"
    model_roots = (
        models_cache_dir,
        models_cache_dir / "hub",
        home_hf_dir / "hub",
    )
    removed_count = 0
    for root in model_roots:
        if not root.exists() or not root.is_dir():
            continue
        try:
            entries = list(root.iterdir())
        except OSError as exc:
            logger.warning("hf_cache_scan_failed | path=%s | err=%s", root, exc)
            continue
        for entry in entries:
            if entry.name.startswith("models--") and entry.name != active_folder and _safe_rmtree(entry, logger):
                logger.info("hf_stale_model_removed | active=%s | path=%s", active_hf_id, entry)
                removed_count += 1

    extra_dirs = (
        models_cache_dir / "xet",
        models_cache_dir / "hub" / "xet",
        home_hf_dir / "xet",
        home_hf_dir / "hub" / "xet",
        models_cache_dir / ".locks",
        models_cache_dir / "hub" / ".locks",
        home_hf_dir / "hub" / ".locks",
        Path.home() / ".cache" / "pip",
    )
    for extra in extra_dirs:
        if _safe_rmtree(extra, logger):
            logger.info("hf_aux_cache_removed | path=%s", extra)
            removed_count += 1

    if removed_count > 0 and models_cache_dir.exists():
        free_gb = shutil.disk_usage(models_cache_dir).free / (1024**3)
        logger.info("hf_preload_cleanup_done | removed=%d | free_gb=%.2f", removed_count, free_gb)
    return removed_count


def cleanup_hf_model_cache(
    hf_id: str,
    models_cache_dir: Path,
    logger: logging.Logger,
) -> bool:
    repo_folder_name = f"models--{hf_id.replace('/', '--')}"
    home_hf_dir = Path.home() / ".cache" / "huggingface"
    repo_candidates = [
        models_cache_dir / repo_folder_name,
        models_cache_dir / "hub" / repo_folder_name,
        home_hf_dir / "hub" / repo_folder_name,
    ]
    xet_candidates = [
        models_cache_dir / "xet",
        home_hf_dir / "xet",
    ]
    removed = False
    for target in repo_candidates:
        if _safe_rmtree(target, logger):
            logger.info("hf_cache_removed | hf_id=%s | path=%s", hf_id, target)
            removed = True

    for xet_dir in xet_candidates:
        if _safe_rmtree(xet_dir, logger):
            logger.info("hf_xet_cache_removed | path=%s", xet_dir)
            removed = True

    if not removed:
        logger.info("hf_cache_absent | hf_id=%s", hf_id)
    return removed
