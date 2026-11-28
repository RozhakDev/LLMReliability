import gc
import logging
from pathlib import Path
import shutil
from typing import Optional

import torch


def resolve_torch_dtype(logger: logging.Logger) -> torch.dtype:
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    selected_dtype = torch.bfloat16 if bf16_supported else torch.float16

    if bf16_supported:
        torch.backends.cuda.matmul.allow_tf32 = True

    logger.info(
        "precision_set | dtype=%s | bf16=%s | tf32=%s | quant=none",
        selected_dtype,
        bf16_supported,
        bf16_supported,
    )
    return selected_dtype


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


def _log_top_disk_consumers(cache_dir: Path, logger: logging.Logger, limit: int = 5) -> None:
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


def log_hardware_status(
    logger: logging.Logger,
    cache_dir: Optional[Path] = None,
    min_vram_gb: float = 0.0,
    min_disk_gb: float = 0.0,
) -> None:
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        free_gb = free_bytes / (1024**3)
        total_gb = total_bytes / (1024**3)
        logger.info(
            "gpu_info | name=%s | free_gb=%.2f | total_gb=%.2f",
            device_name,
            free_gb,
            total_gb,
        )
        if min_vram_gb > 0.0 and total_gb < min_vram_gb:
            logger.warning("vram_low | total_gb=%.2f | min_gb=%.1f", total_gb, min_vram_gb)
    else:
        logger.warning("gpu_unavailable | fallback=cpu")

    if cache_dir is not None and cache_dir.exists():
        usage = shutil.disk_usage(cache_dir)
        disk_free_gb = usage.free / (1024**3)
        logger.info("disk_info | free_gb=%.2f | path=%s", disk_free_gb, cache_dir)
        if min_disk_gb > 0.0 and disk_free_gb < min_disk_gb:
            logger.warning("disk_low | free_gb=%.2f | min_gb=%.1f", disk_free_gb, min_disk_gb)
            _log_top_disk_consumers(cache_dir, logger)


def clear_gpu_memory(logger: Optional[logging.Logger] = None) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        reserved_gb = torch.cuda.memory_reserved() / (1024**3)
        if logger:
            logger.info("vram_cleared | alloc_gb=%.2f | res_gb=%.2f", allocated_gb, reserved_gb)


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
        for entry in root.iterdir():
            if entry.is_dir() and entry.name.startswith("models--") and entry.name != active_folder:
                shutil.rmtree(entry, ignore_errors=False)
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
        if extra.exists() and extra.is_dir():
            shutil.rmtree(extra, ignore_errors=False)
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
        if target.exists() and target.is_dir():
            shutil.rmtree(target, ignore_errors=False)
            logger.info("hf_cache_removed | hf_id=%s | path=%s", hf_id, target)
            removed = True

    for xet_dir in xet_candidates:
        if xet_dir.exists() and xet_dir.is_dir():
            shutil.rmtree(xet_dir, ignore_errors=False)
            logger.info("hf_xet_cache_removed | path=%s", xet_dir)
            removed = True

    if not removed:
        logger.info("hf_cache_absent | hf_id=%s", hf_id)
    return removed
