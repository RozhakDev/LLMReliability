import gc
import logging
from pathlib import Path
import shutil
from typing import Optional

import torch

from qwen3_eval.utils.cache import log_top_disk_consumers


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
            log_top_disk_consumers(cache_dir, logger)


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
