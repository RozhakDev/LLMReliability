import logging
import os
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

from qwen3_eval.config.schemas import ModelSpec
from qwen3_eval.models.runtime_utils import (
    configure_deterministic_generation,
    resolve_input_device,
    warn_if_offloaded,
)


def _resolve_active_hf_token() -> Optional[str]:
    try:
        from huggingface_hub import get_token as hf_get_token

        hub_token = hf_get_token()
        if hub_token and str(hub_token).strip():
            return str(hub_token).strip()
    except (ImportError, OSError, RuntimeError, ValueError):
        pass
    env_token = os.environ.get("HF_TOKEN", "").strip()
    return env_token or None


def _resolve_hf_username(token: Optional[str]) -> str:
    try:
        from huggingface_hub import whoami as hf_whoami

        info = hf_whoami(token=token) if token else hf_whoami()
        if isinstance(info, dict) and info.get("name"):
            return str(info["name"])
    except (ImportError, OSError, RuntimeError, ValueError):
        pass
    return "unauthenticated"


def _from_pretrained_16bit(hf_id: str, dtype: torch.dtype, **kwargs: Any) -> Any:
    try:
        return AutoModelForCausalLM.from_pretrained(hf_id, dtype=dtype, **kwargs)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype, **kwargs)


def load_model_and_tokenizer(
    spec: ModelSpec,
    cache_dir: Path,
    dtype: torch.dtype,
    logger: logging.Logger,
) -> Tuple[Any, Any]:
    hf_logging.disable_progress_bar()
    hf_token = _resolve_active_hf_token()
    logger.info("model_load_start | model=%s | hf_id=%s | est_vram_gb=%.1f", spec.key, spec.hf_id, spec.estimated_vram_gb)

    try:
        tok_kwargs: dict[str, Any] = {
            "cache_dir": str(cache_dir),
            "trust_remote_code": spec.trust_remote_code,
        }
        if hf_token:
            tok_kwargs["token"] = hf_token
        tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, **tok_kwargs)

        load_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "cache_dir": str(cache_dir),
            "trust_remote_code": spec.trust_remote_code,
        }
        if hf_token:
            load_kwargs["token"] = hf_token
        logger.info("model_weights_load | cls=AutoModelForCausalLM | dtype=%s | path=%s", dtype, cache_dir)
        model = _from_pretrained_16bit(spec.hf_id, dtype, **load_kwargs)
        model.eval()
        configure_deterministic_generation(model, logger=logger)
    except Exception as exc:
        if any(tok in str(exc).lower() for tok in ("401", "403", "gated", "restricted", "access")):
            hf_user = _resolve_hf_username(hf_token)
            logger.error(
                "hf_access_denied | hf_id=%s | hf_user=%s | url=https://huggingface.co/%s | action=check_hf_token_permissions",
                spec.hf_id,
                hf_user,
                spec.hf_id,
            )
        raise

    warn_if_offloaded(model, spec.key, logger)
    if torch.cuda.is_available():
        alloc_gb = torch.cuda.memory_allocated() / (1024**3)
        res_gb = torch.cuda.memory_reserved() / (1024**3)
        logger.info(
            "model_loaded | model=%s | device=%s | dtype=%s | alloc_gb=%.2f | res_gb=%.2f",
            spec.key,
            resolve_input_device(model),
            getattr(model, "dtype", dtype),
            alloc_gb,
            res_gb,
        )
    else:
        logger.info("model_loaded | model=%s | device=cpu", spec.key)
    return model, tokenizer
