import logging
import os
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from transformers.utils import logging as hf_logging

from llm_exp.config.schemas import ModelSpec
from llm_exp.models.runtime_utils import (
    configure_deterministic_generation,
    resolve_input_device,
    warn_if_offloaded,
)


def _resolve_model_class(loader_type: str) -> Any:
    if loader_type == "mistral3":
        if hasattr(transformers, "Mistral3ForConditionalGeneration"):
            return transformers.Mistral3ForConditionalGeneration
        if hasattr(transformers, "AutoModelForImageTextToText"):
            return transformers.AutoModelForImageTextToText
        raise ImportError("Mistral Small 3.1 requires transformers>=4.51.0 with Mistral3ForConditionalGeneration.")
    if loader_type == "gemma3" and hasattr(transformers, "Gemma3ForConditionalGeneration"):
        return transformers.Gemma3ForConditionalGeneration
    return AutoModelForCausalLM


def _resolve_active_hf_token() -> Optional[str]:
    try:
        from huggingface_hub import get_token as hf_get_token

        hub_token = hf_get_token()
        if hub_token and str(hub_token).strip():
            return str(hub_token).strip()
    except Exception:
        pass
    env_token = os.environ.get("HF_TOKEN", "").strip()
    return env_token or None


def _resolve_hf_username(token: Optional[str]) -> str:
    try:
        from huggingface_hub import whoami as hf_whoami

        info = hf_whoami(token=token) if token else hf_whoami()
        if isinstance(info, dict) and info.get("name"):
            return str(info["name"])
    except Exception:
        pass
    return "unauthenticated"


def _load_tokenizer_or_processor(
    spec: ModelSpec,
    cache_dir: Path,
    logger: logging.Logger,
    token: Optional[str] = None,
) -> Any:
    common_kwargs: dict[str, Any] = {
        "cache_dir": str(cache_dir),
        "trust_remote_code": spec.trust_remote_code,
    }
    if token:
        common_kwargs["token"] = token
    if spec.is_multimodal_processor:
        try:
            return AutoProcessor.from_pretrained(spec.hf_id, **common_kwargs)
        except Exception as exc:
            logger.warning("processor_fallback | hf_id=%s | to=AutoTokenizer | err=%s", spec.hf_id, exc)
    return AutoTokenizer.from_pretrained(spec.hf_id, **common_kwargs)


def _from_pretrained_16bit(model_cls: Any, hf_id: str, dtype: torch.dtype, **kwargs: Any) -> Any:
    try:
        return model_cls.from_pretrained(hf_id, dtype=dtype, **kwargs)
    except TypeError:
        return model_cls.from_pretrained(hf_id, torch_dtype=dtype, **kwargs)


def load_model_and_processor(
    spec: ModelSpec,
    cache_dir: Path,
    dtype: torch.dtype,
    logger: logging.Logger,
) -> Tuple[Any, Any]:
    hf_logging.disable_progress_bar()
    hf_token = _resolve_active_hf_token()
    if spec.requires_gated_access and not hf_token:
        logger.warning("model_gated_warn | hf_id=%s | action=set_hf_token_and_accept_license", spec.hf_id)
    logger.info("model_load_start | model=%s | hf_id=%s | est_vram_gb=%.1f", spec.key, spec.hf_id, spec.estimated_vram_gb)

    try:
        processor_or_tokenizer = _load_tokenizer_or_processor(spec, cache_dir, logger, token=hf_token)
        model_cls = _resolve_model_class(spec.loader_type)
        load_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "cache_dir": str(cache_dir),
            "trust_remote_code": spec.trust_remote_code,
        }
        if hf_token:
            load_kwargs["token"] = hf_token
        logger.info("model_weights_load | cls=%s | dtype=%s | path=%s", model_cls.__name__, dtype, cache_dir)
        try:
            model = _from_pretrained_16bit(model_cls, spec.hf_id, dtype, **load_kwargs)
        except ValueError:
            if model_cls is AutoModelForCausalLM:
                raise
            logger.warning("model_cls_fallback | from=%s | to=AutoModelForCausalLM | hf_id=%s", model_cls.__name__, spec.hf_id)
            model = _from_pretrained_16bit(AutoModelForCausalLM, spec.hf_id, dtype, **load_kwargs)

        model.eval()
        configure_deterministic_generation(model, logger=logger)
    except Exception as exc:
        if any(tok in str(exc).lower() for tok in ("401", "403", "gated", "restricted", "access")):
            hf_user = _resolve_hf_username(hf_token)
            logger.error(
                "hf_access_denied | hf_id=%s | hf_user=%s | url=https://huggingface.co/%s | action=accept_license_and_check_gated_token_scope",
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
    return model, processor_or_tokenizer
