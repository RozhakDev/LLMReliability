import logging
from typing import Any, Optional

import torch


def resolve_pad_token_id(processor_or_tokenizer: Any) -> int:
    tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None:
        return int(pad_id)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_id, (list, tuple)):
        return int(eos_id[0])
    if eos_id is not None:
        return int(eos_id)
    return 0


def configure_deterministic_generation(
    model: Any,
    logger: Optional[logging.Logger] = None,
) -> None:
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is None:
        return

    gen_cfg.do_sample = False
    cleared = []
    for attr in ("temperature", "top_p", "top_k", "min_p"):
        if hasattr(gen_cfg, attr) and getattr(gen_cfg, attr) is not None:
            setattr(gen_cfg, attr, None)
            cleared.append(attr)

    if logger and cleared:
        logger.info("gen_config_reset | cleared=%s", cleared)


def resolve_input_device(model: Any) -> torch.device:
    if hasattr(model, "get_input_embeddings"):
        try:
            embeddings = model.get_input_embeddings()
            weight = getattr(embeddings, "weight", None)
            if weight is not None and hasattr(weight, "device"):
                return weight.device
        except Exception:
            pass

    model_device = getattr(model, "device", None)
    if model_device is not None:
        return torch.device(model_device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def warn_if_offloaded(model: Any, model_key: str, logger: logging.Logger) -> None:
    hf_device_map = getattr(model, "hf_device_map", None)
    if not isinstance(hf_device_map, dict):
        return
    offloaded = [
        name for name, target_dev in hf_device_map.items() if str(target_dev).lower() in {"cpu", "disk"}
    ]
    if offloaded:
        logger.warning("model_offloaded | model=%s | modules=%d", model_key, len(offloaded))
