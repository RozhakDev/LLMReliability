import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from llm_exp.config.schemas import ModelSpec
from llm_exp.prompts.templates import get_system_prompt


def build_chat_messages(
    user_prompt: str,
    structured_content: bool = False,
    system_prompt: Optional[str] = None,
    prompts_config_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    sys_msg = system_prompt if system_prompt is not None else get_system_prompt(prompts_config_path)
    if structured_content:
        return [
            {"role": "system", "content": [{"type": "text", "text": sys_msg}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_prompt},
    ]


def _resolve_text_tokenizer(processor_or_tokenizer: Any) -> Any:
    return getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)


def format_chat_text(
    processor_or_tokenizer: Any,
    user_prompt: str,
    spec: ModelSpec,
    prompts_config_path: Optional[Union[str, Path]] = None,
) -> str:
    tokenizer = _resolve_text_tokenizer(processor_or_tokenizer)
    messages = build_chat_messages(
        user_prompt,
        structured_content=False,
        prompts_config_path=prompts_config_path,
    )

    kwargs: Dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if spec.use_enable_thinking_flag:
        kwargs["enable_thinking"] = False

    try:
        return str(tokenizer.apply_chat_template(messages, **kwargs))
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return str(tokenizer.apply_chat_template(messages, **kwargs))


def prepare_model_inputs(
    processor_or_tokenizer: Any,
    user_prompt: str,
    spec: ModelSpec,
    device: Union[torch.device, str],
    logger: Optional[logging.Logger] = None,
    prompts_config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, torch.Tensor]:
    if spec.is_multimodal_processor:
        structured_msgs = build_chat_messages(
            user_prompt,
            structured_content=True,
            prompts_config_path=prompts_config_path,
        )
        try:
            encoded = processor_or_tokenizer.apply_chat_template(
                structured_msgs,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            return {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in encoded.items()
            }
        except Exception as exc:
            if logger is not None:
                logger.warning("chat_template_fallback | model=%s | err=%s", spec.key, exc)

    formatted_text = format_chat_text(
        processor_or_tokenizer,
        user_prompt,
        spec,
        prompts_config_path=prompts_config_path,
    )
    tokenizer = _resolve_text_tokenizer(processor_or_tokenizer)
    encoded = tokenizer(
        formatted_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    inputs: Dict[str, Any] = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in encoded.items()
    }
    if not spec.is_multimodal_processor:
        inputs.pop("token_type_ids", None)
        inputs.pop("return_token_type_ids", None)
    return inputs
