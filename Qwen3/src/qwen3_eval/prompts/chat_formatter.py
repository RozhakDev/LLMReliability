from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from qwen3_eval.config.schemas import ModelSpec
from qwen3_eval.prompts.templates import get_system_prompt


def build_chat_messages(
    user_prompt: str,
    system_prompt: Optional[str] = None,
    prompts_config_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, str]]:
    sys_msg = system_prompt if system_prompt is not None else get_system_prompt(prompts_config_path)
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_prompt},
    ]


def format_chat_text(
    tokenizer: Any,
    user_prompt: str,
    spec: ModelSpec,
    prompts_config_path: Optional[Union[str, Path]] = None,
) -> str:
    messages = build_chat_messages(
        user_prompt,
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
    tokenizer: Any,
    user_prompt: str,
    spec: ModelSpec,
    device: Union[torch.device, str],
    prompts_config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, torch.Tensor]:
    formatted_text = format_chat_text(
        tokenizer,
        user_prompt,
        spec,
        prompts_config_path=prompts_config_path,
    )
    encoded = tokenizer(
        formatted_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    inputs: Dict[str, Any] = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in encoded.items()
    }
    inputs.pop("token_type_ids", None)
    inputs.pop("return_token_type_ids", None)
    return inputs
