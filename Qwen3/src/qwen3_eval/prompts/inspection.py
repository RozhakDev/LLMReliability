from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from qwen3_eval.config.schemas import ModelSpec
from qwen3_eval.data.labels import order_candidates
from qwen3_eval.prompts.chat_formatter import build_chat_messages, format_chat_text
from qwen3_eval.prompts.templates import format_prompt


def _count_token_ids(input_ids: Any) -> int:
    if hasattr(input_ids, "shape"):
        shape = tuple(input_ids.shape)
        return int(shape[-1]) if shape else 0
    if isinstance(input_ids, (list, tuple)):
        if not input_ids:
            return 0
        first = input_ids[0]
        if isinstance(first, (list, tuple)):
            return len(first)
        return len(input_ids)
    return int(len(input_ids))


def build_prompt_inspection_table(
    row_data: Any,
    spec: ModelSpec,
    tokenizer: Optional[Any] = None,
    prompts: Sequence[str] = ("P1", "P2", "P3"),
    positions: Sequence[str] = ("AB", "BA"),
    prompts_config_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    input_txt = str(row_data["input"])
    records: List[Dict[str, object]] = []
    for p_id in prompts:
        for pos in positions:
            pos_str = str(pos)
            resp_a, resp_b, pos_target = order_candidates(row_data, pos_str)
            slot_a_src, slot_b_src = (
                ("response_1 (Jawaban 1)", "response_2 (Jawaban 2)")
                if pos_str == "AB"
                else ("response_2 (Jawaban 2)", "response_1 (Jawaban 1)")
            )
            user_prompt = format_prompt(input_txt, resp_a, resp_b, str(p_id), prompts_config_path)
            row_entry: Dict[str, object] = {
                "Prompt": str(p_id),
                "Position": pos_str,
                "Isi RESPONS A": slot_a_src,
                "Isi RESPONS B": slot_b_src,
                "Label Acuan Posisi": pos_target,
                "User Prompt Chars": len(user_prompt),
            }
            if tokenizer is not None:
                chat_text = format_chat_text(tokenizer, user_prompt, spec, prompts_config_path)
                encoded = tokenizer(chat_text, add_special_tokens=False)
                row_entry["Chat Template Tokens"] = _count_token_ids(encoded["input_ids"])
            records.append(row_entry)

    return pd.DataFrame(records)


def render_chat_inspection_text(
    row_data: Any,
    prompt_id: str,
    position: str,
    spec: ModelSpec,
    tokenizer: Optional[Any] = None,
    prompts_config_path: Optional[Union[str, Path]] = None,
) -> str:
    resp_a, resp_b, _ = order_candidates(row_data, position)
    user_prompt = format_prompt(str(row_data["input"]), resp_a, resp_b, prompt_id, prompts_config_path)
    if tokenizer is not None:
        return format_chat_text(tokenizer, user_prompt, spec, prompts_config_path)

    msgs = build_chat_messages(user_prompt, prompts_config_path=prompts_config_path)
    return f"[SYSTEM]\n{msgs[0]['content']}\n\n[USER]\n{msgs[1]['content']}"


def build_chat_template_preview_table(
    row_data: Any,
    prompt_id: str,
    spec: ModelSpec,
    positions: Sequence[str] = ("AB", "BA"),
    tokenizer: Optional[Any] = None,
    prompts_config_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for pos in positions:
        pos_str = str(pos)
        rendered = render_chat_inspection_text(
            row_data=row_data,
            prompt_id=prompt_id,
            position=pos_str,
            spec=spec,
            tokenizer=tokenizer,
            prompts_config_path=prompts_config_path,
        )
        rows.append(
            {
                "Kondisi": f"{prompt_id}_{pos_str}",
                "Panjang Karakter": len(rendered),
                "Chat Template": rendered,
            }
        )
    return pd.DataFrame(rows)
