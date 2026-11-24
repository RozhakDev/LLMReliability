import gc
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional

import pandas as pd
import torch

from qwen3_eval.config import ExperimentConfig, ExperimentPaths, ModelSpec
from qwen3_eval.inference.tasks import (
    build_evaluation_tasks,
    determine_human_winner,
    load_processed_combinations,
)
from qwen3_eval.models.loader import load_model_and_tokenizer
from qwen3_eval.models.runtime_utils import resolve_input_device, resolve_pad_token_id
from qwen3_eval.prompts.chat_formatter import prepare_model_inputs
from qwen3_eval.prompts.templates import format_prompt
from qwen3_eval.utils.gpu import (
    cleanup_hf_model_cache,
    clear_gpu_memory,
    log_hardware_status,
    purge_other_hf_caches,
    resolve_torch_dtype,
)


def _run_single_task(
    task: Dict[str, Any],
    model: Any,
    tokenizer: Any,
    spec: ModelSpec,
    cfg: ExperimentConfig,
    device: torch.device,
    dtype: torch.dtype,
    pad_token_id: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    row = task["row"]
    position = str(task["position"])
    human_winner = determine_human_winner(row)
    if position == "AB":
        resp_a, resp_b, current_human = str(row["response_1"]), str(row["response_2"]), human_winner
    else:
        resp_a, resp_b = str(row["response_2"]), str(row["response_1"])
        current_human = "B" if human_winner == "A" else "A"

    step_start = time.time()
    user_prompt = format_prompt(str(row["input"]), resp_a, resp_b, str(task["prompt_id"]), cfg.prompts_config_path)
    inputs = prepare_model_inputs(tokenizer, user_prompt, spec, device, cfg.prompts_config_path)
    prompt_len = int(inputs["input_ids"].shape[1])

    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=cfg.inference.do_sample, pad_token_id=pad_token_id
    )
    gen_tokens = outputs[0][prompt_len:]
    raw_out = str(tokenizer.decode(gen_tokens, skip_special_tokens=True))
    runtime_sec = round(time.time() - step_start, 2)
    n_toks = int(len(gen_tokens))
    del inputs, outputs, gen_tokens

    return {
        "sample_id": int(task["sample_id"]),
        "domain": str(row["domain"]),
        "prompt_id": str(task["prompt_id"]),
        "position": position,
        "human_preference": current_human,
        "raw_output": raw_out,
        "runtime_sec": runtime_sec,
        "output_tokens": n_toks,
        "tps": round(n_toks / runtime_sec, 2) if runtime_sec > 0 else 0.0,
        "model_key": spec.key,
        "model_id": spec.hf_id,
        "dtype": str(dtype).replace("torch.", ""),
    }


def run_inference(
    spec: ModelSpec,
    df_eval: pd.DataFrame,
    paths: ExperimentPaths,
    logger: logging.Logger,
    config: Optional[ExperimentConfig] = None,
    max_new_tokens: Optional[int] = None,
    log_interval: Optional[int] = None,
    cleanup_cache_after: Optional[bool] = None,
    max_tasks: Optional[int] = None,
) -> Path:
    if max_tasks is not None and max_tasks < 1:
        raise ValueError(f"max_tasks must be >= 1 when specified, got {max_tasks}.")

    cfg = config or paths.config
    inf = cfg.inference
    max_toks = max_new_tokens if max_new_tokens is not None else inf.max_new_tokens
    log_every = log_interval if log_interval is not None else inf.log_interval
    if max_toks < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_toks}.")
    if log_every < 1:
        raise ValueError(f"log_interval must be >= 1, got {log_every}.")
    do_cleanup = cleanup_cache_after if cleanup_cache_after is not None else inf.cleanup_cache_after
    gc_every = max(inf.gc_interval, 1)

    results_dir = paths.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / cfg.artifacts.output_jsonl

    processed = load_processed_combinations(out_file, logger=logger)
    all_tasks = build_evaluation_tasks(df_eval, cfg.evaluation.prompts, cfg.evaluation.positions)
    expected_keys = {t["key"] for t in all_tasks}
    pending_all = [t for t in all_tasks if t["key"] not in processed]
    already_done = len(all_tasks) - len(pending_all)
    pending = pending_all[:max_tasks] if max_tasks is not None else pending_all

    logger.info("resume_status | model=%s | total=%d | done=%d | pending=%d", spec.key, len(all_tasks), already_done, len(pending))
    if not pending:
        logger.info("infer_skip | model=%s | reason=already_done", spec.key)
        return out_file

    if do_cleanup:
        purge_other_hf_caches(spec.hf_id, paths.models_cache_dir, logger)
    log_hardware_status(logger, paths.models_cache_dir, inf.min_vram_gb, inf.min_disk_gb)
    dtype = resolve_torch_dtype(logger)

    model, tokenizer, completed_full = None, None, False
    try:
        model, tokenizer = load_model_and_tokenizer(spec, paths.models_cache_dir, dtype, logger)
        pad_id, device = resolve_pad_token_id(tokenizer), resolve_input_device(model)
        step, session_start = already_done, time.time()

        with torch.no_grad():
            for idx, task in enumerate(pending, start=1):
                step += 1
                rec = _run_single_task(task, model, tokenizer, spec, cfg, device, dtype, pad_id, max_toks)
                with open(out_file, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    handle.flush()
                processed.add((rec["sample_id"], rec["prompt_id"], rec["position"]))

                if (step % log_every == 0) or (idx >= len(pending)):
                    elapsed = time.time() - session_start
                    eta_min = ((len(pending) - idx) * (elapsed / max(idx, 1))) / 60.0
                    logger.info(
                        "infer_progress | model=%s | pct=%.1f | step=%d/%d | id=%d | prompt=%s | pos=%s | tps=%.2f | eta_min=%.1f",
                        spec.key,
                        (step / len(all_tasks)) * 100.0,
                        step,
                        len(all_tasks),
                        rec["sample_id"],
                        rec["prompt_id"],
                        rec["position"],
                        rec["tps"],
                        eta_min,
                    )

                if idx % gc_every == 0 or idx >= len(pending):
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        completed_full = expected_keys.issubset(processed)
        done_count = len(expected_keys.intersection(processed))
        logger.info("infer_done | model=%s | done=%d/%d | complete=%s", spec.key, done_count, len(all_tasks), completed_full)
    finally:
        model, tokenizer = None, None
        clear_gpu_memory(logger=logger)
        if do_cleanup and completed_full:
            cleanup_hf_model_cache(spec.hf_id, paths.models_cache_dir, logger)
        elif do_cleanup:
            logger.info("hf_cache_kept | model=%s | reason=incomplete_run", spec.key)
    return out_file
