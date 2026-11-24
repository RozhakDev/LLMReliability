import json
import logging
from typing import Callable
from unittest.mock import patch

import pandas as pd
import pytest

pytest.importorskip("transformers")

from qwen3_eval.config import ExperimentConfig, ExperimentPaths, load_model_spec
from qwen3_eval.inference.runner import run_inference
from qwen3_eval.inference.tasks import load_processed_combinations

pytestmark = [pytest.mark.integration, pytest.mark.transformers]


def test_runner_preserves_cache_on_partial_or_crash_and_cleans_on_completion(
    qwen3_config: ExperimentConfig,
    qwen3_paths: ExperimentPaths,
    make_eval_df: Callable[[int], pd.DataFrame],
    fake_model_factory,
    fake_tokenizer,
    test_logger: logging.Logger,
) -> None:
    spec = load_model_spec(qwen3_config.model_config_path)
    df_eval = make_eval_df(1)
    out_file = qwen3_paths.results_dir / qwen3_config.artifacts.output_jsonl
    stale_records = [
        {"sample_id": 9990 + idx, "prompt_id": "P1", "position": "AB", "raw_output": "A"}
        for idx in range(6)
    ]
    out_file.write_text("".join(json.dumps(r) + "\n" for r in stale_records), encoding="utf-8")

    with (
        patch("qwen3_eval.inference.runner.load_model_and_tokenizer", return_value=(fake_model_factory(), fake_tokenizer)),
        patch("qwen3_eval.inference.runner.cleanup_hf_model_cache") as mock_cleanup,
    ):
        run_inference(spec, df_eval, qwen3_paths, test_logger, config=qwen3_config, cleanup_cache_after=True, max_tasks=2)
        assert mock_cleanup.call_count == 0

    with (
        patch("qwen3_eval.inference.runner.load_model_and_tokenizer", return_value=(fake_model_factory(fail_on_call=2), fake_tokenizer)),
        patch("qwen3_eval.inference.runner.cleanup_hf_model_cache") as mock_cleanup,
    ):
        with pytest.raises(RuntimeError, match="Simulated mid-run crash"):
            run_inference(spec, df_eval, qwen3_paths, test_logger, config=qwen3_config, cleanup_cache_after=True)
        assert mock_cleanup.call_count == 0

    with (
        patch("qwen3_eval.inference.runner.load_model_and_tokenizer", return_value=(fake_model_factory(), fake_tokenizer)),
        patch("qwen3_eval.inference.runner.cleanup_hf_model_cache") as mock_cleanup,
    ):
        returned_path = run_inference(spec, df_eval, qwen3_paths, test_logger, config=qwen3_config, cleanup_cache_after=True)
        assert mock_cleanup.call_count == 1
        assert returned_path == out_file
        processed = load_processed_combinations(returned_path)
        assert {(100, p, pos) for p in ("P1", "P2", "P3") for pos in ("AB", "BA")}.issubset(processed)


def test_runner_rejects_non_positive_max_tasks(
    qwen3_config: ExperimentConfig,
    qwen3_paths: ExperimentPaths,
    make_eval_df: Callable[[int], pd.DataFrame],
    test_logger: logging.Logger,
) -> None:
    spec = load_model_spec(qwen3_config.model_config_path)
    with pytest.raises(ValueError, match="max_tasks"):
        run_inference(spec, make_eval_df(1), qwen3_paths, test_logger, config=qwen3_config, max_tasks=0)
