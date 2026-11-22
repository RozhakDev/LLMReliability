import logging
from typing import Callable
from unittest.mock import patch

import pandas as pd
import pytest

pytest.importorskip("transformers")

from llm_exp.config import ExperimentConfig, ExperimentPaths, get_model_spec
from llm_exp.inference.runner import run_model_inference
from llm_exp.inference.tasks import load_processed_combinations

pytestmark = [pytest.mark.integration, pytest.mark.transformers]


def test_runner_preserves_cache_on_partial_or_crash_and_cleans_on_completion(
    exp_config: ExperimentConfig,
    exp_paths: ExperimentPaths,
    make_eval_df: Callable[[int], pd.DataFrame],
    fake_model_factory,
    fake_tokenizer,
    test_logger: logging.Logger,
) -> None:
    spec = get_model_spec("qwen3_32b", exp_config.models_config_path)
    df_eval = make_eval_df(1)

    with (
        patch("llm_exp.inference.runner.load_model_and_processor", return_value=(fake_model_factory(), fake_tokenizer)),
        patch("llm_exp.inference.runner.cleanup_hf_model_cache") as mock_cleanup,
    ):
        run_model_inference(spec, df_eval, exp_paths, test_logger, config=exp_config, cleanup_cache_after=True, max_tasks=2)
        assert mock_cleanup.call_count == 0

    with (
        patch("llm_exp.inference.runner.load_model_and_processor", return_value=(fake_model_factory(fail_on_call=2), fake_tokenizer)),
        patch("llm_exp.inference.runner.cleanup_hf_model_cache") as mock_cleanup,
    ):
        with pytest.raises(RuntimeError, match="Simulated mid-run crash"):
            run_model_inference(spec, df_eval, exp_paths, test_logger, config=exp_config, cleanup_cache_after=True)
        assert mock_cleanup.call_count == 0

    with (
        patch("llm_exp.inference.runner.load_model_and_processor", return_value=(fake_model_factory(), fake_tokenizer)),
        patch("llm_exp.inference.runner.cleanup_hf_model_cache") as mock_cleanup,
    ):
        out_file = run_model_inference(spec, df_eval, exp_paths, test_logger, config=exp_config, cleanup_cache_after=True)
        assert mock_cleanup.call_count == 1
        assert len(load_processed_combinations(out_file)) == 6
