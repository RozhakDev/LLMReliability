import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict

import pandas as pd
import pytest

from qwen3_eval.config import ExperimentConfig, ExperimentPaths, load_model_spec
from qwen3_eval.data import (
    build_domain_distribution,
    build_preprocessing_summary,
    build_sample_detail_table,
    build_sample_overview_table,
)
from qwen3_eval.metrics import (
    build_paired_positional_frame,
    build_parsed_predictions_preview,
    build_positional_bias_diagnostics,
    summarize_positional_contingency,
)
from qwen3_eval.prompts import (
    build_chat_template_preview_table,
    build_prompt_inspection_table,
    render_chat_inspection_text,
)

pytestmark = pytest.mark.unit


class _ListTokenizer:
    def apply_chat_template(
        self,
        messages: Any,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: Any = None,
    ) -> str:
        return (
            f"<|im_start|>system\n{messages[0]['content']}\n"
            f"<|im_start|>user\n{messages[1]['content']}\n<think>\n\n</think>\n"
        )

    def __call__(self, text: str, add_special_tokens: bool = False) -> Dict[str, Any]:
        return {"input_ids": [101, 102, 103, 104]}


def test_data_profiling_summary_and_domain_distribution(
    qwen3_config: ExperimentConfig,
    qwen3_paths: ExperimentPaths,
    make_eval_df: Callable[[int], pd.DataFrame],
) -> None:
    df_eval = make_eval_df(6)
    summary_no_raw = build_preprocessing_summary(df_eval, qwen3_paths, qwen3_config)
    assert "Silver Unique Prompts (Cohort)" in summary_no_raw["Stage / Check"].tolist()

    df_raw = pd.concat([df_eval, df_eval.iloc[:2]], ignore_index=True)
    qwen3_paths.raw_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_parquet(qwen3_paths.raw_parquet_path, index=False)

    summary_with_raw = build_preprocessing_summary(df_eval, qwen3_paths, qwen3_config)
    checks = dict(zip(summary_with_raw["Stage / Check"], summary_with_raw["Value"]))
    assert checks["Bronze Raw Records"] == "8 rows"
    assert checks["Duplicate original_id Removed"] == "2 rows"
    assert "36 inferences" in checks["Evaluation Task Matrix"]

    dom_df = build_domain_distribution(df_eval)
    assert int(dom_df["Jumlah Sampel"].sum()) == 6
    assert pytest.approx(float(dom_df["Persentase (%)"].sum()), abs=0.05) == 100.0
    assert build_domain_distribution(pd.DataFrame()).empty


def test_sample_overview_and_detail_tables(
    make_eval_df: Callable[[int], pd.DataFrame],
) -> None:
    df_eval = make_eval_df(3)
    df_eval.loc[0, "input"] = "Kata " * 40
    overview = build_sample_overview_table(df_eval, n_rows=2, max_chars=30)
    assert len(overview) == 2
    assert str(overview.loc[0, "input"]).endswith("...")

    meta_df, content_df = build_sample_detail_table(df_eval, row_index=0)
    assert len(meta_df) == 6
    assert list(content_df.columns) == ["Bagian", "Panjang Karakter", "Teks"]
    assert len(content_df) == 3
    assert content_df.loc[0, "Teks"] == "Kata " * 40

    with pytest.raises(ValueError, match="must not be empty"):
        build_sample_detail_table(pd.DataFrame())


def test_load_tokenizer_only_invokes_pretrained(
    qwen3_config: ExperimentConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qwen3_eval.models import loader as loader_mod

    captured: Dict[str, Any] = {}

    def _fake_from_pretrained(hf_id: str, **kwargs: Any) -> str:
        captured["hf_id"] = hf_id
        captured["kwargs"] = kwargs
        return "sentinel_tokenizer"

    monkeypatch.setattr(loader_mod.AutoTokenizer, "from_pretrained", _fake_from_pretrained)
    monkeypatch.setattr(loader_mod, "_resolve_active_hf_token", lambda: None)

    spec = load_model_spec(qwen3_config.model_config_path)
    tok = loader_mod.load_tokenizer_only(spec, tmp_path)
    assert tok == "sentinel_tokenizer"
    assert captured["hf_id"] == spec.hf_id
    assert captured["kwargs"]["cache_dir"] == str(tmp_path)
    assert captured["kwargs"]["trust_remote_code"] == spec.trust_remote_code


def test_prompt_and_chat_inspection_tables(
    qwen3_config: ExperimentConfig,
    make_eval_df: Callable[[int], pd.DataFrame],
    fake_tokenizer: Any,
) -> None:
    spec = load_model_spec(qwen3_config.model_config_path)
    row = make_eval_df(1).iloc[0]

    df_no_tok = build_prompt_inspection_table(
        row,
        spec,
        tokenizer=None,
        prompts_config_path=qwen3_config.prompts_config_path,
    )
    assert len(df_no_tok) == 6
    assert "Chat Template Tokens" not in df_no_tok.columns
    assert df_no_tok.loc[0, "Label Acuan Posisi"] == "A"
    assert df_no_tok.loc[1, "Label Acuan Posisi"] == "B"

    df_tensor_tok = build_prompt_inspection_table(
        row,
        spec,
        tokenizer=fake_tokenizer,
        prompts_config_path=qwen3_config.prompts_config_path,
    )
    assert df_tensor_tok["Chat Template Tokens"].tolist() == [3] * 6

    df_list_tok = build_prompt_inspection_table(
        row,
        spec,
        tokenizer=_ListTokenizer(),
        prompts_config_path=qwen3_config.prompts_config_path,
    )
    assert df_list_tok["Chat Template Tokens"].tolist() == [4] * 6

    rendered_tok = render_chat_inspection_text(
        row,
        "P1",
        "AB",
        spec,
        tokenizer=_ListTokenizer(),
        prompts_config_path=qwen3_config.prompts_config_path,
    )
    assert "<think>" in rendered_tok
    assert str(row["response_1"]) in rendered_tok

    rendered_plain = render_chat_inspection_text(
        row,
        "P1",
        "BA",
        spec,
        tokenizer=None,
        prompts_config_path=qwen3_config.prompts_config_path,
    )
    assert rendered_plain.startswith("[SYSTEM]")

    preview_df = build_chat_template_preview_table(
        row,
        "P1",
        spec,
        positions=("AB", "BA"),
        tokenizer=_ListTokenizer(),
        prompts_config_path=qwen3_config.prompts_config_path,
    )
    assert list(preview_df.columns) == ["Kondisi", "Panjang Karakter", "Chat Template"]
    assert preview_df["Kondisi"].tolist() == ["P1_AB", "P1_BA"]
    assert "<think>" in str(preview_df.loc[0, "Chat Template"])


def test_metrics_preview_and_positional_diagnostics() -> None:
    df_valid = pd.DataFrame(
        [
            {
                "sample_id": 100,
                "domain": "SHP-AskEngineers",
                "prompt_id": "P1",
                "position": "AB",
                "raw_output": "A",
                "parsed_decision": "A",
                "logical_decision": 1,
                "human_preference": "A",
                "logical_human": 1,
                "is_correct": 100,
                "output_tokens": 1,
                "runtime_sec": 0.42,
                "tps": 2.38,
            },
            {
                "sample_id": 100,
                "domain": "SHP-AskEngineers",
                "prompt_id": "P1",
                "position": "BA",
                "raw_output": "A",
                "parsed_decision": "A",
                "logical_decision": 2,
                "human_preference": "B",
                "logical_human": 1,
                "is_correct": 0,
                "output_tokens": 1,
                "runtime_sec": 0.40,
                "tps": 2.50,
            },
        ]
    )
    fake_artifacts = SimpleNamespace(
        df_all=df_valid,
        df_valid=df_valid,
        invalid_count=0,
        position_consistency=0.0,
        position_flip=100.0,
        first_pos_rate=100.0,
        mcnemar_stat=1.0,
        mcnemar_pvalue=0.01,
        is_significant=True,
    )

    preview = build_parsed_predictions_preview(fake_artifacts, n_rows=6)
    assert len(preview) == 2
    assert preview["mapped_choice"].tolist() == ["Resp. 1", "Resp. 2"]

    diag = build_positional_bias_diagnostics(fake_artifacts, alpha=0.05)
    metrics_map = dict(zip(diag["Diagnostic Metric"], diag["Value"]))
    assert "Slot-A Bias b (AB=1, BA=2): 1" in metrics_map["Discordant Pairs (Position Flip)"]
    assert "Significant Position Bias" in metrics_map["McNemar Continuity-Corrected (alpha=0.05)"]
    assert "Mean Latency" in metrics_map["Inference Runtime Profile"]


def test_plot_dataset_profile_exports_png_and_svg(
    tmp_path: Path,
    make_eval_df: Callable[[int], pd.DataFrame],
    test_logger: logging.Logger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from qwen3_eval.visualization import plot_dataset_profile

    df_eval = make_eval_df(9)
    plot_dataset_profile(df_eval, tmp_path, test_logger, dpi=72, show_inline=False)
    for base_name in ("plot_dataset_domain_distribution", "plot_dataset_preference_balance"):
        assert (tmp_path / f"{base_name}.png").is_file()
        assert (tmp_path / f"{base_name}.svg").is_file()

    with caplog.at_level(logging.WARNING, logger=test_logger.name):
        plot_dataset_profile(pd.DataFrame(), tmp_path, test_logger, dpi=72, show_inline=False)
    assert "plot_skip | figure=dataset_profile | reason=empty_dataset" in caplog.text


def test_paired_positional_contingency_counts() -> None:
    empty_summary = summarize_positional_contingency(build_paired_positional_frame(pd.DataFrame()))
    assert empty_summary.n_pairs == 0
    assert empty_summary.slot_a_bias_b == 0

    df_valid = pd.DataFrame(
        [
            {"sample_id": 1, "prompt_id": "P1", "position": "AB", "logical_decision": 1},
            {"sample_id": 1, "prompt_id": "P1", "position": "BA", "logical_decision": 1},
            {"sample_id": 2, "prompt_id": "P1", "position": "AB", "logical_decision": 2},
            {"sample_id": 2, "prompt_id": "P1", "position": "BA", "logical_decision": 2},
            {"sample_id": 3, "prompt_id": "P1", "position": "AB", "logical_decision": 1},
            {"sample_id": 3, "prompt_id": "P1", "position": "BA", "logical_decision": 2},
            {"sample_id": 4, "prompt_id": "P1", "position": "AB", "logical_decision": 2},
            {"sample_id": 4, "prompt_id": "P1", "position": "BA", "logical_decision": 1},
        ]
    )
    summary = summarize_positional_contingency(build_paired_positional_frame(df_valid))
    assert summary.n_pairs == 4
    assert summary.consistent_1 == 1
    assert summary.consistent_2 == 1
    assert summary.slot_a_bias_b == 1
    assert summary.slot_b_bias_c == 1


def test_notebook_display_configuration_and_multiline_html() -> None:
    from qwen3_eval.visualization import build_multiline_html_table, configure_notebook_display

    prev_colwidth = pd.get_option("display.max_colwidth")
    prev_cols = pd.get_option("display.max_columns")
    try:
        configure_notebook_display()
        assert pd.get_option("display.max_colwidth") is None
        assert pd.get_option("display.max_columns") is None
    finally:
        pd.set_option("display.max_colwidth", prev_colwidth)
        pd.set_option("display.max_columns", prev_cols)

    sample_df = pd.DataFrame(
        [{"Kondisi": "P1_AB", "Chat Template": "<|im_start|>user\nBaris 1\n\n<think>\n\n</think>"}]
    )
    rendered = build_multiline_html_table(sample_df, monospace_columns=("Chat Template",))
    assert "white-space:pre-wrap" in rendered
    assert "monospace" in rendered
    assert "&lt;|im_start|&gt;user\nBaris 1\n\n&lt;think&gt;\n\n&lt;/think&gt;" in rendered


def test_resolve_lazy_export_valid_and_missing_attr() -> None:
    from qwen3_eval.utils.lazy import resolve_lazy_export

    resolved = resolve_lazy_export(
        "qwen3_eval.data",
        "resolve_human_winner",
        {"resolve_human_winner": "qwen3_eval.data.labels"},
    )
    assert callable(resolved)

    with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
        resolve_lazy_export("qwen3_eval.data", "nonexistent", {})
