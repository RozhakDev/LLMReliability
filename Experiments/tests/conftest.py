import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterator, List

import numpy as np
import pandas as pd
import pytest
import torch

os.environ.setdefault("MPLBACKEND", "Agg")

from llm_exp.config import ExperimentConfig, ExperimentPaths, load_experiment_config
from llm_exp.inference.tasks import build_evaluation_tasks, determine_human_winner


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self) -> None:
        self.last_add_special_tokens = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=None):
        return f"SYS:{messages[0]['content']}|USR:{messages[1]['content']}|THINK:{enable_thinking}"

    def __call__(self, text, return_tensors="pt", add_special_tokens=True):
        self.last_add_special_tokens = add_special_tokens
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "token_type_ids": torch.tensor([[0, 0, 0]]),
        }

    def decode(self, tokens, skip_special_tokens=True):
        return "A"


class FakeMultimodalProcessor:
    def __init__(self, fail_template: bool = False) -> None:
        self.fail_template = fail_template
        self.tokenizer = FakeTokenizer()

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"):
        if self.fail_template:
            raise RuntimeError("Multimodal template failure")
        assert isinstance(messages[0]["content"], list)
        return {
            "input_ids": torch.tensor([[10, 20]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "token_type_ids": torch.tensor([[0, 0]]),
        }


class FakeCausalModel:
    def __init__(self, fail_on_call: int = -1, device_map=None) -> None:
        self.device = torch.device("cpu")
        self.calls = 0
        self.fail_on_call = fail_on_call
        self.hf_device_map = device_map
        self.generation_config = SimpleNamespace(
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            eos_token_id=[151645, 151643],
        )

    def get_input_embeddings(self):
        return SimpleNamespace(weight=torch.zeros(1, device=self.device))

    def generate(self, input_ids, attention_mask=None, max_new_tokens=10, do_sample=False, pad_token_id=0):
        self.calls += 1
        if self.fail_on_call > 0 and self.calls == self.fail_on_call:
            raise RuntimeError("Simulated mid-run crash")
        return torch.cat([input_ids, torch.tensor([[42]], dtype=input_ids.dtype)], dim=1)


@pytest.fixture(scope="session")
def exp_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def exp_config(tmp_path: Path) -> ExperimentConfig:
    return load_experiment_config(
        project_root=tmp_path / "LLMReliabilityEXP",
        models_cache_dir=tmp_path / "models",
    )


@pytest.fixture
def exp_paths(exp_config: ExperimentConfig) -> ExperimentPaths:
    paths = ExperimentPaths.from_config(exp_config)
    paths.ensure_base_dirs()
    return paths


@pytest.fixture
def test_logger() -> Iterator[logging.Logger]:
    logger = logging.getLogger("llm_exp.pytest")
    logger.setLevel(logging.DEBUG)
    logger.propagate = True
    yield logger
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


@pytest.fixture
def make_eval_df() -> Callable[[int], pd.DataFrame]:
    domains: List[str] = ["SHP-AskEngineers", "SHP-AskHistorians", "SHP-AskScience"]

    def _factory(n_samples: int = 6) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "original_id": 100 + idx,
                    "domain": domains[idx % len(domains)],
                    "input": f"Pertanyaan contoh {idx}",
                    "response_1": f"Respons satu {idx}",
                    "response_2": f"Respons dua {idx}",
                    "actual_score": "Assistant A" if idx % 2 == 0 else "Assistant B",
                }
                for idx in range(n_samples)
            ]
        )

    return _factory


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def fake_multimodal_processor() -> Callable[[bool], FakeMultimodalProcessor]:
    return FakeMultimodalProcessor


@pytest.fixture
def fake_model_factory() -> Callable[..., FakeCausalModel]:
    return FakeCausalModel


@pytest.fixture
def write_synthetic_jsonl() -> Callable[..., None]:
    def _writer(out_file: Path, df_eval: pd.DataFrame, seed: int, invalid_every: int = 0, perfect: bool = False) -> None:
        rng = np.random.default_rng(seed)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as handle:
            for idx, task in enumerate(build_evaluation_tasks(df_eval)):
                winner = determine_human_winner(task["row"])
                human = winner if task["position"] == "AB" else ("B" if winner == "A" else "A")
                if invalid_every > 0 and idx % invalid_every == 0:
                    choice = "Tidak yakin"
                elif perfect:
                    choice = human
                else:
                    choice = human if rng.random() < 0.72 else ("B" if human == "A" else "A")
                rec = {
                    "sample_id": task["sample_id"],
                    "domain": str(task["row"]["domain"]),
                    "prompt_id": task["prompt_id"],
                    "position": task["position"],
                    "human_preference": "A",
                    "raw_output": choice,
                    "runtime_sec": 0.1,
                    "output_tokens": 1,
                    "tps": 10.0,
                }
                handle.write(json.dumps(rec) + "\n")

    return _writer
