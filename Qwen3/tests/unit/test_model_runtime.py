import logging
from types import SimpleNamespace

import pytest
import torch

from qwen3_eval.models.runtime_utils import (
    configure_deterministic_generation,
    resolve_input_device,
    resolve_pad_token_id,
    warn_if_offloaded,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("tokenizer", "expected"),
    [
        (SimpleNamespace(pad_token_id=7, eos_token_id=2), 7),
        (SimpleNamespace(pad_token_id=None, eos_token_id=2), 2),
        (SimpleNamespace(pad_token_id=None, eos_token_id=[15, 16]), 15),
        (SimpleNamespace(pad_token_id=None, eos_token_id=[]), 0),
        (SimpleNamespace(pad_token_id=None, eos_token_id=None), 0),
    ],
)
def test_resolve_pad_token_id(tokenizer: SimpleNamespace, expected: int) -> None:
    assert resolve_pad_token_id(tokenizer) == expected


def test_configure_deterministic_generation_preserves_eos(
    fake_model_factory,
    test_logger: logging.Logger,
) -> None:
    model = fake_model_factory()
    configure_deterministic_generation(model, logger=test_logger)
    assert model.generation_config.do_sample is False
    assert model.generation_config.temperature is None
    assert model.generation_config.top_p is None
    assert model.generation_config.top_k is None
    assert model.generation_config.min_p is None
    assert model.generation_config.eos_token_id == [151645, 151643]
    assert resolve_input_device(model) == torch.device("cpu")


def test_warn_if_offloaded_logs_cpu_or_disk_modules(
    fake_model_factory,
    test_logger: logging.Logger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = fake_model_factory(device_map={"layer0": 0, "layer1": "cpu", "layer2": "disk"})
    with caplog.at_level(logging.WARNING, logger=test_logger.name):
        warn_if_offloaded(model, "qwen3_32b", test_logger)
    assert "model_offloaded | model=qwen3_32b | modules=2" in caplog.text
