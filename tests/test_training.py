import pytest

from src.model import GPT, GPTConfig
from src.training import (
    configure_optimizer,
    get_learning_rate,
    load_training_config,
)


def test_load_debug_config():
    config = load_training_config(
        "configs/debug.yaml"
    )

    assert config["model"]["n_layer"] == 4
    assert (
        config["training"]["max_steps"]
        == 100
    )


def test_learning_rate_warmup():
    learning_rate = get_learning_rate(
        step=0,
        warmup_steps=10,
        max_steps=100,
        max_learning_rate=1e-3,
        min_learning_rate=1e-4,
    )

    assert learning_rate == pytest.approx(
        1e-4
    )


def test_learning_rate_reaches_maximum():
    learning_rate = get_learning_rate(
        step=9,
        warmup_steps=10,
        max_steps=100,
        max_learning_rate=1e-3,
        min_learning_rate=1e-4,
    )

    assert learning_rate == pytest.approx(
        1e-3
    )


def test_learning_rate_decays():
    learning_rate = get_learning_rate(
        step=55,
        warmup_steps=10,
        max_steps=100,
        max_learning_rate=1e-3,
        min_learning_rate=1e-4,
    )

    expected = (
        1e-4
        + 0.5 * (1e-3 - 1e-4)
    )

    assert learning_rate == pytest.approx(
        expected
    )


def test_learning_rate_reaches_minimum():
    learning_rate = get_learning_rate(
        step=100,
        warmup_steps=10,
        max_steps=100,
        max_learning_rate=1e-3,
        min_learning_rate=1e-4,
    )

    assert learning_rate == pytest.approx(
        1e-4
    )


def test_optimizer_covers_each_parameter_once():
    config = GPTConfig(
        block_size=16,
        vocab_size=100,
        n_layer=2,
        n_head=4,
        n_embd=32,
    )

    model = GPT(config)

    optimizer = configure_optimizer(
        model=model,
        learning_rate=1e-3,
        weight_decay=0.1,
        device_type="cpu",
    )

    model_parameter_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }

    optimizer_parameter_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]

    assert (
        len(optimizer_parameter_ids)
        == len(set(optimizer_parameter_ids))
    )

    assert set(optimizer_parameter_ids) == (
        model_parameter_ids
    )


def test_optimizer_weight_decay_groups():
    config = GPTConfig(
        block_size=16,
        vocab_size=100,
        n_layer=2,
        n_head=4,
        n_embd=32,
    )

    model = GPT(config)

    optimizer = configure_optimizer(
        model=model,
        learning_rate=1e-3,
        weight_decay=0.1,
        device_type="cpu",
    )

    weight_decays = {
        group["weight_decay"]
        for group in optimizer.param_groups
    }

    assert weight_decays == {0.0, 0.1}