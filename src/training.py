import inspect
import math
from pathlib import Path
from typing import Any

import torch
import yaml


def load_training_config(
    path: str | Path,
) -> dict[str, Any]:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration must be a YAML mapping"
        )

    required_sections = {
        "project",
        "data",
        "model",
        "training",
        "evaluation",
        "checkpoint",
    }

    missing_sections = (
        required_sections - config.keys()
    )

    if missing_sections:
        missing = ", ".join(
            sorted(missing_sections)
        )
        raise ValueError(
            f"Missing config sections: {missing}"
        )

    validate_training_config(config)

    return config


def validate_training_config(
    config: dict[str, Any],
) -> None:
    model = config["model"]
    training = config["training"]

    block_size = model["block_size"]
    n_head = model["n_head"]
    n_embd = model["n_embd"]

    micro_batch_size = training[
        "micro_batch_size"
    ]
    total_batch_tokens = training[
        "total_batch_tokens"
    ]
    max_steps = training["max_steps"]
    warmup_steps = training["warmup_steps"]

    tokens_per_micro_batch = (
        micro_batch_size * block_size
    )

    if n_embd % n_head != 0:
        raise ValueError(
            "model.n_embd must be divisible "
            "by model.n_head"
        )

    if (
        total_batch_tokens
        % tokens_per_micro_batch
        != 0
    ):
        raise ValueError(
            "training.total_batch_tokens must "
            "be divisible by micro_batch_size "
            "* block_size"
        )

    if not 0 <= warmup_steps < max_steps:
        raise ValueError(
            "warmup_steps must satisfy "
            "0 <= warmup_steps < max_steps"
        )

    max_lr = training["max_learning_rate"]
    min_lr = training["min_learning_rate"]

    if not 0 < min_lr <= max_lr:
        raise ValueError(
            "Learning rates must satisfy "
            "0 < min_lr <= max_lr"
        )


def get_learning_rate(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_learning_rate: float,
    min_learning_rate: float,
) -> float:
    if step < 0:
        raise ValueError(
            "step must be non-negative"
        )

    if not 0 <= warmup_steps < max_steps:
        raise ValueError(
            "warmup_steps must satisfy "
            "0 <= warmup_steps < max_steps"
        )

    if step < warmup_steps:
        return max_learning_rate * (
            step + 1
        ) / warmup_steps

    if step >= max_steps:
        return min_learning_rate

    decay_ratio = (
        step - warmup_steps
    ) / (
        max_steps - warmup_steps
    )

    coefficient = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )

    return (
        min_learning_rate
        + coefficient
        * (
            max_learning_rate
            - min_learning_rate
        )
    )


def configure_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    device_type: str,
) -> torch.optim.AdamW:
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    # 矩阵参数进行 weight decay；
    # bias 和 LayerNorm 等一维参数不进行 decay。
    decay_parameters = [
        parameter
        for parameter in parameters.values()
        if parameter.dim() >= 2
    ]

    no_decay_parameters = [
        parameter
        for parameter in parameters.values()
        if parameter.dim() < 2
    ]

    optimizer_groups = [
        {
            "params": decay_parameters,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_parameters,
            "weight_decay": 0.0,
        },
    ]

    adamw_parameters = inspect.signature(
        torch.optim.AdamW
    ).parameters

    fused_available = (
        "fused" in adamw_parameters
    )

    use_fused = (
        device_type == "cuda"
        and fused_available
    )

    optimizer_kwargs = {
        "lr": learning_rate,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
    }

    if fused_available:
        optimizer_kwargs["fused"] = use_fused

    return torch.optim.AdamW(
        optimizer_groups,
        **optimizer_kwargs,
    )