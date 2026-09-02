import argparse
import csv
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from src.data import TokenShardLoader
from src.model import GPT, GPTConfig
from src.training import (
    configure_optimizer,
    get_learning_rate,
    load_training_config,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/debug.yaml",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
    )

    return parser.parse_args()


def get_device() -> tuple[torch.device, str]:
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"

    return torch.device("cpu"), "cpu"


def get_autocast_context(
    device_type: str,
    precision: str,
):
    if device_type != "cuda":
        return nullcontext()

    if precision == "bfloat16":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )

    if precision == "float16":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )

    return nullcontext()


@torch.no_grad()
def evaluate(
    model: GPT,
    loader: TokenShardLoader,
    device: torch.device,
    device_type: str,
    precision: str,
    batches: int,
) -> float:
    was_training = model.training
    model.eval()
    loader.reset()

    total_loss = 0.0

    for _ in range(batches):
        x, y = loader.next_batch()

        x = x.to(device)
        y = y.to(device)

        with get_autocast_context(
            device_type,
            precision,
        ):
            _, loss = model(x, y)

        if loss is None:
            raise RuntimeError(
                "Model did not return validation loss"
            )

        total_loss += loss.item()

    model.train(was_training)

    return total_loss / batches


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    train_loader: TokenShardLoader,
    config: dict,
    next_step: int,
    validation_loss: float,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_loader": train_loader.state_dict(),
        "config": config,
        "next_step": next_step,
        "validation_loss": validation_loss,
        "cpu_rng_state": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = (
            torch.cuda.get_rng_state()
        )

    temporary_path = path.with_suffix(".tmp")

    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def remove_old_checkpoints(
    checkpoint_dir: Path,
    keep_last: int,
) -> None:
    checkpoints = sorted(
        checkpoint_dir.glob("step_*.pt"),
        key=lambda path: path.stat().st_mtime,
    )

    for checkpoint in checkpoints[:-keep_last]:
        checkpoint.unlink()


def append_log(
    log_path: Path,
    row: dict,
) -> None:
    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_header = not log_path.exists()

    with log_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "step",
                "train_loss",
                "validation_loss",
                "learning_rate",
                "gradient_norm",
                "milliseconds",
                "tokens_per_second",
            ],
        )

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def main():
    args = parse_args()
    config = load_training_config(args.config)

    seed = config["project"]["seed"]
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device, device_type = get_device()

    if device_type == "cuda":
        torch.set_float32_matmul_precision("high")

    print(f"Device: {device}")

    model_config = GPTConfig(
        **config["model"]
    )

    model = GPT(model_config).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Parameters: {parameter_count:,}")

    training = config["training"]
    evaluation = config["evaluation"]
    checkpoint_config = config["checkpoint"]

    batch_size = training["micro_batch_size"]
    block_size = model_config.block_size

    train_loader = TokenShardLoader(
        data_dir=config["data"]["directory"],
        split="train",
        batch_size=batch_size,
        block_size=block_size,
    )

    val_loader = TokenShardLoader(
        data_dir=config["data"]["directory"],
        split="val",
        batch_size=batch_size,
        block_size=block_size,
    )

    micro_batch_tokens = (
        batch_size * block_size
    )

    gradient_accumulation_steps = (
        training["total_batch_tokens"]
        // micro_batch_tokens
    )

    print(
        "Gradient accumulation steps: "
        f"{gradient_accumulation_steps}"
    )

    optimizer = configure_optimizer(
        model=model,
        learning_rate=training[
            "max_learning_rate"
        ],
        weight_decay=training["weight_decay"],
        device_type=device_type,
    )

    if (
        training["compile"]
        and hasattr(torch, "compile")
    ):
        print("Compiling model...")
        model = torch.compile(model)

    start_step = 0

    if args.resume is not None:
        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )
        train_loader.load_state_dict(
            checkpoint["train_loader"]
        )

        start_step = checkpoint["next_step"]

        rng_state = checkpoint.get("rng_state")

        if rng_state is None:
            print(
                "Warning: checkpoint has no RNG state; "
                "resume will work but will not be bitwise reproducible."
            )
        else:
            torch_rng_state = rng_state.get("torch")

            if torch_rng_state is not None:
                if not isinstance(torch_rng_state, torch.Tensor):
                    torch_rng_state = torch.tensor(
                        torch_rng_state,
                        dtype=torch.uint8,
                    )

                torch.set_rng_state(torch_rng_state.cpu())

            cuda_rng_states = rng_state.get("cuda")

            if torch.cuda.is_available() and cuda_rng_states is not None:
                cuda_rng_states = [
                    state if isinstance(state, torch.Tensor)
                    else torch.tensor(state, dtype=torch.uint8)
                    for state in cuda_rng_states
                ]

                torch.cuda.set_rng_state_all(cuda_rng_states)

        print(f"Resuming from step {start_step}")

    checkpoint_dir = Path(
        checkpoint_config["directory"]
    )

    log_path = Path("logs") / (
        f"{config['project']['name']}.csv"
    )

    latest_validation_loss = float("nan")

    for step in range(
        start_step,
        training["max_steps"],
    ):
        step_start = time.perf_counter()

        model.train()
        optimizer.zero_grad(set_to_none=True)

        accumulated_loss = 0.0

        for _ in range(
            gradient_accumulation_steps
        ):
            x, y = train_loader.next_batch()

            x = x.to(device)
            y = y.to(device)

            with get_autocast_context(
                device_type,
                training["precision"],
            ):
                _, loss = model(x, y)

                if loss is None:
                    raise RuntimeError(
                        "Model did not return loss"
                    )

                scaled_loss = (
                    loss
                    / gradient_accumulation_steps
                )

            scaled_loss.backward()
            accumulated_loss += (
                scaled_loss.detach().item()
            )

        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                training["gradient_clip"],
            )
        )

        learning_rate = get_learning_rate(
            step=step,
            warmup_steps=training[
                "warmup_steps"
            ],
            max_steps=training["max_steps"],
            max_learning_rate=training[
                "max_learning_rate"
            ],
            min_learning_rate=training[
                "min_learning_rate"
            ],
        )

        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        optimizer.step()

        if device_type == "cuda":
            torch.cuda.synchronize()

        elapsed = (
            time.perf_counter() - step_start
        )

        tokens_per_second = (
            training["total_batch_tokens"]
            / elapsed
        )

        should_evaluate = (
            step % evaluation["interval"] == 0
            or step == training["max_steps"] - 1
        )

        if should_evaluate:
            latest_validation_loss = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                device_type=device_type,
                precision=training["precision"],
                batches=evaluation["batches"],
            )

        print(
            f"step {step:5d} | "
            f"train {accumulated_loss:.4f} | "
            f"val {latest_validation_loss:.4f} | "
            f"lr {learning_rate:.2e} | "
            f"norm {gradient_norm.item():.4f} | "
            f"{elapsed * 1000:.2f} ms | "
            f"{tokens_per_second:,.0f} tok/s"
        )

        append_log(
            log_path,
            {
                "step": step,
                "train_loss": accumulated_loss,
                "validation_loss": (
                    latest_validation_loss
                    if should_evaluate
                    else ""
                ),
                "learning_rate": learning_rate,
                "gradient_norm": (
                    gradient_norm.item()
                ),
                "milliseconds": elapsed * 1000,
                "tokens_per_second": (
                    tokens_per_second
                ),
            },
        )

        should_checkpoint = (
            (step + 1)
            % checkpoint_config["interval"]
            == 0
            or step == training["max_steps"] - 1
        )

        if should_checkpoint:
            checkpoint_path = checkpoint_dir / (
                f"step_{step + 1:06d}.pt"
            )

            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                config=config,
                next_step=step + 1,
                validation_loss=(
                    latest_validation_loss
                ),
            )

            remove_old_checkpoints(
                checkpoint_dir,
                checkpoint_config["keep_last"],
            )

            print(
                f"Saved checkpoint: "
                f"{checkpoint_path}"
            )


if __name__ == "__main__":
    main()