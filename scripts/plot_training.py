import argparse
import csv
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log",
        type=str,
        default=(
            "logs/"
            "gpt2-from-scratch-100m.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "results/"
            "training_curves_100m.png"
        ),
    )

    parser.add_argument(
        "--window",
        type=int,
        default=50,
    )

    return parser.parse_args()


def moving_average(
    values: list[float],
    window: int,
) -> list[float]:
    result = []
    current = deque()
    current_sum = 0.0

    for value in values:
        current.append(value)
        current_sum += value

        if len(current) > window:
            current_sum -= current.popleft()

        result.append(
            current_sum / len(current)
        )

    return result


def main():
    args = parse_args()

    steps = []
    train_losses = []
    val_steps = []
    val_losses = []
    throughputs = []
    durations = []

    with Path(args.log).open(
        "r",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            step = int(row["step"])
            train_loss = float(
                row["train_loss"]
            )

            steps.append(step)
            train_losses.append(train_loss)

            throughputs.append(
                float(row["tokens_per_second"])
            )

            durations.append(
                float(row["milliseconds"])
            )

            validation_loss = row[
                "validation_loss"
            ].strip()

            if validation_loss:
                val_steps.append(step)
                val_losses.append(
                    float(validation_loss)
                )

    smoothed_train_loss = moving_average(
        train_losses,
        args.window,
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        constrained_layout=True,
    )

    loss_axis = axes[0]

    loss_axis.plot(
        steps,
        train_losses,
        color="steelblue",
        alpha=0.18,
        linewidth=0.7,
        label="Training loss (raw)",
    )

    loss_axis.plot(
        steps,
        smoothed_train_loss,
        color="navy",
        linewidth=2,
        label=(
            f"Training loss "
            f"({args.window}-step average)"
        ),
    )

    loss_axis.plot(
        val_steps,
        val_losses,
        color="darkorange",
        marker="o",
        linewidth=2,
        label="Validation loss",
    )

    loss_axis.set_title(
        "GPT-2 124M — FineWeb-Edu 100M Tokens"
    )
    loss_axis.set_xlabel("Optimizer step")
    loss_axis.set_ylabel("Cross-entropy loss")
    loss_axis.grid(alpha=0.25)
    loss_axis.legend()

    throughput_axis = axes[1]

    throughput_axis.plot(
        steps,
        throughputs,
        color="seagreen",
        linewidth=1,
    )

    throughput_axis.set_title(
        "Training throughput"
    )
    throughput_axis.set_xlabel("Optimizer step")
    throughput_axis.set_ylabel("Tokens per second")
    throughput_axis.grid(alpha=0.25)

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=180,
    )

    total_minutes = sum(durations) / 1000 / 60
    recent_throughput = (
        sum(throughputs[-100:])
        / len(throughputs[-100:])
    )

    print(f"Steps: {len(steps):,}")
    print(
        f"Initial training loss: "
        f"{train_losses[0]:.4f}"
    )
    print(
        f"Final training loss: "
        f"{train_losses[-1]:.4f}"
    )
    print(
        f"Final validation loss: "
        f"{val_losses[-1]:.4f}"
    )
    print(
        f"Best validation loss: "
        f"{min(val_losses):.4f}"
    )
    print(
        f"Logged training time: "
        f"{total_minutes:.1f} minutes"
    )
    print(
        f"Recent throughput: "
        f"{recent_throughput:,.0f} tok/s"
    )
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
