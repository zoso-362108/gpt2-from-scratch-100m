import argparse
import os
from pathlib import Path

import numpy as np
import tiktoken
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/fineweb_100m",
    )

    parser.add_argument(
        "--train-tokens",
        type=int,
        default=100_000_000,
    )

    parser.add_argument(
        "--val-tokens",
        type=int,
        default=5_000_000,
    )

    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help=(
            "Optional Hugging Face endpoint, for "
            "example https://hf-mirror.com"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def validate_args(args) -> None:
    if args.train_tokens <= 0:
        raise ValueError(
            "train_tokens must be positive"
        )

    if args.val_tokens <= 0:
        raise ValueError(
            "val_tokens must be positive"
        )


def prepare_output_paths(
    output_dir: Path,
    overwrite: bool,
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    val_path = output_dir / "val_000000.npy"
    train_path = output_dir / "train_000000.npy"

    val_partial = output_dir / "val_000000.partial.npy"
    train_partial = (
        output_dir / "train_000000.partial.npy"
    )

    existing = [
        path
        for path in (
            val_path,
            train_path,
            val_partial,
            train_partial,
        )
        if path.exists()
    ]

    if existing and not overwrite:
        formatted = "\n".join(
            str(path) for path in existing
        )

        raise FileExistsError(
            "Output files already exist. "
            "Use --overwrite to replace them:\n"
            f"{formatted}"
        )

    if overwrite:
        for path in existing:
            path.unlink()

    return (
        val_path,
        train_path,
        val_partial,
        train_partial,
    )


def main():
    args = parse_args()
    validate_args(args)

    # 必须在导入 datasets 之前设置。
    if args.endpoint is not None:
        os.environ["HF_ENDPOINT"] = args.endpoint
        print(f"Using endpoint: {args.endpoint}")
    else:
        os.environ.pop("HF_ENDPOINT", None)
        print("Using endpoint: https://huggingface.co")

    from datasets import load_dataset

    output_dir = Path(args.output_dir)

    (
        val_path,
        train_path,
        val_partial,
        train_partial,
    ) = prepare_output_paths(
        output_dir,
        args.overwrite,
    )

    print("Opening FineWeb-Edu stream...")

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    tokenizer = tiktoken.get_encoding("gpt2")
    end_of_text = tokenizer.eot_token

    val_memmap = np.lib.format.open_memmap(
        val_partial,
        mode="w+",
        dtype=np.uint16,
        shape=(args.val_tokens,),
    )

    train_memmap = np.lib.format.open_memmap(
        train_partial,
        mode="w+",
        dtype=np.uint16,
        shape=(args.train_tokens,),
    )

    val_written = 0
    train_written = 0

    total_target = (
        args.val_tokens + args.train_tokens
    )

    progress = tqdm(
        total=total_target,
        unit="token",
        desc="Tokenizing",
    )

    try:
        for document in dataset:
            text = document.get("text")

            if not isinstance(text, str) or not text:
                continue

            encoded = tokenizer.encode_ordinary(text)

            document_tokens = np.asarray(
                [end_of_text, *encoded],
                dtype=np.uint16,
            )

            source_position = 0

            if val_written < args.val_tokens:
                count = min(
                    len(document_tokens),
                    args.val_tokens - val_written,
                )

                val_memmap[
                    val_written : val_written + count
                ] = document_tokens[
                    source_position : source_position + count
                ]

                val_written += count
                source_position += count
                progress.update(count)

            if (
                source_position < len(document_tokens)
                and train_written < args.train_tokens
            ):
                count = min(
                    len(document_tokens) - source_position,
                    args.train_tokens - train_written,
                )

                train_memmap[
                    train_written : train_written + count
                ] = document_tokens[
                    source_position : source_position + count
                ]

                train_written += count
                progress.update(count)

            if (
                val_written >= args.val_tokens
                and train_written >= args.train_tokens
            ):
                break

    finally:
        progress.close()
        val_memmap.flush()
        train_memmap.flush()

    complete = (
        val_written == args.val_tokens
        and train_written == args.train_tokens
    )

    del val_memmap
    del train_memmap

    if not complete:
        raise RuntimeError(
            "Dataset stream ended before enough tokens "
            f"were collected: val={val_written:,}, "
            f"train={train_written:,}"
        )

    val_partial.replace(val_path)
    train_partial.replace(train_path)

    print()
    print("Dataset preparation complete.")
    print(f"Validation tokens: {val_written:,}")
    print(f"Training tokens:   {train_written:,}")
    print(f"Validation file:   {val_path}")
    print(f"Training file:     {train_path}")


if __name__ == "__main__":
    main()