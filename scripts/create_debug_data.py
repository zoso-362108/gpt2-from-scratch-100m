from pathlib import Path

import numpy as np
import tiktoken


def main():
    tokenizer = tiktoken.get_encoding("gpt2")

    text = (
        "The quick brown fox jumps over the lazy dog. "
        "A language model predicts the next token.\n"
    ) * 2000

    tokens = np.array(
        tokenizer.encode(text),
        dtype=np.uint16,
    )

    split_index = int(len(tokens) * 0.9)

    train_tokens = tokens[:split_index]
    val_tokens = tokens[split_index:]

    output_dir = Path("data/debug")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_dir / "train_000000.npy",
        train_tokens,
    )

    np.save(
        output_dir / "val_000000.npy",
        val_tokens,
    )

    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Validation tokens: {len(val_tokens):,}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()