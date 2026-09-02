import argparse
from pathlib import Path

import tiktoken
import torch

from src.model import GPT, GPTConfig


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "checkpoints/gpt2_100m/"
            "step_003052.pt"
        ),
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")
    print(f"Loading: {args.checkpoint}")

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    training_config = checkpoint["config"]

    model_config = GPTConfig(
        **training_config["model"]
    )

    model = GPT(model_config)
    model.load_state_dict(checkpoint["model"])

    trained_step = checkpoint["next_step"]
    validation_loss = checkpoint[
        "validation_loss"
    ]

    del checkpoint

    model.to(device)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    prompts = [
        "Hello, I'm a language model,",
        "The future of artificial intelligence is",
        "In a small town,",
        "The scientist discovered",
    ]

    output_lines = [
        "# GPT-2 100M Token Generation Samples",
        "",
        f"Checkpoint: {args.checkpoint}",
        f"Completed steps: {trained_step}",
        f"Validation loss: {validation_loss:.4f}",
        f"Temperature: {args.temperature}",
        f"Top-K: {args.top_k}",
        f"Seed: {args.seed}",
        "",
    ]

    for index, prompt in enumerate(prompts):
        prompt_tokens = tokenizer.encode(prompt)

        input_ids = torch.tensor(
            prompt_tokens,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        generator = torch.Generator(
            device=device
        )

        # 每个提示词使用固定但不同的种子
        generator.manual_seed(
            args.seed + index
        )

        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            generator=generator,
        )

        generated_text = tokenizer.decode(
            output_ids[0].tolist()
        )

        print()
        print("=" * 60)
        print(generated_text)

        output_lines.extend(
            [
                f"## Sample {index + 1}",
                "",
                f"Prompt: `{prompt}`",
                "",
                generated_text,
                "",
            ]
        )

    output_path = Path(
        "results/generation_samples_100m.md"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        "\n".join(output_lines),
        encoding="utf-8",
    )

    print()
    print(f"Saved samples to: {output_path}")


if __name__ == "__main__":
    main()