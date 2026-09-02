from pathlib import Path

import tiktoken
import torch

from src.model import GPT, GPTConfig


def get_batch(
    tokens: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(tokens) - block_size - 1

    if max_start <= 0:
        raise ValueError(
            "Training text is shorter than block_size"
        )

    starts = torch.randint(
        0,
        max_start,
        (batch_size,),
    )

    x = torch.stack(
        [
            tokens[start : start + block_size]
            for start in starts
        ]
    )

    y = torch.stack(
        [
            tokens[start + 1 : start + block_size + 1]
            for start in starts
        ]
    )

    return x.to(device), y.to(device)


def generate_sample(
    model: GPT,
    tokenizer,
    prompt: str,
    device: str,
) -> str:
    prompt_tokens = tokenizer.encode(prompt)

    idx = torch.tensor(
        prompt_tokens,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    output = model.generate(
        idx,
        max_new_tokens=30,
        temperature=0,
    )

    return tokenizer.decode(output[0].tolist())


def main():
    torch.manual_seed(42)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    tokenizer = tiktoken.get_encoding("gpt2")

    training_text = (
        "The quick brown fox jumps over the lazy dog. "
        "The model learns to predict the next token.\n"
    ) * 200

    tokens = torch.tensor(
        tokenizer.encode(training_text),
        dtype=torch.long,
    )

    config = GPTConfig(
        block_size=64,
        vocab_size=50257,
        n_layer=4,
        n_head=4,
        n_embd=128,
    )

    model = GPT(config).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Parameters: {parameter_count:,}")
    print(f"Training tokens: {len(tokens):,}")

    prompt = "The quick brown"

    print("\nBefore training:")
    print(
        generate_sample(
            model,
            tokenizer,
            prompt,
            device,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    batch_size = 16
    max_steps = 500

    initial_loss = None
    final_loss = None

    model.train()

    for step in range(max_steps):
        x, y = get_batch(
            tokens=tokens,
            batch_size=batch_size,
            block_size=config.block_size,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)

        _, loss = model(x, y)

        if loss is None:
            raise RuntimeError(
                "Model did not return training loss"
            )

        loss.backward()

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        current_loss = loss.item()

        if initial_loss is None:
            initial_loss = current_loss

        final_loss = current_loss

        if step % 50 == 0 or step == max_steps - 1:
            print(
                f"step {step:4d} | "
                f"loss {current_loss:.4f} | "
                f"grad_norm {gradient_norm.item():.4f}"
            )

    if initial_loss is None or final_loss is None:
        raise RuntimeError("Training did not run")

    print(f"\nInitial loss: {initial_loss:.4f}")
    print(f"Final loss:   {final_loss:.4f}")

    if final_loss >= initial_loss:
        raise RuntimeError(
            "Loss did not decrease during training"
        )

    model.eval()

    print("\nAfter training:")
    print(
        generate_sample(
            model,
            tokenizer,
            prompt,
            device,
        )
    )

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_path = (
        checkpoint_dir / "tiny_overfit.pt"
    )

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "step": max_steps - 1,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
        },
        checkpoint_path,
    )

    print(f"\nSaved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()