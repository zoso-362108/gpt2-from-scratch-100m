import torch
from transformers import GPT2LMHeadModel

from src.model import GPT


def main():
    torch.manual_seed(42)

    local_model_path = r"models\gpt2"

    print("Loading local implementation...")
    local_model = GPT.from_pretrained(
        "gpt2",
        pretrained_source=local_model_path,
    )
    local_model.eval()

    print("Loading Hugging Face implementation...")
    hf_model = GPT2LMHeadModel.from_pretrained(
        local_model_path,
        local_files_only=True,
    )
    hf_model.eval()

    # "Hello, I'm a language model,"
    tokens = torch.tensor(
        [[15496, 11, 314, 1101, 257, 3303, 2746, 11]],
        dtype=torch.long,
    )

    with torch.no_grad():
        local_logits, _ = local_model(tokens)
        hf_logits = hf_model(tokens).logits

    max_difference = (
        local_logits - hf_logits
    ).abs().max().item()

    mean_difference = (
        local_logits - hf_logits
    ).abs().mean().item()

    print(f"Local shape: {local_logits.shape}")
    print(f"HF shape:    {hf_logits.shape}")
    print(f"Maximum difference: {max_difference:.8f}")
    print(f"Mean difference:    {mean_difference:.8f}")

    if torch.allclose(
        local_logits,
        hf_logits,
        atol=1e-3,
        rtol=1e-3,
    ):
        print("PASS: Model outputs match.")
    else:
        raise RuntimeError(
            "FAIL: Model outputs do not match."
        )


if __name__ == "__main__":
    main()