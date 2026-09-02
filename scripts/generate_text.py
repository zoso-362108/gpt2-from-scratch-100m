import torch
import tiktoken

from src.model import GPT


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = GPT.from_pretrained(
        "gpt2",
        pretrained_source=r"models\gpt2",
    )

    model.to(device)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    prompt = "Hello, I'm a language model,"
    prompt_tokens = tokenizer.encode(prompt)

    idx = torch.tensor(
        prompt_tokens,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    generator = torch.Generator(device=device)
    generator.manual_seed(42)

    output = model.generate(
        idx,
        max_new_tokens=50,
        temperature=0.8,
        top_k=50,
        generator=generator,
    )

    generated_text = tokenizer.decode(
        output[0].tolist()
    )

    print(f"Device: {device}")
    print()
    print(generated_text)


if __name__ == "__main__":
    main()