import torch

from src.model import CausalSelfAttention, GPTConfig


def test_default_gpt2_config():
    config = GPTConfig()

    assert config.block_size == 1024
    assert config.vocab_size == 50257
    assert config.n_layer == 12
    assert config.n_head == 12
    assert config.n_embd == 768
    assert config.n_embd % config.n_head == 0


def test_attention_output_shape():
    config = GPTConfig(
        block_size=16,
        n_layer=2,
        n_head=4,
        n_embd=32,
    )

    attention = CausalSelfAttention(config)
    x = torch.randn(2, 8, 32)

    output = attention(x)

    assert output.shape == x.shape


def test_attention_is_causal():
    torch.manual_seed(42)

    config = GPTConfig(
        block_size=16,
        n_layer=2,
        n_head=4,
        n_embd=32,
    )

    attention = CausalSelfAttention(config)
    attention.eval()

    original = torch.randn(1, 8, 32)
    modified = original.clone()

    # 只修改未来位置
    modified[:, 4:, :] += 100

    with torch.no_grad():
        original_output = attention(original)
        modified_output = attention(modified)

    # 位置 0～3 不应受到未来 Token 改动的影响
    assert torch.allclose(
        original_output[:, :4, :],
        modified_output[:, :4, :],
        atol=1e-5,
    )
