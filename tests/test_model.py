import torch

from src.model import (
    Block,
    CausalSelfAttention,
    GPT,
    GPTConfig,
    MLP,
)


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

def test_mlp_output_shape():
    config = GPTConfig(
        block_size=16,
        n_layer=2,
        n_head=4,
        n_embd=32,
    )

    mlp = MLP(config)
    x = torch.randn(2, 8, 32)

    output = mlp(x)

    assert output.shape == x.shape
    assert mlp.c_fc.out_features == 4 * config.n_embd
    assert mlp.c_proj.out_features == config.n_embd


def test_block_output_shape():
    config = GPTConfig(
        block_size=16,
        n_layer=2,
        n_head=4,
        n_embd=32,
    )

    block = Block(config)
    x = torch.randn(2, 8, 32)

    output = block(x)

    assert output.shape == x.shape


def test_block_backward():
    config = GPTConfig(
        block_size=16,
        n_layer=2,
        n_head=4,
        n_embd=32,
    )

    block = Block(config)
    x = torch.randn(2, 8, 32, requires_grad=True)

    output = block(x)
    loss = output.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for parameter in block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

def create_tiny_config():
    return GPTConfig(
        block_size=16,
        vocab_size=100,
        n_layer=2,
        n_head=4,
        n_embd=32,
    )


def test_gpt_output_shape():
    config = create_tiny_config()
    model = GPT(config)

    idx = torch.randint(
        0,
        config.vocab_size,
        (2, 8),
    )

    logits, loss = model(idx)

    assert logits.shape == (
        2,
        8,
        config.vocab_size,
    )
    assert loss is None


def test_gpt_training_loss_and_backward():
    config = create_tiny_config()
    model = GPT(config)

    idx = torch.randint(
        0,
        config.vocab_size,
        (2, 8),
    )

    targets = torch.randint(
        0,
        config.vocab_size,
        (2, 8),
    )

    logits, loss = model(idx, targets)

    assert logits.shape == (
        2,
        8,
        config.vocab_size,
    )
    assert loss is not None
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_embedding_and_lm_head_share_weights():
    config = create_tiny_config()
    model = GPT(config)

    embedding_weight = model.transformer["wte"].weight
    output_weight = model.lm_head.weight

    assert embedding_weight is output_weight
    assert (
        embedding_weight.data_ptr()
        == output_weight.data_ptr()
    )


def test_sequence_length_limit():
    config = create_tiny_config()
    model = GPT(config)

    idx = torch.randint(
        0,
        config.vocab_size,
        (1, config.block_size + 1),
    )

    try:
        model(idx)
    except ValueError as error:
        assert "exceeds block size" in str(error)
    else:
        raise AssertionError(
            "Expected ValueError for excessive sequence length"
        )

def test_linear_biases_are_initialized_to_zero():
    config = create_tiny_config()
    model = GPT(config)

    for module in model.modules():
        if (
            isinstance(module, torch.nn.Linear)
            and module.bias is not None
        ):
            assert torch.count_nonzero(module.bias) == 0


def test_initialization_is_reproducible():
    config = create_tiny_config()

    torch.manual_seed(123)
    first_model = GPT(config)

    torch.manual_seed(123)
    second_model = GPT(config)

    first_parameters = dict(
        first_model.named_parameters()
    )
    second_parameters = dict(
        second_model.named_parameters()
    )

    assert first_parameters.keys() == second_parameters.keys()

    for name in first_parameters:
        assert torch.equal(
            first_parameters[name],
            second_parameters[name],
        )


def test_residual_projection_uses_scaled_init():
    config = create_tiny_config()

    torch.manual_seed(42)
    model = GPT(config)

    expected_std = (
        0.02 * (2 * config.n_layer) ** -0.5
    )

    projection = model.transformer["h"][0].attn.c_proj
    actual_std = projection.weight.std().item()

    assert abs(actual_std - expected_std) < 0.003