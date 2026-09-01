from src.model import GPTConfig


def test_default_gpt2_config():
    config = GPTConfig()

    assert config.block_size == 1024
    assert config.vocab_size == 50257
    assert config.n_layer == 12
    assert config.n_head == 12
    assert config.n_embd == 768
    assert config.n_embd % config.n_head == 0
