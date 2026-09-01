from dataclasses import dataclass


@dataclass
class GPTConfig:
    """GPT-2 模型结构配置。"""

    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768