import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        # 一次性计算 Query、Key、Value
        self.c_attn = nn.Linear(
            config.n_embd,
            3 * config.n_embd,
        )

        # 合并所有注意力头后的输出投影
        self.c_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
        )

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape

        # qkv: (B, T, 3C)
        qkv = self.c_attn(x)

        # 每个张量的形状：(B, T, C)
        q, k, v = qkv.split(self.n_embd, dim=2)

        head_size = channels // self.n_head

        # (B, T, C) -> (B, n_head, T, head_size)
        q = q.view(
            batch_size, seq_len, self.n_head, head_size
        ).transpose(1, 2)

        k = k.view(
            batch_size, seq_len, self.n_head, head_size
        ).transpose(1, 2)

        v = v.view(
            batch_size, seq_len, self.n_head, head_size
        ).transpose(1, 2)

        # is_causal=True：当前位置只能关注自己和之前的位置
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
        )

        # (B, n_head, T, head_size) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(
            batch_size, seq_len, channels
        )

        return self.c_proj(y)