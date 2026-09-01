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

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                "n_embd must be divisible by n_head"
            )

        if self.block_size <= 0:
            raise ValueError(
                "block_size must be positive"
            )

        if self.vocab_size <= 0:
            raise ValueError(
                "vocab_size must be positive"
            )

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
        self.c_proj.NANOGPT_SCALE_INIT = True

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

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        # GPT-2 的隐藏层通常扩展为嵌入维度的 4 倍
        self.c_fc = nn.Linear(
            config.n_embd,
            4 * config.n_embd,
        )

        self.gelu = nn.GELU(approximate="tanh")

        self.c_proj = nn.Linear(
            4 * config.n_embd,
            config.n_embd,
        )
        self.c_proj.NANOGPT_SCALE_INIT = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)

        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm + Attention + 残差连接
        x = x + self.attn(self.ln_1(x))

        # Pre-Norm + MLP + 残差连接
        x = x + self.mlp(self.ln_2(x))

        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(
            {
                # Token Embedding
                "wte": nn.Embedding(
                    config.vocab_size,
                    config.n_embd,
                ),

                # Position Embedding
                "wpe": nn.Embedding(
                    config.block_size,
                    config.n_embd,
                ),

                # 多个 Transformer Block
                "h": nn.ModuleList(
                    [
                        Block(config)
                        for _ in range(config.n_layer)
                    ]
                ),

                # 最终 LayerNorm
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )

        # 将隐藏向量映射到整个词表
        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False,
        )

        # GPT-2 使用权重共享：
        # Token Embedding 和输出层共用同一组权重
        self.transformer["wte"].weight = self.lm_head.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02

            # 残差分支会随层数累积，因此缩小输出投影的初始化
            if getattr(
                    module,
                    "NANOGPT_SCALE_INIT",
                    False,
            ):
                std *= (2 * self.config.n_layer) ** -0.5

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=std,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if idx.ndim != 2:
            raise ValueError(
                "idx must have shape (batch_size, sequence_length)"
            )

        batch_size, seq_len = idx.shape

        if seq_len > self.config.block_size:
            raise ValueError(
                f"sequence length {seq_len} exceeds "
                f"block size {self.config.block_size}"
            )

        # 位置编号：0, 1, 2, ..., T-1
        positions = torch.arange(
            0,
            seq_len,
            dtype=torch.long,
            device=idx.device,
        )

        # (B, T, C)
        token_embeddings = self.transformer["wte"](idx)

        # (T, C)，会自动广播到 Batch 维度
        position_embeddings = self.transformer["wpe"](
            positions
        )

        x = token_embeddings + position_embeddings

        for block in self.transformer["h"]:
            x = block(x)

        x = self.transformer["ln_f"](x)

        # (B, T, vocab_size)
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError(
                    "targets must have the same shape as idx"
                )

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss