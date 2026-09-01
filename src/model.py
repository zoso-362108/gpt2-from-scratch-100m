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

    @classmethod
    def from_pretrained(
            cls,
            model_type: str = "gpt2",
            pretrained_source: str | None = None,
    ) -> "GPT":
        from transformers import GPT2LMHeadModel

        supported_models = {
            "gpt2": {
                "n_layer": 12,
                "n_head": 12,
                "n_embd": 768,
            },
            "gpt2-medium": {
                "n_layer": 24,
                "n_head": 16,
                "n_embd": 1024,
            },
            "gpt2-large": {
                "n_layer": 36,
                "n_head": 20,
                "n_embd": 1280,
            },
            "gpt2-xl": {
                "n_layer": 48,
                "n_head": 25,
                "n_embd": 1600,
            },
        }

        if model_type not in supported_models:
            raise ValueError(
                f"Unsupported model type: {model_type}"
            )

        config_args = supported_models[model_type]

        config = GPTConfig(
            block_size=1024,
            vocab_size=50257,
            **config_args,
        )

        model = cls(config)

        model_id = (
            "openai-community/gpt2"
            if model_type == "gpt2"
            else model_type
        )

        source = (
            pretrained_source
            if pretrained_source is not None
            else model_id
        )

        local_files_only = pretrained_source is not None

        print(f"Loading pretrained weights from {source}")

        hf_model = GPT2LMHeadModel.from_pretrained(
            source,
            local_files_only=local_files_only,
        )

        own_state = model.state_dict()
        hf_state = hf_model.state_dict()

        # Hugging Face 中的因果掩码缓冲区不是可训练参数，
        # 我们使用 scaled_dot_product_attention，因此无需复制。
        ignored_suffixes = (
            ".attn.masked_bias",
            ".attn.bias",
        )

        hf_keys = [
            key
            for key in hf_state
            if not key.endswith(ignored_suffixes)
        ]

        own_keys = list(own_state.keys())

        if len(hf_keys) != len(own_keys):
            raise RuntimeError(
                "State dictionary length mismatch: "
                f"ours={len(own_keys)}, "
                f"huggingface={len(hf_keys)}"
            )

        # Hugging Face Conv1D 权重的存储方向与
        # torch.nn.Linear 相反，需要转置。
        transposed_suffixes = (
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        )

        with torch.no_grad():
            for key in hf_keys:
                if key not in own_state:
                    raise KeyError(
                        f"Missing key in local model: {key}"
                    )

                source = hf_state[key]
                destination = own_state[key]

                if key.endswith(transposed_suffixes):
                    source = source.t()

                if source.shape != destination.shape:
                    raise RuntimeError(
                        f"Shape mismatch for {key}: "
                        f"ours={tuple(destination.shape)}, "
                        f"huggingface={tuple(source.shape)}"
                    )

                destination.copy_(source)

        model.eval()
        return model

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