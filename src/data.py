from pathlib import Path

import numpy as np
import torch


class TokenShardLoader:
    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        batch_size: int,
        block_size: int,
    ):
        if split not in {"train", "val"}:
            raise ValueError(
                "split must be 'train' or 'val'"
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive"
            )

        if block_size <= 0:
            raise ValueError(
                "block_size must be positive"
            )

        self.data_dir = Path(data_dir)
        self.split = split
        self.batch_size = batch_size
        self.block_size = block_size

        self.shards = sorted(
            self.data_dir.glob(f"*{split}*.npy")
        )

        if not self.shards:
            raise FileNotFoundError(
                f"No {split} shards found in "
                f"{self.data_dir}"
            )

        self.current_shard = 0
        self.current_position = 0
        self.tokens: np.ndarray

        self._load_current_shard()

    @property
    def batch_tokens(self) -> int:
        return self.batch_size * self.block_size

    def _load_current_shard(self) -> None:
        shard_path = self.shards[self.current_shard]

        self.tokens = np.load(
            shard_path,
            mmap_mode="r",
        )

        minimum_length = self.batch_tokens + 1

        if len(self.tokens) < minimum_length:
            raise ValueError(
                f"Shard {shard_path} contains "
                f"{len(self.tokens)} tokens, but at least "
                f"{minimum_length} are required"
            )

    def reset(self) -> None:
        self.current_shard = 0
        self.current_position = 0
        self._load_current_shard()

    def _advance_shard(self) -> None:
        self.current_shard = (
            self.current_shard + 1
        ) % len(self.shards)

        self.current_position = 0
        self._load_current_shard()

    def next_batch(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        required_tokens = self.batch_tokens + 1

        if (
            self.current_position + required_tokens
            > len(self.tokens)
        ):
            self._advance_shard()

        start = self.current_position
        end = start + required_tokens

        # 转成 int64，并复制出独立可写内存
        buffer = np.array(
            self.tokens[start:end],
            dtype=np.int64,
            copy=True,
        )

        tensor = torch.from_numpy(buffer)

        x = tensor[:-1].view(
            self.batch_size,
            self.block_size,
        )

        y = tensor[1:].view(
            self.batch_size,
            self.block_size,
        )

        self.current_position += self.batch_tokens

        return x, y

    def state_dict(self) -> dict[str, int]:
        return {
            "current_shard": self.current_shard,
            "current_position": self.current_position,
        }

    def load_state_dict(
        self,
        state: dict[str, int],
    ) -> None:
        current_shard = state["current_shard"]
        current_position = state["current_position"]

        if not 0 <= current_shard < len(self.shards):
            raise ValueError(
                "Invalid current_shard in loader state"
            )

        if current_position < 0:
            raise ValueError(
                "Invalid current_position in loader state"
            )

        self.current_shard = current_shard
        self.current_position = current_position
        self._load_current_shard()