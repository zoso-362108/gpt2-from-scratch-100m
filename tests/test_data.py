import numpy as np
import pytest
import torch

from src.data import TokenShardLoader


def save_shard(
    path,
    values: list[int],
) -> None:
    tokens = np.array(
        values,
        dtype=np.uint16,
    )
    np.save(path, tokens)


def test_loader_returns_shifted_targets(tmp_path):
    save_shard(
        tmp_path / "train_000000.npy",
        list(range(100)),
    )

    loader = TokenShardLoader(
        data_dir=tmp_path,
        split="train",
        batch_size=2,
        block_size=4,
    )

    x, y = loader.next_batch()

    expected_x = torch.tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ]
    )

    expected_y = torch.tensor(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ]
    )

    assert torch.equal(x, expected_x)
    assert torch.equal(y, expected_y)


def test_loader_advances_position(tmp_path):
    save_shard(
        tmp_path / "train_000000.npy",
        list(range(100)),
    )

    loader = TokenShardLoader(
        data_dir=tmp_path,
        split="train",
        batch_size=2,
        block_size=4,
    )

    loader.next_batch()
    x, _ = loader.next_batch()

    assert x[0, 0].item() == 8


def test_loader_cycles_between_shards(tmp_path):
    save_shard(
        tmp_path / "train_000000.npy",
        list(range(9)),
    )

    save_shard(
        tmp_path / "train_000001.npy",
        list(range(100, 109)),
    )

    loader = TokenShardLoader(
        data_dir=tmp_path,
        split="train",
        batch_size=2,
        block_size=4,
    )

    first_x, _ = loader.next_batch()
    second_x, _ = loader.next_batch()
    third_x, _ = loader.next_batch()

    assert first_x[0, 0].item() == 0
    assert second_x[0, 0].item() == 100
    assert third_x[0, 0].item() == 0


def test_loader_state_can_be_restored(tmp_path):
    save_shard(
        tmp_path / "train_000000.npy",
        list(range(100)),
    )

    first_loader = TokenShardLoader(
        data_dir=tmp_path,
        split="train",
        batch_size=2,
        block_size=4,
    )

    first_loader.next_batch()
    saved_state = first_loader.state_dict()

    second_loader = TokenShardLoader(
        data_dir=tmp_path,
        split="train",
        batch_size=2,
        block_size=4,
    )

    second_loader.load_state_dict(saved_state)

    first_x, first_y = first_loader.next_batch()
    second_x, second_y = second_loader.next_batch()

    assert torch.equal(first_x, second_x)
    assert torch.equal(first_y, second_y)


def test_loader_requires_matching_shards(tmp_path):
    with pytest.raises(FileNotFoundError):
        TokenShardLoader(
            data_dir=tmp_path,
            split="train",
            batch_size=2,
            block_size=4,
        )