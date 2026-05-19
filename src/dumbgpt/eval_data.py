"""Local eval holdout using HuggingFace ``datasets`` save_to_disk / load_from_disk."""

from __future__ import annotations

import shutil
from pathlib import Path

import torch
from datasets import Dataset, load_from_disk
from torch.utils.data import Dataset as TorchDataset

from .train import HFStreamingDataset

# Arrow dataset on disk (built once via datasets.Dataset.save_to_disk)
DEFAULT_EVAL_CACHE = Path("data/eval_holdout")


class CachedEvalDataset(TorchDataset):
    """Pre-tokenized windows from a HuggingFace dataset saved on disk."""

    def __init__(self, path: Path):
        self._ds = load_from_disk(str(path)).with_format("torch")

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self._ds[idx]
        return row["input_ids"], row["labels"]


def _cache_dir_ready(path: Path) -> bool:
    return path.is_dir() and (path / "dataset_info.json").exists()


def cache_is_valid(path: Path, seq_len: int, num_samples: int) -> bool:
    if not _cache_dir_ready(path):
        return False
    ds = load_from_disk(str(path))
    if len(ds) < num_samples:
        return False
    return len(ds[0]["input_ids"]) == seq_len


def build_eval_cache(
    tokenizer,
    seq_len: int,
    path: Path,
    num_samples: int,
    seed: int = 777,
) -> Path:
    """Stream training corpora once, then persist with ``Dataset.save_to_disk``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.rmtree(path)

    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    for inp, tgt in HFStreamingDataset(tokenizer, seq_len, steps=num_samples, seed=seed):
        input_ids.append(inp.tolist())
        labels.append(tgt.tolist())

    if len(input_ids) < num_samples:
        raise RuntimeError(
            f"Only collected {len(input_ids)} eval windows (wanted {num_samples}). "
            "Try again or lower --ppl-batches."
        )

    Dataset.from_dict({"input_ids": input_ids, "labels": labels}).save_to_disk(str(path))
    return path


def ensure_eval_cache(
    tokenizer,
    seq_len: int,
    num_samples: int,
    path: Path = DEFAULT_EVAL_CACHE,
    seed: int = 777,
    rebuild: bool = False,
) -> Path:
    if rebuild or not cache_is_valid(path, seq_len, num_samples):
        build_eval_cache(tokenizer, seq_len, path, num_samples, seed=seed)
    return path
