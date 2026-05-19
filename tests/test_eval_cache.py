"""Eval holdout on disk (HuggingFace save_to_disk format, no Hub)."""

from datasets import Dataset

from dumbgpt.eval_data import CachedEvalDataset, cache_is_valid


def test_cache_is_valid_and_dataset(tmp_path):
    path = tmp_path / "eval_holdout"
    seq_len = 32
    Dataset.from_dict(
        {
            "input_ids": [[i] * seq_len for i in range(4)],
            "labels": [[i + 1] * seq_len for i in range(4)],
        }
    ).save_to_disk(str(path))

    assert cache_is_valid(path, seq_len=seq_len, num_samples=4)
    assert not cache_is_valid(path, seq_len=64, num_samples=4)
    assert not cache_is_valid(path, seq_len=seq_len, num_samples=8)

    ds = CachedEvalDataset(path)
    assert len(ds) == 4
    inp, tgt = ds[0]
    assert inp.shape == (seq_len,)
    assert tgt[0].item() == 1
