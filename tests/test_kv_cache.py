"""KV cache must match full-sequence forward for the same context."""

import torch

from dumbgpt.model.transformer import GPTModel, empty_kv_cache


def _tiny_model() -> GPTModel:
    return GPTModel(
        vocab_size=128,
        d_model=64,
        num_heads=4,
        d_ff=128,
        num_layers=2,
        max_seq_len=64,
        dropout=0.0,
    ).eval()


def test_kv_cache_matches_full_forward_last_logits():
    torch.manual_seed(0)
    model = _tiny_model()
    ctx = torch.randint(0, 128, (1, 12))

    full_logits = model.forward(ctx)[:, -1, :]

    caches = empty_kv_cache(len(model.blocks))
    model.forward(ctx[:, :-1], kv_caches=caches, start_pos=0)
    decode_logits = model.forward(ctx[:, -1:], kv_caches=caches, start_pos=ctx.size(1) - 1)

    assert torch.allclose(full_logits, decode_logits[:, -1, :], atol=1e-5, rtol=1e-5)


def test_kv_cache_incremental_matches_full_forward_all_positions():
    torch.manual_seed(2)
    model = _tiny_model()
    ctx = torch.randint(0, 128, (1, 8))
    full = model.forward(ctx)

    caches = empty_kv_cache(len(model.blocks))
    for t in range(ctx.size(1)):
        pos_logits = model.forward(
            ctx[:, t : t + 1],
            kv_caches=caches,
            start_pos=t,
        )
        assert torch.allclose(full[:, t, :], pos_logits[:, 0, :], atol=1e-5, rtol=1e-5)


def test_generate_matches_greedy_full_forward_steps():
    torch.manual_seed(1)
    model = _tiny_model()
    ctx = torch.randint(0, 128, (1, 5))

    # Greedy decode without cache (reference)
    ref = ctx.clone()
    for _ in range(3):
        logits = model.forward(ref)[:, -1, :]
        ref = torch.cat([ref, logits.argmax(dim=-1, keepdim=True)], dim=1)

    # Greedy via generate (temperature -> argmax-like: use near-zero temp + top_k=1)
    torch.manual_seed(1)
    model = _tiny_model()
    ctx = torch.randint(0, 128, (1, 5))
    out = model.generate(ctx, max_new_tokens=3, temperature=1e-6, top_k=1, top_p=1.0)

    assert torch.equal(ref, out)
