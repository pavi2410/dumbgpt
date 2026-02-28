import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Modern building blocks (Llama / DeepSeek / Mistral style)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization — faster than LayerNorm, no bias."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute RoPE rotation frequencies. Returns complex tensor (max_seq_len, head_dim/2)."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    return torch.polar(torch.ones(max_seq_len, head_dim // 2), torch.outer(t, freqs))


def apply_rope(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor):
    """
    Apply Rotary Position Embeddings.
    q, k : (B, n_heads, T, head_dim)
    freqs_cis : (T, head_dim/2) complex
    """
    freqs = freqs_cis.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim/2)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        x_c = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
        return torch.view_as_real(x_c * freqs).flatten(-2).type_as(x)

    return rotate(q), rotate(k)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention using PyTorch's fused scaled_dot_product_attention.
    Automatically selects Flash Attention, memory-efficient, or math backend per device.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, freqs_cis)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        return self.proj(out.transpose(1, 2).contiguous().view(B, T, C))


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network. Used in Llama, DeepSeek, PaLM, Mistral.
    FFN(x) = down(SiLU(gate(x)) * up(x))
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff,   d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    """Pre-norm block: RMSNorm → CausalSelfAttention → RMSNorm → SwiGLU."""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn      = CausalSelfAttention(d_model, num_heads, dropout)
        self.ffn_norm  = RMSNorm(d_model)
        self.ffn       = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs_cis)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

class GPTModel(nn.Module):
    """
    Modern decoder-only transformer (Llama/DeepSeek/Mistral style):
      - RoPE rotary position embeddings (generalises to unseen lengths, no learned pos emb)
      - RMSNorm (faster than LayerNorm, no bias)
      - SwiGLU FFN (used in Llama, DeepSeek, PaLM, Mistral)
      - Fused scaled_dot_product_attention (Flash Attention when available)
      - Weight tying (token_emb <-> lm_head)
      - Gradient checkpointing support
    """

    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int,
                 num_layers: int, max_seq_len: int, dropout: float = 0.1,
                 use_checkpointing: bool = False):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.use_checkpointing = use_checkpointing
        self._int8_enabled = False

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.emb_drop  = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm    = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

        freqs = precompute_rope_freqs(d_model // num_heads, max_seq_len)
        self.register_buffer("freqs_cis", freqs, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _enable_int8(self):
        """Apply dynamic INT8 quantization to Linear layers for faster inference."""
        if not self._int8_enabled:
            torch.quantization.quantize_dynamic(self, {nn.Linear}, dtype=torch.qint8, inplace=True)
            self._int8_enabled = True
        return self

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} > max_seq_len {self.max_seq_len}"

        x = self.emb_drop(self.token_emb(input_ids))
        freqs_cis = self.freqs_cis[:T]

        for block in self.blocks:
            if self.use_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, freqs_cis, use_reentrant=False)
            else:
                x = block(x, freqs_cis)

        return self.lm_head(self.norm(x))

    def get_loss(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss over all token positions."""
        logits = self.forward(input_ids)
        return F.cross_entropy(logits.view(-1, self.vocab_size), target_ids.view(-1))

    @torch.no_grad()
    def generate(self, context: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = 50,
                 top_p: float = 0.9, repetition_penalty: float = 1.0) -> torch.Tensor:
        """
        Autoregressive generation with top-k, top-p (nucleus) sampling, and repetition penalty.

        Args:
            context: (1, T) tensor of prompt token ids
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (lower = more deterministic)
            top_k: restrict to top-k logits (0 = disabled)
            top_p: nucleus sampling probability mass threshold
            repetition_penalty: penalty for repeating tokens (1.0 = no penalty)
        Returns:
            (1, T + max_new_tokens) tensor
        """
        self.eval()
        device = next(self.parameters()).device
        ctx = context.to(device)
        generated = []

        for _ in range(max_new_tokens):
            ctx_cond = ctx if ctx.size(1) <= self.max_seq_len else ctx[:, -self.max_seq_len:]
            logits = self.forward(ctx_cond)[:, -1, :] / max(temperature, 1e-6)

            if repetition_penalty != 1.0:
                for token_id in set(generated[-self.max_seq_len:]):
                    logits[0, token_id] /= repetition_penalty

            if top_k > 0:
                top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_vals[:, -1:]] = float('-inf')

            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            generated.append(next_token.item())
            ctx = torch.cat([ctx, next_token], dim=1)

        return ctx