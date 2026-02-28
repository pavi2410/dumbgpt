import torch
import torch.nn as nn
import torch.nn.functional as F


class GPTModel(nn.Module):
    """
    Decoder-only GPT model using TransformerEncoderLayer with a causal mask.
    Uses learned positional embeddings and weight tying between embedding and lm_head.
    """

    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Token + position embeddings (learned)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_drop = nn.Dropout(dropout)

        # Decoder-only transformer: TransformerEncoderLayer + causal mask
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,   # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers, enable_nested_tensor=False)

        # Final layer norm and projection
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share embedding and lm_head weights
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _causal_mask(self, seq_len, device):
        """Upper-triangular boolean mask (True = ignore)."""
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, input_ids):
        B, T = input_ids.size()
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"

        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)
        x = self.emb_drop(self.token_emb(input_ids) + self.pos_emb(positions))

        mask = self._causal_mask(T, input_ids.device)
        x = self.transformer(x, mask=mask, is_causal=True)

        x = self.ln_f(x)
        return self.lm_head(x)  # (B, T, vocab_size)

    def get_loss(self, input_ids, target_ids):
        """Cross-entropy loss over all positions."""
        logits = self.forward(input_ids)
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), target_ids.view(-1))
        return loss

    @torch.no_grad()
    def generate(self, context, max_new_tokens, temperature=1.0, top_k=50):
        """
        Autoregressive generation with top-k sampling.

        Args:
            context: (1, T) tensor of prompt token ids
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (lower = more deterministic)
            top_k: restrict sampling to top-k logits (0 = disabled)
        Returns:
            (1, T + max_new_tokens) tensor
        """
        self.eval()
        device = next(self.parameters()).device
        ctx = context.to(device)

        for _ in range(max_new_tokens):
            ctx_cond = ctx if ctx.size(1) <= self.max_seq_len else ctx[:, -self.max_seq_len:]
            logits = self.forward(ctx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k > 0:
                top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_vals[:, -1:]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            ctx = torch.cat([ctx, next_token], dim=1)

        return ctx