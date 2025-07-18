import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Positional Encoding using sine and cosine functions.
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class GPTModel(nn.Module):
    """
    GPT model using PyTorch's built-in transformer components.
    Much faster and more optimized than custom implementation.
    """

    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Final layer norm and projection
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _create_causal_mask(self, seq_len, device):
        """Create causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
        
        # Token embedding and scaling
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Create causal mask
        tgt_mask = self._create_causal_mask(seq_len, input_ids.device)
        
        # Pass through transformer (using self as memory for decoder)
        x = self.transformer(x, x, tgt_mask=tgt_mask)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits

    def generate(self, context, max_length, temperature=1.0):
        """Generate text autoregressively."""
        self.eval()
        device = next(self.parameters()).device
        context = context.to(device)
        
        generated = []
        current_context = context.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits = self.forward(current_context)
                
                # Get next token logits
                next_token_logits = logits[0, -1] / temperature
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                generated.append(next_token.item())
                
                # Update context
                next_token = next_token.unsqueeze(0)
                current_context = torch.cat([current_context, next_token], dim=1)
                
                # Keep context within max length
                if current_context.size(1) > self.max_seq_len:
                    current_context = current_context[:, -self.max_seq_len:]
        
        return torch.tensor(generated)

    def get_loss(self, input_ids, target_ids):
        """Compute cross-entropy loss."""
        logits = self.forward(input_ids)
        
        # Flatten for cross-entropy
        batch_size, seq_len, vocab_size = logits.size()
        logits = logits.view(-1, vocab_size)
        target_ids = target_ids.view(-1)
        
        # Compute loss
        loss = F.cross_entropy(logits, target_ids)
        return loss