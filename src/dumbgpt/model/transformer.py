import numpy as np
from .layers import Linear, LayerNorm, Embedding, GELU, Softmax
from .attention import MultiHeadAttention, PositionalEncoding


class FeedForward:
    """
    Feed-Forward Network (FFN) component of transformer block.

    Structure: Linear -> GELU -> Linear
    Expands to d_ff dimensions then contracts back to d_model.
    """

    def __init__(self, d_model, d_ff):
        """
        Initialize Feed-Forward Network.

        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension (usually 4 * d_model)
        """
        self.d_model = d_model
        self.d_ff = d_ff

        # Two linear layers with GELU activation
        self.linear1 = Linear(d_model, d_ff)
        self.activation = GELU()
        self.linear2 = Linear(d_ff, d_model)

    def forward(self, x):
        """
        Forward pass of feed-forward network.

        Args:
            x: Input tensor (..., seq_len, d_model)

        Returns:
            Output tensor (..., seq_len, d_model)
        """
        # Expand: d_model -> d_ff
        x = self.linear1.forward(x)

        # Apply activation
        x = self.activation.forward(x)

        # Contract: d_ff -> d_model
        x = self.linear2.forward(x)

        return x


class TransformerBlock:
    """
    Single Transformer Block with self-attention and feed-forward network.

    Structure:
    x -> LayerNorm -> MultiHeadAttention -> Add ->
         LayerNorm -> FeedForward -> Add -> output
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Initialize Transformer Block.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate (not implemented in this version)
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # Self-attention layer
        self.attention = MultiHeadAttention(d_model, num_heads)

        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff)

        # Layer normalization layers
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Forward pass of transformer block.

        Args:
            x: Input tensor (..., seq_len, d_model)
            mask: Optional attention mask (..., seq_len, seq_len)

        Returns:
            Output tensor (..., seq_len, d_model)
        """
        # Pre-norm architecture

        # Self-attention sublayer with residual connection
        norm_x = self.norm1.forward(x)
        attn_output = self.attention.forward(norm_x, mask=mask)
        x = x + attn_output  # Residual connection

        # Feed-forward sublayer with residual connection
        norm_x = self.norm2.forward(x)
        ff_output = self.feed_forward.forward(norm_x)
        x = x + ff_output  # Residual connection

        return x


class GPTModel:
    """
    Complete GPT model with embedding, positional encoding, and transformer blocks.

    Architecture:
    Token Embedding + Positional Encoding ->
    Transformer Block 1 -> ... -> Transformer Block N ->
    Layer Norm -> Linear Head -> Logits
    """

    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len):
        """
        Initialize GPT Model.

        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            num_layers: Number of transformer blocks
            max_seq_len: Maximum sequence length
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Token embedding layer
        self.embedding = Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # Transformer blocks
        self.transformer_blocks = []
        for _ in range(num_layers):
            block = TransformerBlock(d_model, num_heads, d_ff)
            self.transformer_blocks.append(block)

        # Final layer normalization
        self.ln_f = LayerNorm(d_model)

        # Language modeling head
        self.lm_head = Linear(d_model, vocab_size)

        # Softmax for probability computation
        self.softmax = Softmax()

    def forward(self, input_ids):
        """
        Forward pass of GPT model.

        Args:
            input_ids: Input token IDs (..., seq_len)

        Returns:
            Logits (..., seq_len, vocab_size)
        """
        # Handle both single sequence and batch
        if input_ids.ndim == 1:
            seq_len = input_ids.shape[0]
            batch_size = None
        else:
            batch_size, seq_len = input_ids.shape

        # Token embedding
        x = self.embedding.forward(input_ids)

        # Add positional encoding
        x = self.pos_encoding.forward(x)

        # Create causal mask for autoregressive generation
        mask = self._create_causal_mask(seq_len)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block.forward(x, mask=mask)

        # Final layer normalization
        x = self.ln_f.forward(x)

        # Language modeling head
        logits = self.lm_head.forward(x)

        return logits

    def _create_causal_mask(self, seq_len):
        """
        Create causal (lower triangular) mask for autoregressive generation.

        Args:
            seq_len: Sequence length

        Returns:
            Causal mask (seq_len, seq_len)
        """
        mask = np.tril(np.ones((seq_len, seq_len)))
        return mask

    def generate(self, context, max_length, temperature=1.0):
        """
        Generate text autoregressively.

        Args:
            context: Initial context tokens (seq_len,)
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated tokens (max_length,)
        """
        generated = []
        current_context = context.copy()

        for _ in range(max_length):
            # Get logits for current context
            logits = self.forward(current_context)

            # Get logits for next token (last position)
            next_token_logits = logits[-1]

            # Apply temperature scaling
            next_token_logits = next_token_logits / temperature

            # Convert to probabilities
            probs = self.softmax.forward(next_token_logits)

            # Sample next token (using greedy sampling for simplicity)
            next_token = np.argmax(probs)

            # Add to generated sequence
            generated.append(next_token)

            # Update context (keep within max_seq_len)
            current_context = np.append(current_context, next_token)
            if len(current_context) > self.max_seq_len:
                current_context = current_context[-self.max_seq_len :]

        return np.array(generated)

    def get_loss(self, input_ids, target_ids):
        """
        Compute cross-entropy loss for training.

        Args:
            input_ids: Input token IDs (..., seq_len)
            target_ids: Target token IDs (..., seq_len)

        Returns:
            Cross-entropy loss
        """
        # Get logits
        logits = self.forward(input_ids)

        # Flatten for cross-entropy computation
        if logits.ndim == 3:  # Batch case
            batch_size, seq_len, vocab_size = logits.shape
            logits = logits.reshape(-1, vocab_size)
            target_ids = target_ids.reshape(-1)
        else:  # Single sequence case
            seq_len, vocab_size = logits.shape
            target_ids = target_ids.reshape(-1)

        # Compute cross-entropy loss
        # For simplicity, we'll use a basic implementation
        probs = self.softmax.forward(logits)

        # Add small epsilon to prevent log(0)
        epsilon = 1e-10
        probs = np.clip(probs, epsilon, 1.0 - epsilon)

        # Compute negative log likelihood
        loss = 0.0
        for i, target in enumerate(target_ids):
            loss -= np.log(probs[i, target])

        # Average over all tokens
        loss /= len(target_ids)

        return loss
