import numpy as np
from .layers import Linear, Softmax


class ScaledDotProductAttention:
    """
    Scaled Dot-Product Attention mechanism.

    Formula: Attention(Q, K, V) = softmax(QK^T / √d_k) * V
    """

    def __init__(self):
        self.softmax = Softmax()

    def forward(self, Q, K, V, mask=None):
        """
        Forward pass of scaled dot-product attention.

        Args:
            Q: Query matrix (..., seq_len, d_k)
            K: Key matrix (..., seq_len, d_k)
            V: Value matrix (..., seq_len, d_v)
            mask: Optional mask matrix (..., seq_len, seq_len)

        Returns:
            Output matrix (..., seq_len, d_v)
        """
        # Get the dimension of keys for scaling
        d_k = Q.shape[-1]

        # Compute attention scores: QK^T / √d_k
        scores = np.matmul(Q, np.swapaxes(K, -2, -1)) / np.sqrt(d_k)

        # Apply mask if provided
        if mask is not None:
            # Convert mask to boolean and apply large negative value to masked positions
            mask = mask.astype(bool)
            scores = np.where(mask, scores, -1e9)

        # Apply softmax to get attention weights
        attention_weights = self.softmax.forward(scores)

        # Apply attention weights to values
        output = np.matmul(attention_weights, V)

        return output


class MultiHeadAttention:
    """
    Multi-Head Attention mechanism.

    Runs multiple attention heads in parallel and concatenates results.
    """

    def __init__(self, d_model, num_heads):
        """
        Initialize Multi-Head Attention.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
        """
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        # Linear projections for Q, K, V
        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)

        # Output projection
        self.W_o = Linear(d_model, d_model)

        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention()

    def forward(self, query, key=None, value=None, mask=None):
        """
        Forward pass of multi-head attention.

        Args:
            query: Query input (..., seq_len, d_model)
            key: Key input (..., seq_len, d_model). If None, uses query (self-attention)
            value: Value input (..., seq_len, d_model). If None, uses query (self-attention)
            mask: Optional mask (..., seq_len, seq_len)

        Returns:
            Output (..., seq_len, d_model)
        """
        # Default to self-attention if key/value not provided
        if key is None:
            key = query
        if value is None:
            value = query

        # Get batch dimensions
        batch_dims = query.shape[:-2]
        seq_len = query.shape[-2]

        # Linear projections
        Q = self.W_q.forward(query)  # (..., seq_len, d_model)
        K = self.W_k.forward(key)  # (..., seq_len, d_model)
        V = self.W_v.forward(value)  # (..., seq_len, d_model)

        # Reshape for multi-head attention
        # (..., seq_len, d_model) -> (..., seq_len, num_heads, d_k)
        Q = Q.reshape(*batch_dims, seq_len, self.num_heads, self.d_k)
        K = K.reshape(*batch_dims, seq_len, self.num_heads, self.d_k)
        V = V.reshape(*batch_dims, seq_len, self.num_heads, self.d_k)

        # Transpose to (..., num_heads, seq_len, d_k)
        # From (..., seq_len, num_heads, d_k) to (..., num_heads, seq_len, d_k)
        # axes = list(range(len(batch_dims))) + [-2, -3, -1]
        Q = np.moveaxis(Q, -2, -3)
        K = np.moveaxis(K, -2, -3)
        V = np.moveaxis(V, -2, -3)

        # Apply attention to each head
        attention_output = self.attention.forward(Q, K, V, mask)

        # Transpose back and reshape
        # (..., num_heads, seq_len, d_k) -> (..., seq_len, num_heads, d_k)
        attention_output = np.moveaxis(attention_output, -2, -3)

        # Concatenate heads: (..., seq_len, num_heads, d_k) -> (..., seq_len, d_model)
        concat_output = attention_output.reshape(*batch_dims, seq_len, self.d_model)

        # Final linear projection
        output = self.W_o.forward(concat_output)

        return output


class PositionalEncoding:
    """
    Positional Encoding using sine and cosine functions.

    Adds positional information to token embeddings.
    """

    def __init__(self, d_model, max_len=5000):
        """
        Initialize Positional Encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        self.d_model = d_model
        self.max_len = max_len

        # Create positional encoding matrix
        self.encoding = self._create_positional_encoding()

    def _create_positional_encoding(self):
        """
        Create the positional encoding matrix.

        Returns:
            Encoding matrix (max_len, d_model)
        """
        encoding = np.zeros((self.max_len, self.d_model))

        # Create position indices
        position = np.arange(0, self.max_len).reshape(-1, 1)

        # Create dimension indices
        div_term = np.exp(
            np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model)
        )

        # Apply sine to even indices
        encoding[:, 0::2] = np.sin(position * div_term)

        # Apply cosine to odd indices
        if self.d_model % 2 == 0:
            encoding[:, 1::2] = np.cos(position * div_term)
        else:
            encoding[:, 1::2] = np.cos(position * div_term[:-1])

        return encoding

    def forward(self, x):
        """
        Add positional encoding to input.

        Args:
            x: Input tensor (..., seq_len, d_model)

        Returns:
            Input with positional encoding added (..., seq_len, d_model)
        """
        seq_len = x.shape[-2]

        # Check if sequence is too long
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds maximum length ({self.max_len})"
            )

        # Get positional encoding for this sequence length
        pos_encoding = self.encoding[:seq_len, :]

        # Add positional encoding to input
        # Broadcasting will handle batch dimensions automatically
        return x + pos_encoding
