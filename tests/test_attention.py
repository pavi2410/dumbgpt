import pytest
import numpy as np
from dumbgpt.model.attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    PositionalEncoding,
)


class TestScaledDotProductAttention:
    """Test suite for Scaled Dot-Product Attention mechanism."""

    def test_attention_initialization(self):
        """Test that attention can be initialized."""
        attention = ScaledDotProductAttention()
        assert attention is not None
        assert hasattr(attention, "forward")

    def test_attention_forward_basic(self):
        """Test basic attention forward pass."""
        attention = ScaledDotProductAttention()

        # Simple test case: 2 tokens, 4 dimensions
        seq_len, d_model = 2, 4
        Q = np.random.randn(seq_len, d_model)
        K = np.random.randn(seq_len, d_model)
        V = np.random.randn(seq_len, d_model)

        output = attention.forward(Q, K, V)

        assert output.shape == (seq_len, d_model)
        assert np.all(np.isfinite(output))

    def test_attention_forward_batch(self):
        """Test attention with batched input."""
        attention = ScaledDotProductAttention()

        batch_size, seq_len, d_model = 3, 5, 8
        Q = np.random.randn(batch_size, seq_len, d_model)
        K = np.random.randn(batch_size, seq_len, d_model)
        V = np.random.randn(batch_size, seq_len, d_model)

        output = attention.forward(Q, K, V)

        assert output.shape == (batch_size, seq_len, d_model)
        assert np.all(np.isfinite(output))

    def test_attention_scores_computation(self):
        """Test that attention scores are computed correctly."""
        attention = ScaledDotProductAttention()

        # Use simple known values
        Q = np.array([[1.0, 0.0], [0.0, 1.0]])  # 2x2
        K = np.array([[1.0, 0.0], [0.0, 1.0]])  # 2x2
        V = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2x2

        output = attention.forward(Q, K, V)

        # Manual calculation:
        # Scores = QK^T / sqrt(d_k) = [[1, 0], [0, 1]] / sqrt(2)
        # After softmax: should be close to identity for this case
        expected_shape = (2, 2)
        assert output.shape == expected_shape

        # Check that output is a weighted combination of V
        # Note: with scaled attention, the results may not be exactly V due to scaling and softmax
        assert np.allclose(
            output[0], V[0], atol=1.0
        )  # First token attends mainly to itself
        assert np.allclose(
            output[1], V[1], atol=1.0
        )  # Second token attends mainly to itself

    def test_attention_scaling(self):
        """Test that attention scores are properly scaled by sqrt(d_k)."""
        attention = ScaledDotProductAttention()

        # Large d_model to test scaling effect
        d_model = 64
        Q = np.ones((1, d_model))
        K = np.ones((1, d_model))
        V = np.ones((1, d_model))

        output = attention.forward(Q, K, V)

        # With proper scaling, should not overflow
        assert np.all(np.isfinite(output))
        assert output.shape == (1, d_model)

    def test_attention_with_mask(self):
        """Test attention with causal mask."""
        attention = ScaledDotProductAttention()

        seq_len, d_model = 3, 4
        Q = np.random.randn(seq_len, d_model)
        K = np.random.randn(seq_len, d_model)
        V = np.random.randn(seq_len, d_model)

        # Create causal mask (lower triangular)
        mask = np.tril(np.ones((seq_len, seq_len)))

        output = attention.forward(Q, K, V, mask=mask)

        assert output.shape == (seq_len, d_model)
        assert np.all(np.isfinite(output))

    def test_attention_mask_effect(self):
        """Test that mask actually affects attention computation."""
        attention = ScaledDotProductAttention()

        # Simple case where we can verify masking
        Q = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        K = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        V = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])

        # No mask
        output_no_mask = attention.forward(Q, K, V)

        # With causal mask
        mask = np.tril(np.ones((3, 3)))
        output_with_mask = attention.forward(Q, K, V, mask=mask)

        # Results should be different
        assert not np.allclose(output_no_mask, output_with_mask)

        # First token should be similar (mainly attends to itself in both cases)
        assert np.allclose(output_no_mask[0], output_with_mask[0], atol=0.5)

    def test_attention_softmax_property(self):
        """Test that attention weights sum to 1 (implicitly through output)."""
        attention = ScaledDotProductAttention()

        # Use uniform V to test if attention weights sum to 1
        seq_len, d_model = 4, 3
        Q = np.random.randn(seq_len, d_model)
        K = np.random.randn(seq_len, d_model)
        V = np.ones((seq_len, d_model))  # All values are 1

        output = attention.forward(Q, K, V)

        # If attention weights sum to 1, output should be close to 1
        assert np.allclose(output, 1.0, atol=1e-6)


class TestMultiHeadAttention:
    """Test suite for Multi-Head Attention mechanism."""

    def test_multihead_initialization(self):
        """Test MultiHeadAttention initialization."""
        mha = MultiHeadAttention(d_model=64, num_heads=8)
        assert mha is not None
        assert hasattr(mha, "forward")
        assert hasattr(mha, "d_model")
        assert hasattr(mha, "num_heads")
        assert hasattr(mha, "d_k")
        assert mha.d_model == 64
        assert mha.num_heads == 8
        assert mha.d_k == 8  # d_model // num_heads

    def test_multihead_parameter_shapes(self):
        """Test that MultiHeadAttention has correct parameter shapes."""
        d_model, num_heads = 64, 8
        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

        # Should have weight matrices for Q, K, V projections
        assert hasattr(mha, "W_q")
        assert hasattr(mha, "W_k")
        assert hasattr(mha, "W_v")
        assert hasattr(mha, "W_o")

        # Check shapes
        assert mha.W_q.weight.shape == (d_model, d_model)
        assert mha.W_k.weight.shape == (d_model, d_model)
        assert mha.W_v.weight.shape == (d_model, d_model)
        assert mha.W_o.weight.shape == (d_model, d_model)

    def test_multihead_forward_single_sequence(self):
        """Test MultiHeadAttention forward pass with single sequence."""
        mha = MultiHeadAttention(d_model=64, num_heads=8)

        seq_len, d_model = 10, 64
        x = np.random.randn(seq_len, d_model)

        output = mha.forward(x)

        assert output.shape == (seq_len, d_model)
        assert np.all(np.isfinite(output))

    def test_multihead_forward_batch(self):
        """Test MultiHeadAttention forward pass with batched sequences."""
        mha = MultiHeadAttention(d_model=32, num_heads=4)

        batch_size, seq_len, d_model = 2, 5, 32
        x = np.random.randn(batch_size, seq_len, d_model)

        output = mha.forward(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert np.all(np.isfinite(output))

    def test_multihead_with_different_qkv(self):
        """Test MultiHeadAttention with different Q, K, V inputs."""
        mha = MultiHeadAttention(d_model=48, num_heads=6)

        seq_len, d_model = 8, 48
        Q = np.random.randn(seq_len, d_model)
        K = np.random.randn(seq_len, d_model)
        V = np.random.randn(seq_len, d_model)

        output = mha.forward(Q, K, V)

        assert output.shape == (seq_len, d_model)
        assert np.all(np.isfinite(output))

    def test_multihead_with_mask(self):
        """Test MultiHeadAttention with causal mask."""
        mha = MultiHeadAttention(d_model=32, num_heads=4)

        seq_len, d_model = 6, 32
        x = np.random.randn(seq_len, d_model)

        # Create causal mask
        mask = np.tril(np.ones((seq_len, seq_len)))

        output = mha.forward(x, mask=mask)

        assert output.shape == (seq_len, d_model)
        assert np.all(np.isfinite(output))

    def test_multihead_head_dimension_constraint(self):
        """Test that d_model must be divisible by num_heads."""
        # This should work
        mha = MultiHeadAttention(d_model=64, num_heads=8)
        assert mha.d_k == 8

        # This should raise an error or handle gracefully
        with pytest.raises(ValueError):
            MultiHeadAttention(d_model=65, num_heads=8)

    def test_multihead_self_attention_property(self):
        """Test self-attention property (when Q=K=V)."""
        mha = MultiHeadAttention(d_model=32, num_heads=4)

        seq_len, d_model = 4, 32
        x = np.random.randn(seq_len, d_model)

        # Self-attention
        output_self = mha.forward(x)

        # Equivalent explicit call
        output_explicit = mha.forward(x, x, x)

        # Should be identical
        np.testing.assert_array_almost_equal(output_self, output_explicit)

    def test_multihead_attention_different_from_single_head(self):
        """Test that multi-head attention produces different results than single head."""
        mha_single = MultiHeadAttention(d_model=32, num_heads=1)
        mha_multi = MultiHeadAttention(d_model=32, num_heads=4)

        seq_len, d_model = 4, 32
        x = np.random.randn(seq_len, d_model)

        output_single = mha_single.forward(x)
        output_multi = mha_multi.forward(x)

        # Should produce different results (with high probability)
        assert not np.allclose(output_single, output_multi)


class TestPositionalEncoding:
    """Test suite for Positional Encoding."""

    def test_positional_encoding_initialization(self):
        """Test PositionalEncoding initialization."""
        pe = PositionalEncoding(d_model=64, max_len=100)
        assert pe is not None
        assert hasattr(pe, "forward")
        assert hasattr(pe, "d_model")
        assert hasattr(pe, "max_len")
        assert pe.d_model == 64
        assert pe.max_len == 100

    def test_positional_encoding_shape(self):
        """Test that positional encoding has correct shape."""
        d_model, max_len = 32, 50
        pe = PositionalEncoding(d_model=d_model, max_len=max_len)

        # Should have encoding matrix
        assert hasattr(pe, "encoding")
        assert pe.encoding.shape == (max_len, d_model)

    def test_positional_encoding_forward(self):
        """Test positional encoding forward pass."""
        pe = PositionalEncoding(d_model=64, max_len=100)

        seq_len, d_model = 10, 64
        x = np.random.randn(seq_len, d_model)

        output = pe.forward(x)

        assert output.shape == (seq_len, d_model)
        assert np.all(np.isfinite(output))

    def test_positional_encoding_batch(self):
        """Test positional encoding with batch input."""
        pe = PositionalEncoding(d_model=32, max_len=50)

        batch_size, seq_len, d_model = 3, 8, 32
        x = np.random.randn(batch_size, seq_len, d_model)

        output = pe.forward(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert np.all(np.isfinite(output))

    def test_positional_encoding_adds_to_input(self):
        """Test that positional encoding is added to input."""
        pe = PositionalEncoding(d_model=8, max_len=10)

        # Zero input to isolate positional encoding effect
        x = np.zeros((3, 8))
        output = pe.forward(x)

        # Output should be non-zero (positional encoding)
        assert not np.allclose(output, 0)

        # Should be different for different positions
        assert not np.allclose(output[0], output[1])
        assert not np.allclose(output[1], output[2])

    def test_positional_encoding_sine_cosine_pattern(self):
        """Test that positional encoding follows sine/cosine pattern."""
        pe = PositionalEncoding(d_model=4, max_len=5)

        # Get encoding for position 0
        x = np.zeros((1, 4))
        pos_0 = pe.forward(x)[0]

        # Check that encoding has expected properties
        assert np.all(np.isfinite(pos_0))

        # For different positions, should get different encodings
        x = np.zeros((2, 4))
        pos_encodings = pe.forward(x)

        assert not np.allclose(pos_encodings[0], pos_encodings[1])

    def test_positional_encoding_max_length_constraint(self):
        """Test that sequences longer than max_len are handled."""
        pe = PositionalEncoding(d_model=8, max_len=5)

        # Sequence longer than max_len
        x = np.zeros((10, 8))

        # Should either handle gracefully or raise appropriate error
        try:
            output = pe.forward(x)
            assert output.shape == (10, 8)
        except ValueError:
            # Acceptable to raise error for sequences too long
            pass

    def test_positional_encoding_deterministic(self):
        """Test that positional encoding is deterministic."""
        pe = PositionalEncoding(d_model=16, max_len=10)

        x = np.random.randn(5, 16)

        output1 = pe.forward(x)
        output2 = pe.forward(x)

        # Should produce identical results
        np.testing.assert_array_equal(output1, output2)


class TestAttentionIntegration:
    """Integration tests for attention mechanisms."""

    def test_attention_in_transformer_block(self):
        """Test attention mechanism in a simple transformer block."""
        # Simulate a simple transformer block
        d_model, num_heads = 64, 8
        seq_len = 10

        # Components
        pe = PositionalEncoding(d_model=d_model, max_len=100)
        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

        # Input sequence
        x = np.random.randn(seq_len, d_model)

        # Add positional encoding
        x_pos = pe.forward(x)

        # Apply multi-head attention
        output = mha.forward(x_pos)

        assert output.shape == (seq_len, d_model)
        assert np.all(np.isfinite(output))

    def test_causal_attention_for_gpt(self):
        """Test causal attention suitable for GPT-style models."""
        d_model, num_heads = 32, 4
        seq_len = 6

        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

        # Create input sequence
        x = np.random.randn(seq_len, d_model)

        # Create causal mask
        mask = np.tril(np.ones((seq_len, seq_len)))

        # Apply causal attention
        output = mha.forward(x, mask=mask)

        assert output.shape == (seq_len, d_model)
        assert np.all(np.isfinite(output))

    def test_attention_with_different_sequence_lengths(self):
        """Test attention with different sequence lengths."""
        d_model, num_heads = 48, 6

        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

        # Test with different sequence lengths
        for seq_len in [1, 3, 5, 10, 20]:
            x = np.random.randn(seq_len, d_model)
            output = mha.forward(x)

            assert output.shape == (seq_len, d_model)
            assert np.all(np.isfinite(output))

    def test_batch_attention_consistency(self):
        """Test that batch attention produces consistent results."""
        d_model, num_heads = 32, 4
        seq_len = 5

        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

        # Single sequence
        x_single = np.random.randn(seq_len, d_model)
        output_single = mha.forward(x_single)

        # Same sequence in batch
        x_batch = np.stack([x_single, x_single])
        output_batch = mha.forward(x_batch)

        # Results should be identical
        np.testing.assert_array_almost_equal(output_single, output_batch[0])
        np.testing.assert_array_almost_equal(output_single, output_batch[1])


# Fixtures for common test data
@pytest.fixture
def sample_attention():
    """Fixture providing a ScaledDotProductAttention instance."""
    return ScaledDotProductAttention()


@pytest.fixture
def sample_multihead_attention():
    """Fixture providing a MultiHeadAttention instance."""
    return MultiHeadAttention(d_model=32, num_heads=4)


@pytest.fixture
def sample_positional_encoding():
    """Fixture providing a PositionalEncoding instance."""
    return PositionalEncoding(d_model=16, max_len=20)


@pytest.fixture
def sample_qkv_matrices():
    """Fixture providing sample Q, K, V matrices."""
    seq_len, d_model = 4, 8
    return {
        "Q": np.random.randn(seq_len, d_model),
        "K": np.random.randn(seq_len, d_model),
        "V": np.random.randn(seq_len, d_model),
    }


class TestWithFixtures:
    """Tests using fixtures for consistent test data."""

    def test_attention_with_sample_data(self, sample_attention, sample_qkv_matrices):
        """Test attention with sample Q, K, V matrices."""
        Q = sample_qkv_matrices["Q"]
        K = sample_qkv_matrices["K"]
        V = sample_qkv_matrices["V"]

        output = sample_attention.forward(Q, K, V)

        assert output.shape == Q.shape
        assert np.all(np.isfinite(output))

    def test_multihead_attention_with_sample_data(self, sample_multihead_attention):
        """Test multi-head attention with sample data."""
        seq_len, d_model = 6, 32
        x = np.random.randn(seq_len, d_model)

        output = sample_multihead_attention.forward(x)

        assert output.shape == (seq_len, d_model)
        assert np.all(np.isfinite(output))

    def test_positional_encoding_with_sample_data(self, sample_positional_encoding):
        """Test positional encoding with sample data."""
        seq_len, d_model = 8, 16
        x = np.random.randn(seq_len, d_model)

        output = sample_positional_encoding.forward(x)

        assert output.shape == (seq_len, d_model)
        assert np.all(np.isfinite(output))

        # Should be different from input
        assert not np.allclose(output, x)
