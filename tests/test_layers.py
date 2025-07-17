import pytest
import numpy as np
from dumbgpt.model.layers import Linear, ReLU, GELU, Softmax, LayerNorm, Embedding


class TestLinear:
    """Test suite for Linear layer implementation."""
    
    def test_linear_initialization(self):
        """Test that Linear layer can be initialized."""
        layer = Linear(in_features=10, out_features=5)
        assert layer is not None
        assert hasattr(layer, 'weight')
        assert hasattr(layer, 'bias')
        assert hasattr(layer, 'in_features')
        assert hasattr(layer, 'out_features')
    
    def test_linear_weight_bias_shapes(self):
        """Test that weight and bias have correct shapes."""
        in_features, out_features = 8, 4
        layer = Linear(in_features, out_features)
        
        # Weight should be (in_features, out_features)
        assert layer.weight.shape == (in_features, out_features)
        # Bias should be (out_features,)
        assert layer.bias.shape == (out_features,)
    
    def test_linear_forward_single_input(self):
        """Test forward pass with single input vector."""
        layer = Linear(in_features=3, out_features=2)
        x = np.array([1.0, 2.0, 3.0])
        
        output = layer.forward(x)
        
        assert output.shape == (2,)
        assert isinstance(output, np.ndarray)
        
        # Manual calculation: y = xW + b
        expected = np.dot(x, layer.weight) + layer.bias
        np.testing.assert_array_almost_equal(output, expected)
    
    def test_linear_forward_batch_input(self):
        """Test forward pass with batch of inputs."""
        layer = Linear(in_features=4, out_features=3)
        batch_size = 5
        x = np.random.randn(batch_size, 4)
        
        output = layer.forward(x)
        
        assert output.shape == (batch_size, 3)
        
        # Manual calculation for batch
        expected = np.dot(x, layer.weight) + layer.bias
        np.testing.assert_array_almost_equal(output, expected)
    
    def test_linear_no_bias(self):
        """Test Linear layer without bias."""
        layer = Linear(in_features=3, out_features=2, bias=False)
        x = np.array([1.0, 2.0, 3.0])
        
        assert layer.bias is None
        output = layer.forward(x)
        
        expected = np.dot(x, layer.weight)
        np.testing.assert_array_almost_equal(output, expected)
    
    def test_linear_weight_initialization(self):
        """Test that weights are initialized reasonably."""
        layer = Linear(in_features=100, out_features=50)
        
        # Weights should be small random values
        assert np.all(np.abs(layer.weight) < 1.0)
        # Should not be all zeros
        assert not np.allclose(layer.weight, 0)
        # Should have reasonable variance
        assert 0.01 < np.var(layer.weight) < 0.1


class TestActivationFunctions:
    """Test suite for activation functions."""
    
    def test_relu_initialization(self):
        """Test ReLU activation initialization."""
        relu = ReLU()
        assert relu is not None
        assert hasattr(relu, 'forward')
    
    def test_relu_forward(self):
        """Test ReLU forward pass."""
        relu = ReLU()
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        
        output = relu.forward(x)
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        
        np.testing.assert_array_almost_equal(output, expected)
    
    def test_relu_batch(self):
        """Test ReLU with batch input."""
        relu = ReLU()
        x = np.random.randn(3, 4)
        
        output = relu.forward(x)
        
        assert output.shape == x.shape
        # All negative values should be zero
        assert np.all(output >= 0)
        # Positive values should be unchanged
        positive_mask = x > 0
        np.testing.assert_array_almost_equal(output[positive_mask], x[positive_mask])
    
    def test_gelu_initialization(self):
        """Test GELU activation initialization."""
        gelu = GELU()
        assert gelu is not None
        assert hasattr(gelu, 'forward')
    
    def test_gelu_forward(self):
        """Test GELU forward pass."""
        gelu = GELU()
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        
        output = gelu.forward(x)
        
        assert output.shape == x.shape
        # GELU should be smooth (unlike ReLU)
        # GELU(0) should be exactly 0
        assert output[2] == 0.0  # x[2] = 0.0
        # Non-zero inputs should not produce exact zeros
        non_zero_mask = x != 0
        assert not np.any(output[non_zero_mask] == 0)
        # Should be approximately x * sigmoid(1.702 * x)
        expected_approx = x * (1 / (1 + np.exp(-1.702 * x)))
        # Allow some tolerance for approximation
        assert np.allclose(output, expected_approx, atol=0.1)
    
    def test_gelu_properties(self):
        """Test GELU mathematical properties."""
        gelu = GELU()
        
        # GELU(0) should be 0
        assert np.isclose(gelu.forward(np.array([0.0]))[0], 0.0, atol=1e-6)
        
        # GELU should be smooth and continuous
        x = np.linspace(-2, 2, 50)
        y = gelu.forward(x)
        assert np.all(np.isfinite(y))  # Should not have NaN or inf values
    
    def test_softmax_initialization(self):
        """Test Softmax activation initialization."""
        softmax = Softmax()
        assert softmax is not None
        assert hasattr(softmax, 'forward')
    
    def test_softmax_forward(self):
        """Test Softmax forward pass."""
        softmax = Softmax()
        x = np.array([1.0, 2.0, 3.0])
        
        output = softmax.forward(x)
        
        assert output.shape == x.shape
        # Should sum to 1
        assert np.isclose(np.sum(output), 1.0)
        # All values should be positive
        assert np.all(output > 0)
        # Larger inputs should have larger outputs
        assert output[2] > output[1] > output[0]
    
    def test_softmax_batch(self):
        """Test Softmax with batch input."""
        softmax = Softmax()
        x = np.random.randn(3, 4)
        
        output = softmax.forward(x)
        
        assert output.shape == x.shape
        # Each row should sum to 1
        row_sums = np.sum(output, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(3))
    
    def test_softmax_numerical_stability(self):
        """Test Softmax numerical stability with large values."""
        softmax = Softmax()
        x = np.array([1000.0, 1001.0, 1002.0])
        
        output = softmax.forward(x)
        
        # Should not overflow
        assert np.all(np.isfinite(output))
        assert np.isclose(np.sum(output), 1.0)


class TestLayerNorm:
    """Test suite for Layer Normalization."""
    
    def test_layernorm_initialization(self):
        """Test LayerNorm initialization."""
        norm = LayerNorm(normalized_shape=10)
        assert norm is not None
        assert hasattr(norm, 'weight')
        assert hasattr(norm, 'bias')
        assert hasattr(norm, 'eps')
        assert hasattr(norm, 'normalized_shape')
    
    def test_layernorm_parameters(self):
        """Test LayerNorm parameter shapes."""
        normalized_shape = 8
        norm = LayerNorm(normalized_shape)
        
        # Weight should be initialized to ones
        assert norm.weight.shape == (normalized_shape,)
        np.testing.assert_array_almost_equal(norm.weight, np.ones(normalized_shape))
        
        # Bias should be initialized to zeros
        assert norm.bias.shape == (normalized_shape,)
        np.testing.assert_array_almost_equal(norm.bias, np.zeros(normalized_shape))
    
    def test_layernorm_forward_1d(self):
        """Test LayerNorm forward pass with 1D input."""
        norm = LayerNorm(normalized_shape=4)
        x = np.array([1.0, 2.0, 3.0, 4.0])
        
        output = norm.forward(x)
        
        assert output.shape == x.shape
        # Output should be normalized (mean ~0, std ~1)
        assert np.isclose(np.mean(output), 0.0, atol=1e-6)
        assert np.isclose(np.std(output), 1.0, atol=1e-6)
    
    def test_layernorm_forward_2d(self):
        """Test LayerNorm forward pass with 2D input."""
        norm = LayerNorm(normalized_shape=5)
        x = np.random.randn(3, 5)
        
        output = norm.forward(x)
        
        assert output.shape == x.shape
        # Each row should be normalized
        for i in range(3):
            assert np.isclose(np.mean(output[i]), 0.0, atol=1e-6)
            assert np.isclose(np.std(output[i]), 1.0, atol=1e-6)
    
    def test_layernorm_with_learned_params(self):
        """Test LayerNorm with modified weight and bias."""
        norm = LayerNorm(normalized_shape=3)
        # Set custom weight and bias
        norm.weight = np.array([2.0, 1.0, 0.5])
        norm.bias = np.array([1.0, 0.0, -1.0])
        
        x = np.array([1.0, 2.0, 3.0])
        output = norm.forward(x)
        
        # Should apply normalization then scale and shift
        mean = np.mean(x)
        var = np.var(x)
        normalized = (x - mean) / np.sqrt(var + norm.eps)
        expected = normalized * norm.weight + norm.bias
        np.testing.assert_array_almost_equal(output, expected)
    
    def test_layernorm_eps_handling(self):
        """Test LayerNorm handles small variance with eps."""
        norm = LayerNorm(normalized_shape=3, eps=1e-5)
        # All same values (zero variance)
        x = np.array([2.0, 2.0, 2.0])
        
        output = norm.forward(x)
        
        # Should not crash and should return bias values
        assert np.all(np.isfinite(output))
        expected = norm.bias  # Since normalized values are 0
        np.testing.assert_array_almost_equal(output, expected)


class TestEmbedding:
    """Test suite for Embedding layer."""
    
    def test_embedding_initialization(self):
        """Test Embedding layer initialization."""
        embedding = Embedding(vocab_size=100, embed_dim=64)
        assert embedding is not None
        assert hasattr(embedding, 'weight')
        assert hasattr(embedding, 'vocab_size')
        assert hasattr(embedding, 'embed_dim')
    
    def test_embedding_weight_shape(self):
        """Test Embedding weight has correct shape."""
        vocab_size, embed_dim = 1000, 128
        embedding = Embedding(vocab_size, embed_dim)
        
        assert embedding.weight.shape == (vocab_size, embed_dim)
        assert embedding.vocab_size == vocab_size
        assert embedding.embed_dim == embed_dim
    
    def test_embedding_forward_single_token(self):
        """Test Embedding forward pass with single token."""
        embedding = Embedding(vocab_size=10, embed_dim=5)
        token_id = 3
        
        output = embedding.forward(token_id)
        
        assert output.shape == (5,)
        # Should return the 3rd row of weight matrix
        expected = embedding.weight[token_id]
        np.testing.assert_array_almost_equal(output, expected)
    
    def test_embedding_forward_sequence(self):
        """Test Embedding forward pass with sequence of tokens."""
        embedding = Embedding(vocab_size=10, embed_dim=4)
        token_ids = np.array([1, 5, 2, 8])
        
        output = embedding.forward(token_ids)
        
        assert output.shape == (4, 4)  # (seq_len, embed_dim)
        # Each row should be the corresponding embedding
        for i, token_id in enumerate(token_ids):
            expected = embedding.weight[token_id]
            np.testing.assert_array_almost_equal(output[i], expected)
    
    def test_embedding_forward_batch(self):
        """Test Embedding forward pass with batch of sequences."""
        embedding = Embedding(vocab_size=20, embed_dim=6)
        token_ids = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        output = embedding.forward(token_ids)
        
        assert output.shape == (3, 3, 6)  # (batch_size, seq_len, embed_dim)
        # Check each embedding
        for i in range(3):
            for j in range(3):
                expected = embedding.weight[token_ids[i, j]]
                np.testing.assert_array_almost_equal(output[i, j], expected)
    
    def test_embedding_out_of_bounds(self):
        """Test Embedding handles out-of-bounds indices gracefully."""
        embedding = Embedding(vocab_size=10, embed_dim=5)
        
        # Should raise error for out-of-bounds access
        with pytest.raises(IndexError):
            embedding.forward(15)  # 15 >= vocab_size
    
    def test_embedding_weight_initialization(self):
        """Test Embedding weight initialization."""
        embedding = Embedding(vocab_size=100, embed_dim=50)
        
        # Weights should be small random values
        assert np.all(np.abs(embedding.weight) < 1.0)
        # Should not be all zeros
        assert not np.allclose(embedding.weight, 0)
        # Should have reasonable variance
        assert 0.01 < np.var(embedding.weight) < 0.1


class TestLayerIntegration:
    """Integration tests for combining layers."""
    
    def test_linear_relu_chain(self):
        """Test chaining Linear and ReLU layers."""
        linear = Linear(in_features=4, out_features=3)
        relu = ReLU()
        
        x = np.array([-1.0, 0.5, 1.0, -0.5])
        
        # Forward pass through both layers
        linear_out = linear.forward(x)
        final_out = relu.forward(linear_out)
        
        assert final_out.shape == (3,)
        assert np.all(final_out >= 0)  # ReLU ensures non-negative
    
    def test_embedding_linear_chain(self):
        """Test chaining Embedding and Linear layers."""
        embedding = Embedding(vocab_size=10, embed_dim=5)
        linear = Linear(in_features=5, out_features=3)
        
        token_ids = np.array([1, 3, 5])
        
        # Forward pass through both layers
        embed_out = embedding.forward(token_ids)
        linear_out = linear.forward(embed_out)
        
        assert embed_out.shape == (3, 5)
        assert linear_out.shape == (3, 3)
    
    def test_full_mlp_block(self):
        """Test a complete MLP block: Linear -> LayerNorm -> GELU -> Linear."""
        linear1 = Linear(in_features=8, out_features=16)
        norm = LayerNorm(normalized_shape=16)
        gelu = GELU()
        linear2 = Linear(in_features=16, out_features=4)
        
        x = np.random.randn(8)
        
        # Forward pass through all layers
        out1 = linear1.forward(x)
        out2 = norm.forward(out1)
        out3 = gelu.forward(out2)
        final_out = linear2.forward(out3)
        
        assert final_out.shape == (4,)
        assert np.all(np.isfinite(final_out))


# Fixtures for common test data
@pytest.fixture
def sample_linear_layer():
    """Fixture providing a Linear layer with known weights."""
    layer = Linear(in_features=3, out_features=2)
    # Set known weights for predictable testing
    layer.weight = np.array([[1.0, 0.5], [0.0, 1.0], [-1.0, 0.5]])
    layer.bias = np.array([0.1, -0.1])
    return layer


@pytest.fixture
def sample_embedding():
    """Fixture providing an Embedding layer with known weights."""
    embedding = Embedding(vocab_size=5, embed_dim=3)
    # Set known weights
    embedding.weight = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [1.0, 1.1, 1.2],
        [1.3, 1.4, 1.5]
    ])
    return embedding


class TestWithFixtures:
    """Tests using fixtures for predictable results."""
    
    def test_linear_with_known_weights(self, sample_linear_layer):
        """Test Linear layer with known weights."""
        x = np.array([1.0, 2.0, 3.0])
        output = sample_linear_layer.forward(x)
        
        # Manual calculation: [1*1 + 2*0 + 3*(-1), 1*0.5 + 2*1 + 3*0.5] + [0.1, -0.1]
        expected = np.array([-2.0 + 0.1, 4.0 - 0.1])
        np.testing.assert_array_almost_equal(output, expected)
    
    def test_embedding_with_known_weights(self, sample_embedding):
        """Test Embedding layer with known weights."""
        token_ids = np.array([0, 2, 4])
        output = sample_embedding.forward(token_ids)
        
        expected = np.array([
            [0.1, 0.2, 0.3],
            [0.7, 0.8, 0.9],
            [1.3, 1.4, 1.5]
        ])
        np.testing.assert_array_almost_equal(output, expected)