import pytest
import numpy as np
from dumbgpt.model.transformer import TransformerBlock, FeedForward, GPTModel


class TestFeedForward:
    """Test suite for Feed-Forward Network."""
    
    def test_ffn_initialization(self):
        """Test FeedForward initialization."""
        ffn = FeedForward(d_model=64, d_ff=256)
        assert ffn is not None
        assert hasattr(ffn, 'forward')
        assert hasattr(ffn, 'd_model')
        assert hasattr(ffn, 'd_ff')
        assert ffn.d_model == 64
        assert ffn.d_ff == 256
    
    def test_ffn_parameter_shapes(self):
        """Test FeedForward parameter shapes."""
        d_model, d_ff = 32, 128
        ffn = FeedForward(d_model=d_model, d_ff=d_ff)
        
        # Should have two linear layers
        assert hasattr(ffn, 'linear1')
        assert hasattr(ffn, 'linear2')
        assert hasattr(ffn, 'activation')
        
        # Check shapes
        assert ffn.linear1.weight.shape == (d_model, d_ff)
        assert ffn.linear2.weight.shape == (d_ff, d_model)
    
    def test_ffn_forward_single_input(self):
        """Test FeedForward forward pass with single input."""
        ffn = FeedForward(d_model=16, d_ff=64)
        
        seq_len, d_model = 5, 16
        x = np.random.randn(seq_len, d_model)
        
        output = ffn.forward(x)
        
        assert output.shape == (seq_len, d_model)
        assert np.all(np.isfinite(output))
    
    def test_ffn_forward_batch(self):
        """Test FeedForward forward pass with batch input."""
        ffn = FeedForward(d_model=32, d_ff=128)
        
        batch_size, seq_len, d_model = 2, 8, 32
        x = np.random.randn(batch_size, seq_len, d_model)
        
        output = ffn.forward(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert np.all(np.isfinite(output))
    
    def test_ffn_activation_function(self):
        """Test that FeedForward uses activation function."""
        ffn = FeedForward(d_model=8, d_ff=32)
        
        # Test with zero input
        x = np.zeros((3, 8))
        output = ffn.forward(x)
        
        # Should not be all zeros due to bias and activation
        assert output.shape == (3, 8)
    
    def test_ffn_expansion_contraction(self):
        """Test that FFN expands then contracts dimensions."""
        d_model, d_ff = 16, 64
        ffn = FeedForward(d_model=d_model, d_ff=d_ff)
        
        x = np.random.randn(4, d_model)
        
        # First linear layer should expand
        intermediate = ffn.linear1.forward(x)
        assert intermediate.shape == (4, d_ff)
        
        # After activation
        activated = ffn.activation.forward(intermediate)
        assert activated.shape == (4, d_ff)
        
        # Second linear layer should contract back
        final = ffn.linear2.forward(activated)
        assert final.shape == (4, d_model)


class TestTransformerBlock:
    """Test suite for Transformer Block."""
    
    def test_transformer_block_initialization(self):
        """Test TransformerBlock initialization."""
        block = TransformerBlock(d_model=64, num_heads=8, d_ff=256)
        assert block is not None
        assert hasattr(block, 'forward')
        assert hasattr(block, 'd_model')
        assert hasattr(block, 'num_heads')
        assert hasattr(block, 'd_ff')
    
    def test_transformer_block_components(self):
        """Test that TransformerBlock has all required components."""
        block = TransformerBlock(d_model=32, num_heads=4, d_ff=128)
        
        # Should have attention, feed-forward, and layer norms
        assert hasattr(block, 'attention')
        assert hasattr(block, 'feed_forward')
        assert hasattr(block, 'norm1')
        assert hasattr(block, 'norm2')
    
    def test_transformer_block_forward_single_sequence(self):
        """Test TransformerBlock forward pass with single sequence."""
        block = TransformerBlock(d_model=48, num_heads=6, d_ff=192)
        
        seq_len, d_model = 10, 48
        x = np.random.randn(seq_len, d_model)
        
        output = block.forward(x)
        
        assert output.shape == (seq_len, d_model)
        assert np.all(np.isfinite(output))
    
    def test_transformer_block_forward_batch(self):
        """Test TransformerBlock forward pass with batch input."""
        block = TransformerBlock(d_model=32, num_heads=4, d_ff=128)
        
        batch_size, seq_len, d_model = 3, 6, 32
        x = np.random.randn(batch_size, seq_len, d_model)
        
        output = block.forward(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert np.all(np.isfinite(output))
    
    def test_transformer_block_with_mask(self):
        """Test TransformerBlock with causal mask."""
        block = TransformerBlock(d_model=32, num_heads=4, d_ff=128)
        
        seq_len, d_model = 8, 32
        x = np.random.randn(seq_len, d_model)
        
        # Create causal mask
        mask = np.tril(np.ones((seq_len, seq_len)))
        
        output = block.forward(x, mask=mask)
        
        assert output.shape == (seq_len, d_model)
        assert np.all(np.isfinite(output))
    
    def test_transformer_block_residual_connections(self):
        """Test that residual connections are working."""
        block = TransformerBlock(d_model=32, num_heads=4, d_ff=128)
        
        seq_len, d_model = 4, 32
        x = np.random.randn(seq_len, d_model)
        
        output = block.forward(x)
        
        # Output should be different from input due to transformations
        assert not np.allclose(output, x)
        
        # But should maintain same shape
        assert output.shape == x.shape
    
    def test_transformer_block_layer_norm_order(self):
        """Test pre-norm vs post-norm architecture."""
        block = TransformerBlock(d_model=16, num_heads=2, d_ff=64)
        
        x = np.random.randn(3, 16)
        output = block.forward(x)
        
        # Should process without errors
        assert output.shape == (3, 16)
        assert np.all(np.isfinite(output))


class TestGPTModel:
    """Test suite for GPT Model."""
    
    def test_gpt_model_initialization(self):
        """Test GPTModel initialization."""
        model = GPTModel(
            vocab_size=1000,
            d_model=64,
            num_heads=8,
            d_ff=256,
            num_layers=6,
            max_seq_len=128
        )
        assert model is not None
        assert hasattr(model, 'forward')
        assert hasattr(model, 'generate')
        assert model.vocab_size == 1000
        assert model.d_model == 64
        assert model.num_layers == 6
    
    def test_gpt_model_components(self):
        """Test that GPTModel has all required components."""
        model = GPTModel(
            vocab_size=100,
            d_model=32,
            num_heads=4,
            d_ff=128,
            num_layers=3,
            max_seq_len=64
        )
        
        # Should have embedding, positional encoding, transformer blocks, and output layer
        assert hasattr(model, 'embedding')
        assert hasattr(model, 'pos_encoding')
        assert hasattr(model, 'transformer_blocks')
        assert hasattr(model, 'ln_f')
        assert hasattr(model, 'lm_head')
        
        # Check number of transformer blocks
        assert len(model.transformer_blocks) == 3
    
    def test_gpt_model_forward_single_sequence(self):
        """Test GPTModel forward pass with single sequence."""
        model = GPTModel(
            vocab_size=50,
            d_model=32,
            num_heads=4,
            d_ff=128,
            num_layers=2,
            max_seq_len=16
        )
        
        seq_len = 8
        input_ids = np.random.randint(0, 50, size=(seq_len,))
        
        output = model.forward(input_ids)
        
        assert output.shape == (seq_len, 50)  # (seq_len, vocab_size)
        assert np.all(np.isfinite(output))
    
    def test_gpt_model_forward_batch(self):
        """Test GPTModel forward pass with batch input."""
        model = GPTModel(
            vocab_size=100,
            d_model=48,
            num_heads=6,
            d_ff=192,
            num_layers=3,
            max_seq_len=32
        )
        
        batch_size, seq_len = 2, 10
        input_ids = np.random.randint(0, 100, size=(batch_size, seq_len))
        
        output = model.forward(input_ids)
        
        assert output.shape == (batch_size, seq_len, 100)  # (batch_size, seq_len, vocab_size)
        assert np.all(np.isfinite(output))
    
    def test_gpt_model_causal_mask(self):
        """Test that GPTModel applies causal mask correctly."""
        model = GPTModel(
            vocab_size=30,
            d_model=16,
            num_heads=2,
            d_ff=64,
            num_layers=2,
            max_seq_len=8
        )
        
        seq_len = 5
        input_ids = np.random.randint(0, 30, size=(seq_len,))
        
        output = model.forward(input_ids)
        
        assert output.shape == (seq_len, 30)
        assert np.all(np.isfinite(output))
    
    def test_gpt_model_generate_single_token(self):
        """Test GPTModel generation of single token."""
        model = GPTModel(
            vocab_size=20,
            d_model=16,
            num_heads=2,
            d_ff=64,
            num_layers=1,
            max_seq_len=10
        )
        
        context = np.array([1, 2, 3])
        
        next_tokens = model.generate(context, max_length=1)
        
        assert len(next_tokens) == 1
        assert isinstance(next_tokens[0], (int, np.integer))
        assert 0 <= next_tokens[0] < 20
    
    def test_gpt_model_generate_sequence(self):
        """Test GPTModel generation of sequence."""
        model = GPTModel(
            vocab_size=25,
            d_model=24,
            num_heads=3,
            d_ff=96,
            num_layers=2,
            max_seq_len=16
        )
        
        context = np.array([1, 2])
        max_length = 5
        
        generated = model.generate(context, max_length=max_length)
        
        assert len(generated) == max_length
        assert all(0 <= token < 25 for token in generated)
    
    def test_gpt_model_different_architectures(self):
        """Test GPTModel with different architecture sizes."""
        configs = [
            (50, 16, 2, 64, 1, 8),   # Small
            (100, 32, 4, 128, 2, 16), # Medium
            (200, 64, 8, 256, 4, 32), # Large
        ]
        
        for vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len in configs:
            model = GPTModel(
                vocab_size=vocab_size,
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                num_layers=num_layers,
                max_seq_len=max_seq_len
            )
            
            seq_len = min(5, max_seq_len)
            input_ids = np.random.randint(0, vocab_size, size=(seq_len,))
            
            output = model.forward(input_ids)
            
            assert output.shape == (seq_len, vocab_size)
            assert np.all(np.isfinite(output))
    
    def test_gpt_model_parameter_count(self):
        """Test that GPTModel has reasonable parameter count."""
        model = GPTModel(
            vocab_size=100,
            d_model=32,
            num_heads=4,
            d_ff=128,
            num_layers=2,
            max_seq_len=64
        )
        
        # Should have embedding, positional encoding, transformer blocks, and output layer
        # This is more of a sanity check
        assert hasattr(model, 'embedding')
        assert hasattr(model, 'pos_encoding')
        assert len(model.transformer_blocks) == 2
        assert hasattr(model, 'lm_head')


class TestTransformerIntegration:
    """Integration tests for complete transformer architecture."""
    
    def test_full_transformer_pipeline(self):
        """Test complete transformer pipeline from tokens to logits."""
        model = GPTModel(
            vocab_size=50,
            d_model=32,
            num_heads=4,
            d_ff=128,
            num_layers=2,
            max_seq_len=16
        )
        
        # Test sequence: "hello world" -> [1, 2, 3, 4, 5]
        input_ids = np.array([1, 2, 3, 4, 5])
        
        # Forward pass
        logits = model.forward(input_ids)
        
        assert logits.shape == (5, 50)
        assert np.all(np.isfinite(logits))
        
        # Should be able to get next token probabilities
        next_token_logits = logits[-1]  # Last token's predictions
        assert next_token_logits.shape == (50,)
        
        # Convert to probabilities
        probs = np.exp(next_token_logits) / np.sum(np.exp(next_token_logits))
        assert np.isclose(np.sum(probs), 1.0)
    
    def test_transformer_with_different_sequence_lengths(self):
        """Test transformer with various sequence lengths."""
        model = GPTModel(
            vocab_size=30,
            d_model=24,
            num_heads=3,
            d_ff=96,
            num_layers=2,
            max_seq_len=20
        )
        
        for seq_len in [1, 3, 5, 10, 15]:
            input_ids = np.random.randint(0, 30, size=(seq_len,))
            
            output = model.forward(input_ids)
            
            assert output.shape == (seq_len, 30)
            assert np.all(np.isfinite(output))
    
    def test_transformer_autoregressive_generation(self):
        """Test that transformer can generate autoregressively."""
        model = GPTModel(
            vocab_size=20,
            d_model=16,
            num_heads=2,
            d_ff=64,
            num_layers=1,
            max_seq_len=10
        )
        
        # Start with a simple context
        context = np.array([1])
        
        # Generate step by step
        for _ in range(3):
            logits = model.forward(context)
            next_token_logits = logits[-1]
            
            # Simple greedy sampling
            next_token = np.argmax(next_token_logits)
            context = np.append(context, next_token)
        
        assert len(context) == 4
        assert all(0 <= token < 20 for token in context)
    
    def test_transformer_batch_consistency(self):
        """Test that batch processing gives consistent results."""
        model = GPTModel(
            vocab_size=25,
            d_model=20,
            num_heads=4,
            d_ff=80,
            num_layers=1,
            max_seq_len=8
        )
        
        # Single sequence
        seq = np.array([1, 2, 3, 4])
        output_single = model.forward(seq)
        
        # Same sequence in batch
        batch = np.array([seq, seq])
        output_batch = model.forward(batch)
        
        # Results should be identical
        np.testing.assert_array_almost_equal(output_single, output_batch[0])
        np.testing.assert_array_almost_equal(output_single, output_batch[1])


# Fixtures for common test data
@pytest.fixture
def small_gpt_model():
    """Fixture providing a small GPT model for testing."""
    return GPTModel(
        vocab_size=50,
        d_model=32,
        num_heads=4,
        d_ff=128,
        num_layers=2,
        max_seq_len=16
    )


@pytest.fixture
def sample_transformer_block():
    """Fixture providing a transformer block for testing."""
    return TransformerBlock(d_model=32, num_heads=4, d_ff=128)


@pytest.fixture
def sample_feed_forward():
    """Fixture providing a feed-forward network for testing."""
    return FeedForward(d_model=16, d_ff=64)


class TestWithFixtures:
    """Tests using fixtures for consistent test data."""
    
    def test_gpt_model_with_fixture(self, small_gpt_model):
        """Test GPT model using fixture."""
        input_ids = np.array([1, 2, 3, 4, 5])
        output = small_gpt_model.forward(input_ids)
        
        assert output.shape == (5, 50)
        assert np.all(np.isfinite(output))
    
    def test_transformer_block_with_fixture(self, sample_transformer_block):
        """Test transformer block using fixture."""
        x = np.random.randn(6, 32)
        output = sample_transformer_block.forward(x)
        
        assert output.shape == (6, 32)
        assert np.all(np.isfinite(output))
    
    def test_feed_forward_with_fixture(self, sample_feed_forward):
        """Test feed-forward network using fixture."""
        x = np.random.randn(4, 16)
        output = sample_feed_forward.forward(x)
        
        assert output.shape == (4, 16)
        assert np.all(np.isfinite(output))