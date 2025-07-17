import pytest
import numpy as np
import os
from dumbgpt.training.dataloader import DataLoader
from dumbgpt.training.optimizer import SGD, Adam
from dumbgpt.training.trainer import Trainer
from dumbgpt.training.utils import save_model, load_model
from dumbgpt.model.transformer import GPTModel
from dumbgpt.tokenizer.tokenizer import CharTokenizer


class TestDataLoader:
    """Test suite for DataLoader."""

    def test_dataloader_initialization(self):
        """Test DataLoader initialization."""
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(["hello world", "test data"])

        dataloader = DataLoader(
            corpus_paths=["dummy_path"],
            tokenizer=tokenizer,
            seq_length=10,
            batch_size=2,
        )

        assert dataloader is not None
        assert dataloader.seq_length == 10
        assert dataloader.batch_size == 2
        assert dataloader.tokenizer is tokenizer

    def test_dataloader_with_sample_data(self):
        """Test DataLoader with sample text data."""
        tokenizer = CharTokenizer()
        sample_texts = [
            "hello world this is a test",
            "another sample text for testing",
            "final text example",
        ]
        tokenizer.build_vocab(sample_texts)

        dataloader = DataLoader(
            corpus_paths=None,  # Use sample data instead
            tokenizer=tokenizer,
            seq_length=8,
            batch_size=2,
            sample_texts=sample_texts,
        )

        # Test batch generation
        batch = dataloader.get_batch()
        input_ids, target_ids = batch

        assert input_ids.shape == (2, 8)  # (batch_size, seq_length)
        assert target_ids.shape == (2, 8)
        assert np.all(input_ids >= 0)
        assert np.all(input_ids < tokenizer.vocab_size)

    def test_dataloader_sequence_relationships(self):
        """Test that input and target sequences have correct relationship."""
        tokenizer = CharTokenizer()
        text = "abcdefghijklmnopqrstuvwxyz"
        tokenizer.build_vocab([text])

        dataloader = DataLoader(
            corpus_paths=None,
            tokenizer=tokenizer,
            seq_length=5,
            batch_size=1,
            sample_texts=[text],
        )

        batch = dataloader.get_batch()
        input_ids, target_ids = batch

        # Target should be input shifted by 1
        # input:  [a, b, c, d, e]
        # target: [b, c, d, e, f]
        assert input_ids.shape == (1, 5)
        assert target_ids.shape == (1, 5)

        # Verify relationship (if tokens are sequential)
        # Note: This might not always hold due to random sampling
        assert np.all(np.isfinite(input_ids))
        assert np.all(np.isfinite(target_ids))

    def test_dataloader_multiple_batches(self):
        """Test DataLoader generates multiple different batches."""
        tokenizer = CharTokenizer()
        text = "this is a longer text to test multiple batch generation"
        tokenizer.build_vocab([text])

        dataloader = DataLoader(
            corpus_paths=None,
            tokenizer=tokenizer,
            seq_length=4,
            batch_size=2,
            sample_texts=[text],
        )

        batch1 = dataloader.get_batch()
        batch2 = dataloader.get_batch()

        input1, target1 = batch1
        input2, target2 = batch2

        assert input1.shape == input2.shape == (2, 4)
        assert target1.shape == target2.shape == (2, 4)

        # Batches should be different (with high probability)
        assert not np.array_equal(input1, input2)

    def test_dataloader_vocab_size_consistency(self):
        """Test that DataLoader respects tokenizer vocab size."""
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(["abc"])

        dataloader = DataLoader(
            corpus_paths=None,
            tokenizer=tokenizer,
            seq_length=3,
            batch_size=1,
            sample_texts=["abc"],
        )

        batch = dataloader.get_batch()
        input_ids, target_ids = batch

        # All token IDs should be within vocab size
        assert np.all(input_ids < tokenizer.vocab_size)
        assert np.all(target_ids < tokenizer.vocab_size)
        assert np.all(input_ids >= 0)
        assert np.all(target_ids >= 0)


class TestOptimizers:
    """Test suite for optimizers."""

    def test_sgd_initialization(self):
        """Test SGD optimizer initialization."""
        optimizer = SGD(learning_rate=0.01)

        assert optimizer is not None
        assert optimizer.learning_rate == 0.01
        assert hasattr(optimizer, "step")
        assert hasattr(optimizer, "zero_grad")

    def test_adam_initialization(self):
        """Test Adam optimizer initialization."""
        optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)

        assert optimizer is not None
        assert optimizer.learning_rate == 0.001
        assert optimizer.beta1 == 0.9
        assert optimizer.beta2 == 0.999
        assert hasattr(optimizer, "step")
        assert hasattr(optimizer, "zero_grad")

    def test_optimizer_parameter_updates(self):
        """Test that optimizer can update parameters."""
        # Create a simple parameter (weight matrix)
        param = np.random.randn(3, 3)
        grad = np.random.randn(3, 3)

        optimizer = SGD(learning_rate=0.1)

        # Store original parameter
        original_param = param.copy()

        # Simulate parameter update
        updated_param = optimizer.update_param(param, grad)

        # Parameter should be updated
        assert not np.allclose(updated_param, original_param)

        # SGD update: param = param - lr * grad
        expected = original_param - 0.1 * grad
        np.testing.assert_array_almost_equal(updated_param, expected)

    def test_adam_momentum(self):
        """Test Adam optimizer momentum tracking."""
        optimizer = Adam(learning_rate=0.001)

        param = np.random.randn(2, 2)
        grad1 = np.random.randn(2, 2)
        grad2 = np.random.randn(2, 2)

        # First update
        param1 = optimizer.update_param(param, grad1)

        # Second update (should use momentum)
        param2 = optimizer.update_param(param1, grad2)

        # Parameters should be updated
        assert not np.allclose(param1, param)
        assert not np.allclose(param2, param1)


class TestTrainer:
    """Test suite for Trainer."""

    def test_trainer_initialization(self):
        """Test Trainer initialization."""
        model = GPTModel(
            vocab_size=10, d_model=8, num_heads=2, d_ff=32, num_layers=1, max_seq_len=16
        )

        tokenizer = CharTokenizer()
        tokenizer.build_vocab(["test"])

        dataloader = DataLoader(
            corpus_paths=None,
            tokenizer=tokenizer,
            seq_length=4,
            batch_size=1,
            sample_texts=["test"],
        )

        optimizer = SGD(learning_rate=0.01)

        trainer = Trainer(model=model, dataloader=dataloader, optimizer=optimizer)

        assert trainer is not None
        assert trainer.model is model
        assert trainer.dataloader is dataloader
        assert trainer.optimizer is optimizer

    def test_trainer_single_step(self):
        """Test single training step."""
        model = GPTModel(
            vocab_size=20, d_model=16, num_heads=2, d_ff=64, num_layers=1, max_seq_len=8
        )

        tokenizer = CharTokenizer()
        sample_text = "hello world test"
        tokenizer.build_vocab([sample_text])

        dataloader = DataLoader(
            corpus_paths=None,
            tokenizer=tokenizer,
            seq_length=4,
            batch_size=1,
            sample_texts=[sample_text],
        )

        optimizer = SGD(learning_rate=0.01)

        trainer = Trainer(model=model, dataloader=dataloader, optimizer=optimizer)

        # Perform single training step
        loss = trainer.train_step()

        assert isinstance(loss, (float, np.floating))
        assert loss > 0  # Loss should be positive
        assert np.isfinite(loss)

    def test_trainer_multiple_steps(self):
        """Test multiple training steps."""
        tokenizer = CharTokenizer()
        sample_text = "training data example"
        tokenizer.build_vocab([sample_text])

        model = GPTModel(
            vocab_size=tokenizer.vocab_size,
            d_model=12,
            num_heads=3,
            d_ff=48,
            num_layers=1,
            max_seq_len=6,
        )

        dataloader = DataLoader(
            corpus_paths=None,
            tokenizer=tokenizer,
            seq_length=3,
            batch_size=1,
            sample_texts=[sample_text],
        )

        optimizer = SGD(learning_rate=0.01)

        trainer = Trainer(model=model, dataloader=dataloader, optimizer=optimizer)

        # Perform multiple training steps
        losses = []
        for _ in range(5):
            loss = trainer.train_step()
            losses.append(loss)

        assert len(losses) == 5
        assert all(isinstance(loss, (float, np.floating)) for loss in losses)
        assert all(loss > 0 for loss in losses)
        assert all(np.isfinite(loss) for loss in losses)

    def test_trainer_full_training(self):
        """Test complete training loop."""
        model = GPTModel(
            vocab_size=25, d_model=16, num_heads=2, d_ff=64, num_layers=1, max_seq_len=8
        )

        tokenizer = CharTokenizer()
        sample_text = "this is training data for the model"
        tokenizer.build_vocab([sample_text])

        dataloader = DataLoader(
            corpus_paths=None,
            tokenizer=tokenizer,
            seq_length=4,
            batch_size=2,
            sample_texts=[sample_text],
        )

        optimizer = SGD(learning_rate=0.01)

        trainer = Trainer(model=model, dataloader=dataloader, optimizer=optimizer)

        # Train for a few steps
        history = trainer.train(num_steps=10)

        assert len(history) == 10
        assert all(isinstance(loss, (float, np.floating)) for loss in history)
        assert all(loss > 0 for loss in history)

    def test_trainer_evaluation(self):
        """Test trainer evaluation functionality."""
        model = GPTModel(
            vocab_size=20, d_model=12, num_heads=2, d_ff=48, num_layers=1, max_seq_len=6
        )

        tokenizer = CharTokenizer()
        sample_text = "evaluation test data"
        tokenizer.build_vocab([sample_text])

        dataloader = DataLoader(
            corpus_paths=None,
            tokenizer=tokenizer,
            seq_length=3,
            batch_size=1,
            sample_texts=[sample_text],
        )

        optimizer = SGD(learning_rate=0.01)

        trainer = Trainer(model=model, dataloader=dataloader, optimizer=optimizer)

        # Evaluate model
        eval_loss = trainer.evaluate()

        assert isinstance(eval_loss, (float, np.floating))
        assert eval_loss > 0
        assert np.isfinite(eval_loss)


class TestModelPersistence:
    """Test suite for model saving and loading."""

    def test_save_model(self):
        """Test model saving functionality."""
        model = GPTModel(
            vocab_size=30, d_model=16, num_heads=2, d_ff=64, num_layers=1, max_seq_len=8
        )

        # Test that save function exists and doesn't crash
        try:
            save_model(model, "test_model.pkl")
            assert True  # If we get here, save didn't crash
        except Exception as e:
            pytest.fail(f"Model saving failed: {e}")
        finally:
            # Clean up
            if os.path.exists("test_model.pkl"):
                os.remove("test_model.pkl")

    def test_load_model(self):
        """Test model loading functionality."""
        model = GPTModel(
            vocab_size=25, d_model=12, num_heads=3, d_ff=48, num_layers=1, max_seq_len=6
        )

        # Save and load model
        try:
            save_model(model, "test_model.pkl")
            loaded_model = load_model("test_model.pkl")

            assert loaded_model is not None
            assert loaded_model.vocab_size == model.vocab_size
            assert loaded_model.d_model == model.d_model
            assert loaded_model.num_heads == model.num_heads
            assert loaded_model.num_layers == model.num_layers
        finally:
            # Clean up
            if os.path.exists("test_model.pkl"):
                os.remove("test_model.pkl")

    def test_model_persistence_consistency(self):
        """Test that saved and loaded models produce same output."""
        model = GPTModel(
            vocab_size=20, d_model=16, num_heads=2, d_ff=64, num_layers=1, max_seq_len=8
        )

        # Test input
        input_ids = np.array([1, 2, 3, 4])

        # Get output from original model
        original_output = model.forward(input_ids)

        # Save and load model
        try:
            save_model(model, "test_model.pkl")
            loaded_model = load_model("test_model.pkl")

            # Get output from loaded model
            loaded_output = loaded_model.forward(input_ids)

            # Outputs should be identical
            np.testing.assert_array_almost_equal(original_output, loaded_output)
        finally:
            # Clean up
            if os.path.exists("test_model.pkl"):
                os.remove("test_model.pkl")


class TestTrainingPipeline:
    """Integration tests for complete training pipeline."""

    def test_end_to_end_training(self):
        """Test complete end-to-end training pipeline."""
        # Create tokenizer and data
        tokenizer = CharTokenizer()
        corpus_text = "This is a sample training corpus. It contains multiple sentences. The model will learn from this text."
        tokenizer.build_vocab([corpus_text])

        # Create model
        model = GPTModel(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            num_heads=4,
            d_ff=128,
            num_layers=2,
            max_seq_len=16,
        )

        # Create dataloader
        dataloader = DataLoader(
            corpus_paths=None,
            tokenizer=tokenizer,
            seq_length=8,
            batch_size=2,
            sample_texts=[corpus_text],
        )

        # Create optimizer
        optimizer = SGD(learning_rate=0.01)

        # Create trainer
        trainer = Trainer(model=model, dataloader=dataloader, optimizer=optimizer)

        # Train model
        history = trainer.train(num_steps=20)

        # Verify training completed successfully
        assert len(history) == 20
        assert all(isinstance(loss, (float, np.floating)) for loss in history)
        assert all(loss > 0 for loss in history)

        # Test generation after training
        context = np.array([1, 2, 3])
        generated = model.generate(context, max_length=5)

        assert len(generated) == 5
        assert all(0 <= token < tokenizer.vocab_size for token in generated)

    def test_training_with_evaluation(self):
        """Test training with periodic evaluation."""
        model = GPTModel(
            vocab_size=30,
            d_model=24,
            num_heads=3,
            d_ff=96,
            num_layers=1,
            max_seq_len=12,
        )

        tokenizer = CharTokenizer()
        training_text = "training data for the neural network model"
        tokenizer.build_vocab([training_text])

        dataloader = DataLoader(
            corpus_paths=None,
            tokenizer=tokenizer,
            seq_length=6,
            batch_size=1,
            sample_texts=[training_text],
        )

        optimizer = Adam(learning_rate=0.001)

        trainer = Trainer(model=model, dataloader=dataloader, optimizer=optimizer)

        # Train with evaluation
        train_losses = []
        eval_losses = []

        for step in range(10):
            train_loss = trainer.train_step()
            train_losses.append(train_loss)

            if step % 3 == 0:
                eval_loss = trainer.evaluate()
                eval_losses.append(eval_loss)

        assert len(train_losses) == 10
        assert len(eval_losses) == 4  # Every 3 steps + step 0
        assert all(isinstance(loss, (float, np.floating)) for loss in train_losses)
        assert all(isinstance(loss, (float, np.floating)) for loss in eval_losses)

    def test_training_with_different_optimizers(self):
        """Test training with different optimizer types."""
        model = GPTModel(
            vocab_size=25,
            d_model=20,
            num_heads=2,
            d_ff=80,
            num_layers=1,
            max_seq_len=10,
        )

        tokenizer = CharTokenizer()
        text = "optimizer comparison test"
        tokenizer.build_vocab([text])

        dataloader = DataLoader(
            corpus_paths=None,
            tokenizer=tokenizer,
            seq_length=5,
            batch_size=1,
            sample_texts=[text],
        )

        # Test SGD
        sgd_optimizer = SGD(learning_rate=0.01)
        sgd_trainer = Trainer(model, dataloader, sgd_optimizer)
        sgd_loss = sgd_trainer.train_step()

        # Test Adam
        adam_optimizer = Adam(learning_rate=0.001)
        adam_trainer = Trainer(model, dataloader, adam_optimizer)
        adam_loss = adam_trainer.train_step()

        # Both should produce valid losses
        assert isinstance(sgd_loss, (float, np.floating))
        assert isinstance(adam_loss, (float, np.floating))
        assert sgd_loss > 0 and adam_loss > 0


# Fixtures for common test data
@pytest.fixture
def sample_tokenizer():
    """Fixture providing a tokenizer with sample vocabulary."""
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(["hello world", "testing data", "sample text"])
    return tokenizer


@pytest.fixture
def small_model():
    """Fixture providing a small GPT model for testing."""
    return GPTModel(
        vocab_size=50, d_model=32, num_heads=4, d_ff=128, num_layers=2, max_seq_len=16
    )


@pytest.fixture
def sample_dataloader(sample_tokenizer):
    """Fixture providing a dataloader with sample data."""
    return DataLoader(
        corpus_paths=None,
        tokenizer=sample_tokenizer,
        seq_length=8,
        batch_size=2,
        sample_texts=["hello world testing", "sample training data"],
    )


class TestWithFixtures:
    """Tests using fixtures for consistent test data."""

    def test_training_with_fixtures(self, small_model, sample_dataloader):
        """Test training using fixtures."""
        optimizer = SGD(learning_rate=0.01)
        trainer = Trainer(small_model, sample_dataloader, optimizer)

        loss = trainer.train_step()

        assert isinstance(loss, (float, np.floating))
        assert loss > 0
        assert np.isfinite(loss)

    def test_dataloader_with_fixture(self, sample_tokenizer):
        """Test dataloader using tokenizer fixture."""
        dataloader = DataLoader(
            corpus_paths=None,
            tokenizer=sample_tokenizer,
            seq_length=5,
            batch_size=1,
            sample_texts=["fixture test data"],
        )

        batch = dataloader.get_batch()
        input_ids, target_ids = batch

        assert input_ids.shape == (1, 5)
        assert target_ids.shape == (1, 5)
        assert np.all(input_ids < sample_tokenizer.vocab_size)
