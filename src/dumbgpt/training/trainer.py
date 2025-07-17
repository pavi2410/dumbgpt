import numpy as np
import os
from typing import List, Optional
from ..model.layers import Softmax


class Trainer:
    """
    Trainer class for GPT model training.

    Handles the complete training pipeline including forward pass,
    loss calculation, backward pass, and parameter updates.
    """

    def __init__(self, model, dataloader, optimizer):
        """
        Initialize Trainer.

        Args:
            model: GPTModel instance
            dataloader: DataLoader instance
            optimizer: Optimizer instance (SGD or Adam)
        """
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.softmax = Softmax()

    def _compute_loss(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute cross-entropy loss.

        Args:
            logits: Model predictions with shape (batch_size, seq_len, vocab_size)
            targets: Target token IDs with shape (batch_size, seq_len)

        Returns:
            Cross-entropy loss
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Reshape for easier computation
        logits_flat = logits.reshape(
            -1, vocab_size
        )  # (batch_size * seq_len, vocab_size)
        targets_flat = targets.reshape(-1)  # (batch_size * seq_len,)

        # Apply softmax to get probabilities
        probs = self.softmax.forward(logits_flat)

        # Compute cross-entropy loss
        # loss = -sum(log(prob[target_class]))
        loss = 0.0
        for i in range(len(targets_flat)):
            target_class = targets_flat[i]
            # Add small epsilon to avoid log(0)
            prob = max(probs[i, target_class], 1e-10)
            loss += -np.log(prob)

        # Average over all tokens
        return loss / len(targets_flat)

    def _compute_gradients(self, logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Compute gradients for cross-entropy loss.

        Args:
            logits: Model predictions with shape (batch_size, seq_len, vocab_size)
            targets: Target token IDs with shape (batch_size, seq_len)

        Returns:
            Gradients with respect to logits
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Reshape for easier computation
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        # Apply softmax to get probabilities
        probs = self.softmax.forward(logits_flat)

        # Compute gradients: d_loss/d_logits = prob - target_onehot
        gradients = probs.copy()

        # Subtract 1 from the target class probabilities
        for i in range(len(targets_flat)):
            target_class = targets_flat[i]
            gradients[i, target_class] -= 1.0

        # Average over batch
        gradients = gradients / len(targets_flat)

        # Reshape back to original shape
        return gradients.reshape(batch_size, seq_len, vocab_size)

    def _backward_pass(self, model_output: np.ndarray, targets: np.ndarray) -> dict:
        """
        Perform backward pass to compute parameter gradients.

        Args:
            model_output: Model predictions
            targets: Target token IDs

        Returns:
            Dictionary of parameter gradients
        """
        # Compute loss gradients
        # loss_gradients = self._compute_gradients(model_output, targets)

        # For this implementation, we'll use a simple approximation
        # In practice, this would involve proper backpropagation through the model

        # Get all model parameters
        parameters = self._get_model_parameters()
        gradients = {}

        # Compute approximate gradients using finite differences
        epsilon = 1e-5

        for param_name, param in parameters.items():
            grad = np.zeros_like(param)

            # For each parameter, compute gradient using finite differences
            it = np.nditer(param, flags=["multi_index", "refs_ok"])
            while not it.finished:
                idx = it.multi_index

                # Save original value
                original_value = param[idx]

                # Compute loss with positive perturbation
                param[idx] = original_value + epsilon
                loss_pos = self._compute_loss(model_output, targets)

                # Compute loss with negative perturbation
                param[idx] = original_value - epsilon
                loss_neg = self._compute_loss(model_output, targets)

                # Compute gradient
                grad[idx] = (loss_pos - loss_neg) / (2 * epsilon)

                # Restore original value
                param[idx] = original_value

                it.iternext()

            gradients[param_name] = grad

        return gradients

    def _get_model_parameters(self) -> dict:
        """
        Get all model parameters.

        Returns:
            Dictionary of parameter arrays
        """
        parameters = {}

        # Embedding parameters
        parameters["embedding.weight"] = self.model.embedding.weight

        # Transformer block parameters
        for i, block in enumerate(self.model.transformer_blocks):
            # Attention parameters
            parameters[f"block_{i}.attention.W_q.weight"] = block.attention.W_q.weight
            parameters[f"block_{i}.attention.W_q.bias"] = block.attention.W_q.bias
            parameters[f"block_{i}.attention.W_k.weight"] = block.attention.W_k.weight
            parameters[f"block_{i}.attention.W_k.bias"] = block.attention.W_k.bias
            parameters[f"block_{i}.attention.W_v.weight"] = block.attention.W_v.weight
            parameters[f"block_{i}.attention.W_v.bias"] = block.attention.W_v.bias
            parameters[f"block_{i}.attention.W_o.weight"] = block.attention.W_o.weight
            parameters[f"block_{i}.attention.W_o.bias"] = block.attention.W_o.bias

            # Feed-forward parameters
            parameters[f"block_{i}.feed_forward.linear1.weight"] = (
                block.feed_forward.linear1.weight
            )
            parameters[f"block_{i}.feed_forward.linear1.bias"] = (
                block.feed_forward.linear1.bias
            )
            parameters[f"block_{i}.feed_forward.linear2.weight"] = (
                block.feed_forward.linear2.weight
            )
            parameters[f"block_{i}.feed_forward.linear2.bias"] = (
                block.feed_forward.linear2.bias
            )

            # Layer norm parameters
            parameters[f"block_{i}.norm1.weight"] = block.norm1.weight
            parameters[f"block_{i}.norm1.bias"] = block.norm1.bias
            parameters[f"block_{i}.norm2.weight"] = block.norm2.weight
            parameters[f"block_{i}.norm2.bias"] = block.norm2.bias

        # Final layer norm
        parameters["ln_f.weight"] = self.model.ln_f.weight
        parameters["ln_f.bias"] = self.model.ln_f.bias

        # Output layer
        parameters["lm_head.weight"] = self.model.lm_head.weight
        parameters["lm_head.bias"] = self.model.lm_head.bias

        return parameters

    def train_step(self) -> float:
        """
        Perform single training step.

        Returns:
            Training loss
        """
        # Get batch of training data
        input_ids, target_ids = self.dataloader.get_batch()

        # Forward pass
        logits = self.model.forward(input_ids)

        # Compute loss
        loss = self._compute_loss(logits, target_ids)

        # Backward pass (simplified)
        gradients = self._backward_pass(logits, target_ids)

        # Update parameters
        parameters = self._get_model_parameters()
        self.optimizer.step(parameters, gradients)

        return loss

    def evaluate(self, num_batches: int = 10) -> float:
        """
        Evaluate model on validation data.

        Args:
            num_batches: Number of batches to evaluate on

        Returns:
            Average evaluation loss
        """
        total_loss = 0.0

        for _ in range(num_batches):
            # Get batch of data
            input_ids, target_ids = self.dataloader.get_batch()

            # Forward pass (no gradients needed for evaluation)
            logits = self.model.forward(input_ids)

            # Compute loss
            loss = self._compute_loss(logits, target_ids)
            total_loss += loss

        return total_loss / num_batches

    def train(
        self,
        num_steps: int,
        eval_interval: Optional[int] = None,
        save_interval: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> List[float]:
        """
        Train model for specified number of steps.

        Args:
            num_steps: Number of training steps
            eval_interval: Interval for evaluation (if None, no evaluation)
            save_interval: Interval for saving checkpoints (if None, no saving)
            checkpoint_dir: Directory to save checkpoints

        Returns:
            List of training losses
        """
        losses = []

        for step in range(num_steps):
            # Training step
            loss = self.train_step()
            losses.append(loss)

            # Evaluation
            if eval_interval is not None and step % eval_interval == 0:
                eval_loss = self.evaluate()
                perplexity = self.compute_perplexity()
                print(
                    f"Step {step}: Train Loss = {loss:.4f}, Eval Loss = {eval_loss:.4f}, Perplexity = {perplexity:.2f}"
                )

            # Save checkpoint
            if (
                save_interval is not None
                and checkpoint_dir is not None
                and step % save_interval == 0
            ):
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"checkpoint_step_{step}.pkl"
                )
                from .utils import save_model

                save_model(self.model, checkpoint_path)
                print(f"Saved checkpoint at step {step}")

            # Progress update
            if step % 10 == 0:
                print(f"Step {step}: Loss = {loss:.4f}")

        return losses

    def compute_perplexity(self, num_batches: int = 10) -> float:
        """
        Compute perplexity on evaluation data.

        Args:
            num_batches: Number of batches to evaluate on

        Returns:
            Perplexity score
        """
        total_loss = 0.0
        total_tokens = 0

        for _ in range(num_batches):
            # Get batch of data
            input_ids, target_ids = self.dataloader.get_batch()

            # Forward pass
            logits = self.model.forward(input_ids)

            # Compute loss
            loss = self._compute_loss(logits, target_ids)
            total_loss += loss * target_ids.size
            total_tokens += target_ids.size

        # Compute average loss
        avg_loss = total_loss / total_tokens

        # Perplexity is exp(average_loss)
        perplexity = np.exp(avg_loss)

        return perplexity

    def generate_samples(
        self, context_tokens: List[int], num_samples: int = 5, max_length: int = 50
    ) -> List[List[int]]:
        """
        Generate text samples from the model.

        Args:
            context_tokens: Initial context tokens
            num_samples: Number of samples to generate
            max_length: Maximum length of generated samples

        Returns:
            List of generated token sequences
        """
        samples = []
        context = np.array(context_tokens)

        for _ in range(num_samples):
            # Generate sample
            generated = self.model.generate(context, max_length)
            samples.append(generated.tolist())

        return samples
