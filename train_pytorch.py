#!/usr/bin/env python3
"""
DumbGPT PyTorch Training Script

Simplified training script using PyTorch for M2 optimization.
"""

import sys
import torch
import torch.optim as optim
from pathlib import Path
from typing import List
import time

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dumbgpt.model.transformer import GPTModel
from dumbgpt.tokenizer.tokenizer import CharTokenizer


def load_sample_data() -> List[str]:
    """Load sample training data."""
    return [
        "Hello world! This is a sample text for training our GPT model.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating and powerful.",
        "Natural language processing helps computers understand text.",
        "Deep learning models can generate human-like text.",
        "Transformers revolutionized the field of AI.",
        "GPT models use attention mechanisms to process sequences.",
        "Training neural networks requires careful optimization.",
        "Python is a great language for machine learning.",
        "PyTorch provides excellent GPU acceleration."
    ]


def create_dataset(texts: List[str], tokenizer, seq_len: int):
    """Create training dataset from texts."""
    # Tokenize all texts
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
    
    # Create sequences
    inputs = []
    targets = []
    
    for i in range(0, len(all_tokens) - seq_len, seq_len // 2):  # Overlapping sequences
        if i + seq_len + 1 < len(all_tokens):
            input_seq = all_tokens[i:i + seq_len]
            target_seq = all_tokens[i + 1:i + seq_len + 1]
            inputs.append(input_seq)
            targets.append(target_seq)
    
    return torch.tensor(inputs), torch.tensor(targets)


def train_step(model, inputs, targets, optimizer, device):
    """Single training step."""
    model.train()
    
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    optimizer.zero_grad()
    
    # Forward pass
    loss = model.get_loss(inputs, targets)
    
    # Backward pass
    loss.backward()
    
    # Update parameters
    optimizer.step()
    
    return loss.item()


def main():
    """Main training function."""
    print("🧠 DumbGPT PyTorch Training")
    print("=" * 50)
    
    # Check if MPS is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")
    
    print(f"PyTorch version: {torch.__version__}")
    
    # Load data
    print("\n📚 Loading training data...")
    texts = load_sample_data()
    print(f"Loaded {len(texts)} text samples")
    
    # Build tokenizer
    print("\n🔤 Building tokenizer...")
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(texts)
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    
    # Model configuration - 1M+ parameters!
    config = {
        "vocab_size": vocab_size,
        "d_model": 256,         # Larger embedding
        "num_heads": 8,         # More attention heads
        "d_ff": 512,            # Larger feed-forward
        "num_layers": 6,        # More transformer layers
        "max_seq_len": 128      # Longer sequences
    }
    
    print("\n🏗️ Creating model...")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create model
    model = GPTModel(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        max_seq_len=config["max_seq_len"]
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create dataset
    print("\n📊 Preparing dataset...")
    inputs, targets = create_dataset(texts, tokenizer, config["max_seq_len"] // 2)
    print(f"Created {len(inputs)} training sequences")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    print("\n🚀 Starting training...")
    num_epochs = 10
    batch_size = 4
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Simple batching
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i + batch_size]
            batch_targets = targets[i:i + batch_size]
            
            if len(batch_inputs) == batch_size:  # Only full batches
                loss = train_step(model, batch_inputs, batch_targets, optimizer, device)
                epoch_loss += loss
                num_batches += 1
        
        avg_loss = epoch_loss / max(1, num_batches)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")
    
    print("✅ Training completed!")
    
    # Test generation
    print("\n🎯 Testing generation...")
    model.eval()
    test_context = "Hello"
    context_tokens = tokenizer.encode(test_context)
    context_tensor = torch.tensor(context_tokens).unsqueeze(0).to(device)
    
    with torch.no_grad():
        generated = model.generate(context_tensor, max_length=20, temperature=1.0)
        generated_text = tokenizer.decode(generated.tolist())
        print(f"Context: '{test_context}'")
        print(f"Generated: '{generated_text}'")
    
    # Save model
    print("\n💾 Saving model...")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "pytorch_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'tokenizer_vocab': tokenizer.vocab
    }, model_path)
    print(f"Model saved to: {model_path}")
    
    print(f"\n🎉 Training completed in {time.time() - start_time:.1f}s!")


if __name__ == "__main__":
    main()