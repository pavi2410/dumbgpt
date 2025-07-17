#!/usr/bin/env python3
"""
DumbGPT Training Script

A standalone script to build, train, and save GPT models.
The TUI can then load and interact with the saved models.

Usage:
    python train_model.py
    uv run train_model.py
"""

import sys
from pathlib import Path
from typing import List

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dumbgpt.model.transformer import GPTModel
from dumbgpt.tokenizer.tokenizer import CharTokenizer
from dumbgpt.training.dataloader import DataLoader
from dumbgpt.training.optimizer import Adam
from dumbgpt.training.trainer import Trainer
from dumbgpt.training.utils import save_model


def load_corpus_data(corpus_dir: str = "corpus") -> List[str]:
    """
    Load all text data from the corpus directory.
    
    Args:
        corpus_dir: Path to corpus directory
        
    Returns:
        List of text strings from all corpus files
    """
    texts = []
    corpus_path = Path(corpus_dir)
    
    if not corpus_path.exists():
        print(f"Warning: Corpus directory '{corpus_dir}' not found")
        print("Creating sample data for demonstration...")
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
            "NumPy provides efficient numerical computations."
        ]
    
    # Load from novels directory
    novels_dir = corpus_path / "novels"
    if novels_dir.exists():
        for file_path in novels_dir.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        texts.append(content)
                        print(f"Loaded: {file_path.name}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    # Load from code directory
    code_dir = corpus_path / "code"
    if code_dir.exists():
        for file_path in code_dir.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        texts.append(content)
                        print(f"Loaded: {file_path.name}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    return texts


def create_model_config() -> dict:
    """Create model configuration."""
    return {
        "vocab_size": 128,      # Will be updated based on tokenizer
        "d_model": 64,          # Small model for quick training
        "num_heads": 4,
        "d_ff": 128,
        "num_layers": 2,        # Few layers for demonstration
        "max_seq_len": 128
    }


def main():
    """Main training function."""
    print("🧠 DumbGPT Training Script")
    print("=" * 50)
    
    # Load corpus data
    print("\n📚 Loading corpus data...")
    texts = load_corpus_data()
    print(f"Loaded {len(texts)} text samples")
    
    if not texts:
        print("❌ No training data found!")
        return
    
    # Build tokenizer
    print("\n🔤 Building tokenizer...")
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(texts)
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    
    # Create model
    print("\n🏗️ Creating model...")
    config = create_model_config()
    config["vocab_size"] = vocab_size
    
    model = GPTModel(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        max_seq_len=config["max_seq_len"]
    )
    
    print(f"Model configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create data loader
    print("\n📊 Preparing data...")
    dataloader = DataLoader(
        corpus_paths=None,  # We'll use sample_texts instead
        tokenizer=tokenizer,
        seq_length=config["max_seq_len"] // 2,  # Use shorter sequences for training
        batch_size=4,
        sample_texts=texts
    )
    
    # Create optimizer
    print("\n⚙️ Setting up optimizer...")
    optimizer = Adam(learning_rate=0.001)
    
    # Create trainer
    print("\n🔄 Creating trainer...")
    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer
    )
    
    # Train the model
    print("\n🚀 Starting training...")
    print("This may take a few minutes...")
    
    try:
        # Calculate number of steps (epochs * batches per epoch)
        data_size = dataloader.get_data_size()
        steps_per_epoch = max(1, data_size // (dataloader.batch_size * dataloader.seq_length))
        num_steps = 5 * steps_per_epoch  # 5 epochs worth of steps
        
        print(f"Training for {num_steps} steps ({steps_per_epoch} steps per epoch)")
        
        losses = trainer.train(
            num_steps=num_steps,
            eval_interval=max(1, num_steps // 10)  # Evaluate 10 times during training
        )
        print("✅ Training completed!")
        print(f"Final loss: {losses[-1]:.4f}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Save the model
    print("\n💾 Saving model...")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "trained_model.pkl"
    save_model(model, str(model_path))
    print(f"Model saved to: {model_path}")
    
    # Test generation
    print("\n🎯 Testing generation...")
    try:
        test_context = "Hello"
        context_tokens = tokenizer.encode(test_context)
        generated_samples = trainer.generate_samples(
            context_tokens=context_tokens,
            num_samples=1,
            max_length=50
        )
        
        # Decode the generated tokens
        if generated_samples:
            generated_tokens = generated_samples[0]
            generated_text = tokenizer.decode(generated_tokens)
            print(f"Context: '{test_context}'")
            print(f"Generated: '{generated_text}'")
        else:
            print("No samples generated")
    except Exception as e:
        print(f"Generation test failed: {e}")
    
    print("\n🎉 Training script completed!")
    print("You can now use the TUI to interact with the trained model:")
    print("  uv run main.py")


if __name__ == "__main__":
    main()