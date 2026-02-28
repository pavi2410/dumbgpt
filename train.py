#!/usr/bin/env python3
"""
DumbGPT PyTorch Training Script

Trains GPT model with TikToken BPE tokenizer.
"""

import sys
import torch
import torch.optim as optim
from pathlib import Path
from typing import List
import time
from tqdm import tqdm

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dumbgpt.model.transformer import GPTModel
from dumbgpt.tokenizer.tiktoken_tokenizer import TikTokenTokenizer


def load_corpus_data() -> List[str]:
    """Load training data from corpus directory."""
    corpus_dir = Path("corpus")
    texts = []
    
    # Load novels
    novels_dir = corpus_dir / "novels"
    if novels_dir.exists():
        for novel_file in novels_dir.glob("*.txt"):
            try:
                with open(novel_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        texts.append(content)
                        print(f"Loaded {novel_file.name}: {len(content)} characters")
            except Exception as e:
                print(f"Error loading {novel_file.name}: {e}")
    
    # Load code samples from node_modules
    node_modules_dir = corpus_dir / "node_modules"
    if node_modules_dir.exists():
        code_extensions = {'.js', '.ts', '.jsx', '.tsx'}
        code_files = list(node_modules_dir.glob("**/*"))
        code_files = [f for f in code_files if f.suffix in code_extensions]
        
        for i, code_file in enumerate(code_files):
            try:
                with open(code_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().strip()
                    if content and len(content) > 50:  # Skip very short files
                        texts.append(content)
            except Exception:
                pass
        
        print(f"Loaded {len(code_files)} code files from node_modules")
    
    # Load code samples from corpus/code
    code_dir = corpus_dir / "code"
    if code_dir.exists():
        for code_file in code_dir.glob("*"):
            try:
                with open(code_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        texts.append(content)
                        print(f"Loaded {code_file.name}: {len(content)} characters")
            except Exception as e:
                print(f"Error loading {code_file.name}: {e}")
    
    return texts


def create_dataset(texts: List[str], tokenizer, seq_len: int):
    """Create training dataset from texts."""
    # Tokenize all texts
    print("Tokenizing corpus...")
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
    
    print(f"Total tokens: {len(all_tokens):,}")
    
    # Create sequences with overlap
    inputs = []
    targets = []
    
    step = seq_len // 2  # 50% overlap for more training data
    for i in range(0, len(all_tokens) - seq_len, step):
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
    print("🧠 DumbGPT TikToken Training")
    print("=" * 50)
    
    # Check if MPS is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using Apple Silicon GPU (MPS)")
    elif torch.xpu.is_available():
        device = torch.device("xpu")
        print("🚀 Using Intel GPU (XPU)")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")
    
    print(f"PyTorch version: {torch.__version__}")
    
    # Load data from corpus
    print("\n📚 Loading corpus data...")
    texts = load_corpus_data()
    print(f"Loaded {len(texts)} files from corpus")
    
    if not texts:
        print("❌ No corpus data found! Make sure corpus/ directory exists.")
        return
    
    # Setup tokenizer
    print("\n🔤 Setting up TikToken tokenizer...")
    tokenizer = TikTokenTokenizer()
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size:,}")
    
    # Model configuration
    config = {
        "vocab_size": vocab_size,
        "d_model": 256,         # Larger embedding
        "num_heads": 8,         # More attention heads
        "d_ff": 512,            # Larger feed-forward
        "num_layers": 6,        # More transformer layers
        "max_seq_len": 128      # Longer sequences
    }
    
    print("\n🏗️ Model configuration:")
    for key, value in config.items():
        print(f"  {key}: {value:,}")
    
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
    seq_len = 128  # Longer sequences for better context
    inputs, targets = create_dataset(texts, tokenizer, seq_len)
    print(f"Created {len(inputs)} training sequences")
    
    if len(inputs) == 0:
        print("❌ No training sequences created! Data too small.")
        return
    
    # Train/validation split (90/10)
    split_idx = int(len(inputs) * 0.9)
    train_inputs, val_inputs = inputs[:split_idx], inputs[split_idx:]
    train_targets, val_targets = targets[:split_idx], targets[split_idx:]
    print(f"Train: {len(train_inputs)} | Validation: {len(val_inputs)}")
    
    # Create optimizer with learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    
    # Training
    print("\n🚀 Starting training...")
    batch_size = 16  # Larger batch size for XPU
    
    start_time = time.time()
    best_val_loss = float('inf')
    
    for epoch in range(5):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Shuffle training data
        perm = torch.randperm(len(train_inputs))
        train_inputs = train_inputs[perm]
        train_targets = train_targets[perm]
        
        # Simple batching with progress bar
        num_batches = (len(train_inputs) // batch_size)
        pbar = tqdm(range(0, len(train_inputs), batch_size), desc=f"Epoch {epoch+1}/5", leave=False)
        for i in pbar:
            batch_inputs = train_inputs[i:i + batch_size]
            batch_targets = train_targets[i:i + batch_size]
            
            if len(batch_inputs) == batch_size:  # Only full batches
                loss = train_step(model, batch_inputs, batch_targets, optimizer, device)
                epoch_loss += loss
                num_batches += 1
                pbar.set_postfix({"loss": f"{loss:.4f}"})
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for i in tqdm(range(0, len(val_inputs), batch_size), desc="Validating", leave=False):
                batch_inputs = val_inputs[i:i + batch_size]
                batch_targets = val_targets[i:i + batch_size]
                if len(batch_inputs) == batch_size:
                    loss = model.get_loss(batch_inputs.to(device), batch_targets.to(device))
                    val_loss += loss.item()
                    val_batches += 1
        
        scheduler.step()
        
        avg_loss = epoch_loss / max(1, num_batches)
        avg_val_loss = val_loss / max(1, val_batches)
        elapsed = time.time() - start_time
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'tokenizer_type': 'tiktoken'
            }, "models/best_model.pt")
        
        print(f"Epoch {epoch + 1}/5 | Loss: {avg_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | Time: {elapsed:.1f}s")
    
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
        'tokenizer_type': 'tiktoken'
    }, model_path)
    print(f"Model saved to: {model_path}")
    
    training_time = time.time() - start_time
    print(f"\n🎉 Training completed in {training_time:.1f}s!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()