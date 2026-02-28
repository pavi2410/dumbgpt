#!/usr/bin/env python3
"""
Console Model Evaluation - Test TikToken GPT model
"""

import sys
from pathlib import Path
import tiktoken
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dumbgpt.model.transformer import GPTModel


def load_model(model_path: str = "models/best_model.pt"):
    """Load the DumbGPT model."""
    path = Path(model_path)
    if not path.exists():
        # fallback
        path = Path("models/model.pt")
    if not path.exists():
        print(f"❌ Model not found. Run train.py first.")
        return None, None

    print(f"Loading model from {path}…")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    tokenizer = tiktoken.get_encoding("gpt2")

    model = GPTModel(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        max_seq_len=config["max_seq_len"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"✅ Loaded! {params:,} params  vocab={tokenizer.n_vocab:,}  "
          f"d_model={config['d_model']}  layers={config['num_layers']}")
    return model, tokenizer


def evaluate_prompt(prompt, model, tokenizer, prompt_num):
    """Evaluate a single prompt and display results."""
    print(f"\n📝 Prompt {prompt_num}: '{prompt}'")
    print("=" * 80)
    
    try:
        # Tokenize input
        tokens = tokenizer.encode(prompt)
        context = torch.tensor(tokens).unsqueeze(0)
        
        # Generate response
        with torch.no_grad():
            out = model.generate(context, max_new_tokens=50, temperature=0.8, top_k=50)
            new_tokens = out[0, len(tokens):].tolist()
            response = tokenizer.decode(new_tokens)

        print(f"🤖 Response: '{response}'")
        print(f"📊 Tokens: {len(tokens)} → {len(tokens) + len(new_tokens)}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print("-" * 80)


def main():
    """Run model evaluation."""
    print("🧠 DumbGPT TikToken Model Evaluation")
    print("=" * 50)
    
    # Test prompts
    prompts = [
        "Hello, how are you?",
        "The quick brown fox",
        "Once upon a time",
        "What is the meaning of",
        "In a galaxy far far away",
        "def factorial(n):",
        "Alice was beginning to get",
        "function mergeSort(arr) {",
        "The journey of a thousand miles",
        "import numpy as np"
    ]
    
    # Load model
    model, tokenizer = load_model()
    if not model or not tokenizer:
        return
    
    print(f"\n🚀 Evaluating {len(prompts)} prompts...")
    
    # Run evaluation on all prompts
    for i, prompt in enumerate(prompts, 1):
        evaluate_prompt(prompt, model, tokenizer, i)
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()