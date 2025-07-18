#!/usr/bin/env python3
"""
Console Model Evaluation - Test TikToken GPT model
"""

import torch
from pathlib import Path

from src.dumbgpt.model.transformer import GPTModel
from src.dumbgpt.tokenizer.tiktoken_tokenizer import TikTokenTokenizer


def load_model():
    """Load the DumbGPT model."""
    model_path = Path("models/pytorch_model.pt")
    if not model_path.exists():
        print("❌ Model not found: models/pytorch_model.pt")
        return None, None
    
    print("Loading TikToken model...")
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']
    
    tokenizer = TikTokenTokenizer()
    
    model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        max_seq_len=config['max_seq_len']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded! {params:,} parameters, vocab: {tokenizer.vocab_size:,}")
    
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
            generated = model.generate(
                context, 
                max_length=50, 
                temperature=0.8
            )
            response = tokenizer.decode(generated.tolist())
        
        print(f"🤖 Response: '{response}'")
        print(f"📊 Tokens: {len(tokens)} → {len(generated.tolist())}")
        
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