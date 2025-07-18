# 🧠 DumbGPT

A high-performance GPT implementation using PyTorch for educational purposes. Train transformer models with millions of parameters in seconds on Apple Silicon!

## ✨ Features

- 🚀 **PyTorch Built-ins**: Uses `nn.TransformerDecoderLayer` for maximum performance
- 🎯 **Apple Silicon Optimized**: M2 GPU acceleration with MPS backend
- 📈 **Scalable**: Train models from 15K to 4.7M+ parameters
- 💻 **Rich Terminal Interface**: Interactive TUI for model management and text generation
- 🔤 **Custom Tokenization**: Character-level tokenizer for any text corpus
- ⚡ **Fast Training**: Train 4.7M parameter models in ~6 seconds

## 🏗️ Architecture

**Modern PyTorch Implementation:**
- Token Embedding + Positional Encoding
- PyTorch `TransformerDecoder` layers (built-in optimized)
- Layer Normalization + Linear projection head
- Cross-entropy loss + Adam optimizer

**Performance on M2 MacBook Air:**
- **47,092 tokens/sec** throughput
- **10.87ms** average forward pass
- **4.7M parameters** in 6.5 seconds training time

## 🚀 Quick Start

### Installation
```bash
# Clone and install dependencies
git clone <repo-url>
cd dumbgpt
uv sync
```

### Train a Model
```bash
# Train a 4.7M parameter model (fast!)
uv run python train_pytorch.py
```

### Interactive TUI
```bash
# Launch terminal interface
uv run main.py
```

## 📊 Model Configurations

**Small Model (15K params):**
```python
config = {
    "d_model": 32,
    "num_heads": 2, 
    "d_ff": 64,
    "num_layers": 1
}
```

**Large Model (4.7M params):**
```python
config = {
    "d_model": 256,
    "num_heads": 8,
    "d_ff": 512, 
    "num_layers": 6
}
```

## 🧪 Testing

```bash
# Run tokenizer tests
uv run pytest -v
```

## 🏆 Performance Comparison

| Implementation | Parameters | Training Time | Framework |
|---------------|------------|---------------|-----------|
| NumPy (old) | 11K | ~5 minutes | Pure NumPy |
| PyTorch (new) | 4.7M | 6.5 seconds | PyTorch + MPS |

**313x more parameters, 45x faster training!**

## 📁 Project Structure

```
src/dumbgpt/
├── model/
│   └── transformer.py     # PyTorch GPT model
├── tokenizer/
│   └── tokenizer.py       # Character tokenizer  
└── tui/                   # Terminal interface
    ├── app.py
    └── app.css

train_pytorch.py          # Training script
main.py                   # TUI entry point
```

## 🎓 Educational Journey

This project demonstrates the evolution from educational NumPy code to production-ready PyTorch:

1. **Custom Implementation** → **PyTorch Built-ins**
2. **CPU-only NumPy** → **GPU-accelerated PyTorch** 
3. **Manual Gradients** → **Automatic Differentiation**
4. **Custom Optimizers** → **Built-in Adam/SGD**
5. **Slow Training** → **Real-time Performance**

## 🔧 Requirements

- Python 3.13+
- PyTorch 2.7+ (with MPS support)
- Apple Silicon Mac (for optimal performance)
- 8GB+ RAM recommended for large models

## 🎯 Next Steps

- [ ] Implement Flash Attention for longer sequences
- [ ] Add CUDA support for NVIDIA GPUs  
- [ ] Integrate with Hugging Face transformers
- [ ] Support for different tokenization schemes
- [ ] Model quantization for mobile deployment

## 📚 Learning Resources

Perfect for understanding:
- Transformer architecture fundamentals
- PyTorch best practices and optimization
- Apple Silicon ML acceleration
- Character-level language modeling
- Terminal-based ML interfaces

Built with ❤️ for learning and experimentation!