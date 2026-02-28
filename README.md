# 🧠 DumbGPT

A high-performance GPT implementation using PyTorch for educational purposes. Train transformer models with millions of parameters in seconds on Apple Silicon!

## ✨ Features

- 🚀 **Custom PyTorch Transformer**: Multi-head attention from scratch
- 🎯 **Multi-Backend Support**: MPS (Apple), CUDA (NVIDIA), XPU (Intel)
- 📈 **Scalable Presets**: nano (15K) → medium (2.6M) parameters
- 💻 **Rich Terminal Interface**: Interactive TUI with Textual
- 🔤 **TikToken (GPT-2)**: BPE tokenization, 50K+ vocab
- ⚡ **Fast Training**: Mixed precision, gradient clipping, streaming data

## 🏗️ Architecture

**Custom PyTorch Implementation:**
- Token Embedding + Learned Positional Embeddings
- Multi-Head Self-Attention (causal masked)
- Feed-forward layers with GELU activation
- Layer Normalization + Residual connections
- Weight tying (input/output embeddings)
- Cross-entropy loss + AdamW optimizer with cosine LR schedule

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
# Train with preset (nano/small/medium)
uv run train --preset small --epochs 5

# Train with custom settings
uv run train --preset medium --batch 32 --lr 3e-4
```

### Interactive TUI
```bash
# Launch terminal interface
uv run tui
```

### Evaluate a Model
```bash
# Run evaluation on test prompts
uv run eval
```

## 📊 Model Configurations

**Presets:**

| Preset | Parameters | d_model | heads | layers | seq_len |
|--------|-----------|---------|-------|--------|---------|
| nano   | ~200K     | 128     | 4     | 3      | 128     |
| small  | ~500K     | 256     | 4     | 4      | 128     |
| medium | ~2.6M     | 384     | 6     | 6      | 256     |

```bash
# Use a preset
uv run train --preset small --epochs 5 --batch 32
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
├── __init__.py
├── model/
│   ├── __init__.py         # exports GPTModel
│   └── transformer.py      # GPT implementation
├── tui/
│   ├── __init__.py
│   ├── app.py              # Textual TUI
│   └── README.md
├── train.py                # Training script
└── eval.py                 # Evaluation script

corpus/                     # Training data
├── novels/                 # Literary texts
├── code/                   # Code samples
└── node_modules/           # JS/TS code

models/                     # Saved checkpoints
```

## 🎓 Educational Journey

This project demonstrates the evolution from educational NumPy code to production-ready PyTorch:

1. **Custom NumPy** → **Custom PyTorch**
2. **CPU-only** → **GPU-accelerated (MPS/CUDA/XPU)**
3. **Manual Gradients** → **Automatic Differentiation**
4. **Custom Optimizers** → **AdamW with Cosine LR**
5. **Character Tokenizer** → **BPE (tiktoken)**

## 🔧 Requirements

- Python 3.13+
- PyTorch 2.8+ (with MPS, CUDA, or XPU support)
- MPS: Apple Silicon Mac
- CUDA: NVIDIA GPU (Linux/Windows)
- XPU: Intel Arc GPU (Windows/Linux)
- 8GB+ RAM recommended for large models

## 🎯 Next Steps

- [ ] Implement Flash Attention for longer sequences
- [ ] Add beam search and sampling strategies
- [ ] Integrate with Hugging Face for model sharing
- [ ] Add fine-tuning capability
- [ ] Model quantization for deployment

## 📚 Learning Resources

Perfect for understanding:
- Transformer architecture fundamentals
- PyTorch best practices and optimization
- Apple Silicon / Intel Arc / NVIDIA GPU acceleration
- BPE tokenization with tiktoken
- Terminal-based ML interfaces

Built with ❤️ for learning and experimentation!