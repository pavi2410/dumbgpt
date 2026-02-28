# 🧠 DumbGPT

A high-performance GPT implementation using PyTorch for educational purposes. Train transformer models with millions of parameters in seconds on Apple Silicon!

## ✨ Features

- 🏗️ **Modern SLM Architecture**: RoPE + RMSNorm + SwiGLU + fused SDPA (Llama/Mistral/DeepSeek style)
- 🎯 **Multi-Backend Support**: MPS (Apple), CUDA (NVIDIA), XPU (Intel) with BF16 mixed precision
- 📈 **Two Presets**: `micro` (7M, fast iteration) and `base` (117M, main training)
- 📦 **HuggingFace Datasets**: Streams directly from `TinyStories` + `fineweb-edu` — no download step
- 💻 **Rich Terminal UI**: Coloured progress bars, tables, and panels via `rich`
- 🔤 **TikToken (GPT-2)**: BPE tokenization, 50,257 vocab
- ⚡ **Training Optimisations**: Linear warmup + cosine LR, TF32, gradient checkpointing, `torch.compile`
- 🧮 **Quantization**: INT8 dynamic quantization for faster inference
- 🔄 **Sampling**: Top-k, top-p (nucleus), and repetition penalty

## 🏗️ Architecture

Decoder-only transformer using modern SLM building blocks (same style as Llama 3 / Mistral / DeepSeek):

| Component | Implementation | vs Old GPT-2 |
|-----------|---------------|-------------|
| Position encoding | **RoPE** (rotary, no learned weights) | Learned absolute embeddings |
| Normalisation | **RMSNorm** (no mean subtraction) | LayerNorm |
| Feed-forward | **SwiGLU** `down(SiLU(gate(x)) * up(x))` | GELU MLP |
| Attention | **Fused `scaled_dot_product_attention`** (Flash Attention when available) | Manual QKV matmul |
| LR schedule | **Linear warmup + cosine decay** | Cosine only |

## 📦 Training Data

Data is **streamed directly from HuggingFace** during training — no download step needed.
Shards are cached locally (`~/.cache/huggingface/datasets`) and reused on subsequent runs.

| Dataset | Mix | Tokens (est.) | Why |
|---------|-----|--------------|-----|
| [`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories) | **30%** | ~1.5B | Coherent simple narratives; good early signal |
| [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | **70%** | ~5B+ | High-quality educational web text; broad knowledge |

Mix weights are set in `DATASET_SOURCES` in `src/dumbgpt/train.py`.

## Quick Start

### Installation
```bash
# Clone and install dependencies
git clone <repo-url>
cd dumbgpt
uv sync
```

### Train a Model
```bash
# Quick sanity check (micro, ~7M params, ~2 min)
uv run train --preset micro --epochs 1 --steps 50 --batch 8

# Main training run (base, ~117M params)
uv run train --preset base --epochs 5 --steps 1000 --batch 8 --checkpoint

# Resume from checkpoint
uv run train --preset base --resume models/best_model.pt --epochs 5 --steps 2000
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

## Model Configurations

| Preset | Params | d_model | heads | layers | seq_len | Use |
|--------|--------|---------|-------|--------|---------|-----|
| `micro` | ~7M   | 256     | 4     | 4      | 256     | Fast iteration / debugging |
| `base`  | ~117M | 768     | 12    | 12     | 1024    | Main training target |

Use `--checkpoint` flag to halve VRAM at ~15% speed cost.

**`base` chinchilla-optimal target**: ~2.3B tokens (~2000 steps × batch 8 × seq_len 1024 ≈ 16M tokens/epoch → ~150 epochs, or equivalently just run more steps).

## Optimization Features:

| Feature | Flag | Benefit | Use When |
|---------|------|---------|----------|
| BF16 Mixed Precision | Automatic on XPU | 50% VRAM, faster | Always (XPU) |
| Gradient Checkpointing | `--checkpoint` | 50% VRAM | Training `base` on 16GB |
| INT8 Quantization | `model._enable_int8()` | 50% memory, 10-30% speed | Inference on large models |

## Testing

```bash
# Run tokenizer tests
uv run pytest -v
```

## 📁 Project Structure

```
src/dumbgpt/
├── model/
│   └── transformer.py      # GPTModel: RoPE + RMSNorm + SwiGLU + fused SDPA
├── tui/
│   └── app.py              # Textual TUI (Chat + Train tabs)
├── train.py                # Training (HFStreamingDataset, warmup LR, TF32)
└── eval.py                 # Perplexity + generation eval

models/                     # Saved checkpoints
├── best_model.pt
└── model.pt
```

## 🎓 Educational Journey

This project demonstrates the evolution from educational NumPy code to production-ready PyTorch:
- **Custom NumPy** → **Custom PyTorch**
- **CPU-only** → **GPU-accelerated (MPS/CUDA/XPU)**
- **Manual Gradients** → **Automatic Differentiation**
- **Custom Optimizers** → **AdamW with Cosine LR**
- **Character Tokenizer** → **BPE (tiktoken)**

## 🔧 Requirements

- Python 3.13+
- PyTorch 2.8+ (with MPS, CUDA, or XPU support)
- MPS: Apple Silicon Mac
- CUDA: NVIDIA GPU (Linux/Windows)
- XPU: Intel Arc GPU (Windows/Linux)
- 8GB+ RAM recommended for large models

## 🎯 Next Steps

- [x] Modern SLM architecture (RoPE, RMSNorm, SwiGLU, fused SDPA)
- [x] BF16 mixed precision + gradient checkpointing
- [x] INT8 quantization for inference
- [x] HuggingFace dataset streaming (no pre-download)
- [x] Linear warmup + cosine LR decay
- [ ] KV caching for faster inference
- [ ] SFT / instruction fine-tuning (smoltalk + LIMA)
- [ ] Chat template + special tokens

## 📚 Learning Resources

Perfect for understanding:
- Transformer architecture fundamentals
- PyTorch best practices and optimization
- Apple Silicon / Intel Arc / NVIDIA GPU acceleration
- BPE tokenization with tiktoken
- Terminal-based ML interfaces

Built with ❤️ for learning and experimentation!