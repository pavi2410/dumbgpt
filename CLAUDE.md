# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DumbGPT is a GPT implementation from scratch for learning purposes. The project builds a transformer language model using pure Python and NumPy, with a rich terminal user interface built with Textual.

## Development Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Run the TUI application
uv run tui

# Train a model
uv run train --preset small --epochs 5

# Evaluate a model
uv run eval

# Run tests
uv run pytest

# Run specific test file
uv run pytest tests/test_tokenizer.py

# Run tests with verbose output
uv run pytest -v
```

### Project Structure
```
src/dumbgpt/
├── model/       # GPT transformer implementation
│   ├── __init__.py
│   └── transformer.py
├── tui/         # Terminal user interface
│   ├── __init__.py
│   ├── app.py
│   └── README.md
├── train.py     # Training script
└── eval.py      # Evaluation script

corpus/          # Training data (novels + code samples)
models/          # Saved trained models
```

## Architecture

### Core Components

**Model Module** (`src/dumbgpt/model/`):
- Transformer implementation using NumPy
- Attention mechanisms and layer implementations
- Model serialization/deserialization for saving to `models/`

**Training** (`src/dumbgpt/train.py`):
- Training loop implementation with streaming dataset
- Command-line interface with presets (nano/small/medium)
- Progress tracking and metrics
- Model checkpointing to `models/`

**Evaluation** (`src/dumbgpt/eval.py`):
- Batch prompt evaluation script
- Test model on sample prompts

**TUI Module** (`src/dumbgpt/tui/`):
- Terminal interface using Textual
- Training progress visualization
- Interactive model testing
- Model management interface

### Training Data

The project includes a curated corpus:
- **Literary texts**: Alice in Wonderland, Gulliver's Travels, philosophical texts, proverbs
- **Code samples**: Java and C++ algorithm implementations, primarily LeetCode solutions

### Key Design Principles

- **CPU-first**: Designed for CPU training using NumPy
- **Educational focus**: Code should be readable and educational rather than optimized
- **Modular**: Clean separation between model, training, and UI components
- **Terminal-based**: Rich TUI for all interactions, no web interface

## Dependencies

- **NumPy**: All mathematical operations and tensor manipulations
- **Textual + Rich**: Terminal user interface and formatting
- **tiktoken**: GPT-style tokenization
- **Python 3.13+**: Latest Python features

## Entry Points

- `uv run tui`: Run the terminal UI
- `uv run train`: Train the model (use `--help` for options)
- `uv run eval`: Evaluate the model

## Curriculum

Chapter 1: Tokenization 🔤
- Build a tokenizer to convert text into numbers
- Understand vocabulary, encoding/decoding
- Handle the corpus data (novels + code)

Chapter 2: Basic Neural Network Components 🧠
- Implement linear layers, activation functions
- Build the foundation with NumPy
- Create the building blocks for transformers

Chapter 3: Attention Mechanism 👁️
- Self-attention from scratch
- Multi-head attention
- The core innovation of transformers

Chapter 4: Transformer Architecture 🏗️
- Combine attention with feed-forward networks
- Layer normalization, residual connections
- Build the complete GPT block

Chapter 5: Training Loop 🔄
- Loss functions, backpropagation
- Data loading and batching
- Model optimization

Chapter 6: TUI Interface 💻
- Build the terminal interface
- Training progress, model interaction
- Put it all together
