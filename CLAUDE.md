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
uv run main.py
```

### Project Structure
```
src/dumbgpt/
├── model/       # GPT transformer implementation
├── tokenizer/   # Text tokenization logic
├── training/    # Model training infrastructure
└── tui/         # Terminal user interface

corpus/          # Training data (novels + code samples)
models/          # Saved trained models
```

## Architecture

### Core Components

**Model Module** (`src/dumbgpt/model/`):
- Transformer implementation using NumPy
- Attention mechanisms and layer implementations
- Model serialization/deserialization for saving to `models/`

**Tokenizer Module** (`src/dumbgpt/tokenizer/`):
- Text preprocessing and tokenization
- Integration with tiktoken for GPT-style tokenization
- Handles both literary text and code samples from `corpus/`

**Training Module** (`src/dumbgpt/training/`):
- Training loop implementation
- Dataset loading from `corpus/novels/` and `corpus/code/`
- Progress tracking and metrics
- Model checkpointing to `models/`

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

- `main.py`: Direct script execution with `uv run main.py`
- Main function: `src.dumbgpt.tui.app:main`