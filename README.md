# DumbGPT

A **Mini-GPT implementation from scratch** in TypeScript, designed for learning how transformers work.

## Features

🧠 **Complete GPT Implementation**
- Multi-head attention mechanism
- Transformer blocks with residual connections
- Layer normalization and GELU activation
- Custom tensor operations (no external ML libraries)
- BPE-style tokenizer for JavaScript/TypeScript

🎯 **Code-Focused Training**
- Trains on JavaScript/TypeScript files from `node_modules`
- Generates contextually aware code completions
- Supports various JavaScript patterns and syntax

🖥️ **Modern TUI Interface**
- Beautiful terminal interface built with Ink
- Real-time training progress visualization
- Command system for configuration
- Input history with arrow key navigation

## Installation

```bash
bun install
```

## Usage

```bash
# Start the application
bun run start

# Development mode (with file watching)
bun run dev

# Run tests
bun test
```

## Commands

- `:help` - Show help
- `:config` - Show configuration  
- `:set <key> <value>` - Change settings
- `:clear` - Clear messages
- `:quit` - Exit

## Configuration

- `contextSize` - Context window size (default: 32)
- `maxOutputTokens` - Maximum tokens to generate (default: 50)
- `modelType` - Model type: 'code' or 'text' (default: 'code')

## Project Structure

```
src/
├── models/
│   ├── gpt/
│   │   ├── tensor.ts        # Custom tensor operations
│   │   ├── tokenizer.ts     # BPE tokenizer for code
│   │   ├── attention.ts     # Multi-head attention
│   │   └── transformer.ts   # Complete GPT model
│   ├── gpt-model.ts         # GPT model integration
│   └── index.ts             # Models API
├── ui/
│   ├── hooks/
│   │   └── useInputHistory.ts  # Input handling hooks
│   ├── App.tsx              # Main application
│   ├── ChatArea.tsx         # Chat interface
│   ├── InputArea.tsx        # Input with history
│   ├── TrainingProgress.tsx # Training visualization
│   └── theme.ts             # UI theme
└── tui.tsx                  # Application entry point
```

## Model Architecture

**Mini-GPT Configuration:**
- **Parameters**: ~650K (much smaller than production models)
- **Layers**: 4 transformer blocks
- **Heads**: 8 attention heads  
- **Embedding**: 256 dimensions
- **Vocabulary**: 8000 tokens
- **Context**: 32 tokens

## Learning Objectives

This implementation demonstrates:
- How transformers work from first principles
- Multi-head attention mechanism
- Tokenization for code
- Training loop and loss calculation
- Tensor operations without external libraries

## Performance

- **Training**: CPU-only (no GPU required)
- **Speed**: ~600-800ms for full test suite
- **Memory**: Reasonable for development machines
- **Quality**: Generates coherent code tokens

## Testing

Comprehensive test suite covering:
- Tensor operations (matrix multiplication, softmax, etc.)
- Tokenization (encode/decode, training sequences)
- Attention mechanism
- Complete model training and generation

```bash
bun test  # Run all tests
```

## Next Steps

Potential improvements:
1. Implement proper backpropagation
2. Add beam search for better generation
3. Expand training data
4. Add fine-tuning capabilities
5. Implement more sophisticated attention patterns

---

**Note**: This is an educational implementation focused on understanding transformers. For production use, consider established libraries like transformers.js or TensorFlow.js.
