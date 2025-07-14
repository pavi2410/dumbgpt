# dumbgpt

A Markov chain-based text generator that can learn from novels or code to generate new text.

## Installation

```bash
bun install
```

## Usage

```bash
bun run src/index.ts
```

### Switching Models

Edit `src/index.ts` to change the model type:

```typescript
const MODEL_TYPE = 'code'; // 'text' or 'code'
```

- **`text`** - Trains on novels in `./corpus/novels/*.txt`
- **`code`** - Trains on Java files in `./corpus/code/*.java`

### Interactive Commands

Once running, you can:
- Type any text to generate completions
- Type `/q` to quit

### Configuration

Modify these constants in `src/index.ts`:
- `CONTEXT_SIZE` - How many previous words to consider (default: 4)
- `MAX_OUTPUT_TOKENS` - Maximum tokens to generate (default: 100)

## Project Structure

```
src/
├── core/
│   ├── markov.ts       # Core Markov chain algorithm
│   ├── generator.ts    # Text generation logic
│   └── cli.ts          # CLI interface
├── tokenizers/
│   ├── text.ts         # Text tokenization for novels
│   └── code.ts         # Code tokenization for Java
├── text-model.ts       # Text model implementation
├── code-model.ts       # Code model implementation
└── index.ts           # Entry point
```

## Corpus Data

The project expects training data in the following structure:

```
corpus/
├── novels/          # Text files for novel training
│   ├── *.txt       # Any .txt files
└── code/           # Java files for code training
    ├── *.java      # Any .java files
```

### Setting up your corpus:

1. **For text model**: Place any `.txt` files in `corpus/novels/`
2. **For code model**: Place any `.java` files in `corpus/code/`

The corpus directory is gitignored to avoid committing large training files.

## How It Works

1. **Tokenization** - Splits text into meaningful tokens (words, punctuation, etc.)
2. **Training** - Builds a Markov chain from the corpus, learning word transition probabilities
3. **Generation** - Uses the trained model to predict the next most likely word given context
4. **Output** - Colorizes tokens and formats the generated text

This project was created using `bun init` in bun v1.1.26. [Bun](https://bun.sh) is a fast all-in-one JavaScript runtime.
