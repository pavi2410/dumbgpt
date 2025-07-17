# DumbGPT TUI Interface

A rich terminal user interface for training and interacting with GPT models, built with Textual.

## Features

### 🏋️ Training Interface
- **Model Configuration**: Set vocabulary size, model dimensions, attention heads, layers
- **Training Parameters**: Configure learning rate, batch size, training steps, optimizer
- **Real-time Progress**: Live training progress bar with loss and perplexity tracking
- **Corpus Loading**: Automatically loads training data from `corpus/` directory
- **Model Saving**: Save trained models to `models/` directory

### ✨ Text Generation
- **Model Loading**: Load and switch between saved models
- **Interactive Generation**: Generate text with customizable length and temperature
- **Context Input**: Provide context for conditional text generation
- **Real-time Preview**: See generated text immediately

### 📦 Model Management
- **Model Browser**: View all saved models with file details
- **Model Details**: Inspect model architecture and parameters
- **Delete Models**: Remove unwanted models from storage
- **Model Statistics**: View file size and modification dates

### ⚙️ Settings & Configuration
- **Default Parameters**: Set default values for model and training configuration
- **Path Configuration**: Customize corpus and models directories
- **Training Options**: Enable/disable features like auto-save and progress sounds

## Usage

Run the TUI application:

```bash
uv run main.py
```

### Navigation
- **Tab Navigation**: Use `Ctrl+T` (Training), `Ctrl+G` (Generation), `Ctrl+M` (Models)
- **Quit**: Press `Q` to exit the application
- **Enter**: Continue from welcome screen

### Training a Model

1. **Configure Model**: Set vocabulary size, dimensions, layers in Training tab
2. **Set Training Parameters**: Choose learning rate, batch size, steps, optimizer
3. **Specify Corpus**: Point to your training data directory (default: `corpus/`)
4. **Start Training**: Click "Start Training" to begin
5. **Monitor Progress**: Watch real-time loss and perplexity metrics
6. **Save Model**: Click "Save Model" when training completes

### Generating Text

1. **Load Model**: Select a trained model from the dropdown
2. **Enter Context**: Provide initial text for generation
3. **Set Parameters**: Adjust max length and temperature
4. **Generate**: Click "Generate" to produce text
5. **View Results**: See generated text in the output area

### Managing Models

1. **Browse Models**: View all saved models in the table
2. **View Details**: Click on a model to see its configuration
3. **Delete Models**: Select and delete unwanted models
4. **Refresh List**: Update the models list

## Architecture

The TUI is built using Textual and consists of several key components:

- **WelcomeScreen**: Initial greeting and project introduction
- **MainScreen**: Tabbed interface with four main panels
- **TrainingPanel**: Model training with progress tracking
- **GenerationPanel**: Text generation interface
- **ModelsPanel**: Model management and browsing
- **SettingsPanel**: Configuration and preferences

## File Structure

```
src/dumbgpt/tui/
├── app.py          # Main TUI application
├── app.css         # Styling and layout
└── README.md       # This documentation
```

## Dependencies

The TUI requires the following packages:
- `textual` - Rich terminal UI framework
- `asyncio` - Asynchronous programming support
- All DumbGPT core modules (model, training, tokenizer)

## Educational Value

This TUI demonstrates:
- **Interactive Learning**: Hands-on experience with GPT training
- **Real-time Feedback**: Live metrics and progress tracking
- **Model Management**: Practical model versioning and storage
- **Text Generation**: Interactive exploration of model capabilities
- **Terminal UI Design**: Modern console application development

The interface makes the complex process of transformer training accessible and visual, perfect for educational purposes and experimentation.