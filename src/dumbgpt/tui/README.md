# DumbGPT TUI Interface

A rich terminal user interface for interacting with trained GPT models, built with Textual.

## Features

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
- **Generation Defaults**: Set default values for text generation parameters
- **Path Configuration**: Customize models directory
- **Interface Options**: Enable/disable UI features like auto-refresh and timestamps

## Usage

Run the TUI application:

```bash
uv run main.py
```

### Navigation
- **Tab Navigation**: Use `Ctrl+G` (Generation), `Ctrl+M` (Models), `Ctrl+S` (Settings)
- **Quit**: Press `Q` to exit the application
- **Enter**: Continue from welcome screen

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
- **MainScreen**: Tabbed interface with three main panels
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
- **Model Interaction**: Hands-on experience with trained GPT models
- **Text Generation**: Interactive exploration of model capabilities
- **Model Management**: Practical model versioning and storage
- **Terminal UI Design**: Modern console application development

The interface makes trained transformer models accessible and interactive, perfect for experimentation and understanding model behavior. For training models, use Python scripts with the training modules directly.