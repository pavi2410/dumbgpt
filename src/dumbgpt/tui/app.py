"""
DumbGPT TUI Application

A rich terminal interface for training and interacting with GPT models.
Built with Textual for an interactive learning experience.
"""

import os
import asyncio
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Button, Header, Footer, Static, Label, Input, 
    ProgressBar, TextArea, DataTable, TabbedContent, 
    TabPane, Select, Checkbox, Log
)
from textual.screen import Screen
from textual.binding import Binding
from textual.reactive import reactive
from textual import on

from ..model.transformer import GPTModel
from ..tokenizer.tokenizer import CharTokenizer
from ..training.dataloader import DataLoader
from ..training.optimizer import SGD, Adam
from ..training.trainer import Trainer
from ..training.utils import save_model, load_model


class WelcomeScreen(Screen):
    """Welcome screen with project introduction."""
    
    BINDINGS = [
        Binding("enter", "continue", "Continue"),
        Binding("q", "quit", "Quit"),
    ]
    
    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("""
┌─────────────────────────────────────────────────────────────────┐
│                        🧠 DumbGPT v1.0                         │
│                                                                 │
│         A GPT Implementation from Scratch for Learning         │
│                                                                 │
│  📚 Built with pure Python & NumPy for educational purposes    │
│  🎯 Features: Tokenization, Attention, Training, Generation    │
│  💻 Rich terminal interface with Textual                       │
│                                                                 │
│              Press ENTER to continue or Q to quit              │
└─────────────────────────────────────────────────────────────────┘
            """, id="welcome-text"),
            id="welcome-container"
        )
        yield Footer()
    
    def action_continue(self) -> None:
        """Continue to main application."""
        self.app.switch_mode("main")
    
    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()


class MainScreen(Screen):
    """Main application screen with tabbed interface."""
    
    BINDINGS = [
        Binding("ctrl+t", "toggle_training", "Training"),
        Binding("ctrl+g", "toggle_generation", "Generation"),
        Binding("ctrl+m", "toggle_models", "Models"),
        Binding("q", "quit", "Quit"),
    ]
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with TabbedContent(initial="training"):
            with TabPane("Training", id="training"):
                yield TrainingPanel()
            
            with TabPane("Generation", id="generation"):
                yield GenerationPanel()
            
            with TabPane("Models", id="models"):
                yield ModelsPanel()
            
            with TabPane("Settings", id="settings"):
                yield SettingsPanel()
        
        yield Footer()
    
    def action_toggle_training(self) -> None:
        """Switch to training tab."""
        self.query_one(TabbedContent).active = "training"
    
    def action_toggle_generation(self) -> None:
        """Switch to generation tab."""
        self.query_one(TabbedContent).active = "generation"
    
    def action_toggle_models(self) -> None:
        """Switch to models tab."""
        self.query_one(TabbedContent).active = "models"
    
    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()


class TrainingPanel(Container):
    """Training interface with progress tracking."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainer: Optional[Trainer] = None
        self.is_training = reactive(False)
    
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("🏋️ Model Training", classes="panel-title")
            
            with Horizontal():
                with Vertical():
                    yield Label("Model Configuration")
                    yield Input(placeholder="Vocabulary Size", id="vocab-size", value="1000")
                    yield Input(placeholder="Model Dimension", id="d-model", value="128")
                    yield Input(placeholder="Number of Heads", id="num-heads", value="8")
                    yield Input(placeholder="Feed Forward Dim", id="d-ff", value="512")
                    yield Input(placeholder="Number of Layers", id="num-layers", value="6")
                    yield Input(placeholder="Max Sequence Length", id="max-seq-len", value="256")
                
                with Vertical():
                    yield Label("Training Configuration")
                    yield Input(placeholder="Learning Rate", id="learning-rate", value="0.001")
                    yield Input(placeholder="Batch Size", id="batch-size", value="32")
                    yield Input(placeholder="Training Steps", id="training-steps", value="1000")
                    yield Select(
                        [("SGD", "sgd"), ("Adam", "adam")],
                        value="adam",
                        id="optimizer-select"
                    )
                    yield Input(placeholder="Corpus Path", id="corpus-path", value="corpus/")
            
            with Horizontal():
                yield Button("Start Training", id="start-training", variant="primary")
                yield Button("Stop Training", id="stop-training", variant="error", disabled=True)
                yield Button("Save Model", id="save-model", disabled=True)
            
            yield Static("Training Progress", classes="section-title")
            yield ProgressBar(total=100, id="training-progress")
            yield Static("Step: 0 | Loss: 0.0000 | Perplexity: 0.0", id="training-stats")
            
            yield Static("Training Log", classes="section-title")
            yield Log(id="training-log")
    
    @on(Button.Pressed, "#start-training")
    def start_training(self) -> None:
        """Start model training."""
        if self.is_training:
            return
        
        try:
            # Get configuration values
            vocab_size = int(self.query_one("#vocab-size").value or "1000")
            d_model = int(self.query_one("#d-model").value or "128")
            num_heads = int(self.query_one("#num-heads").value or "8")
            d_ff = int(self.query_one("#d-ff").value or "512")
            num_layers = int(self.query_one("#num-layers").value or "6")
            max_seq_len = int(self.query_one("#max-seq-len").value or "256")
            
            learning_rate = float(self.query_one("#learning-rate").value or "0.001")
            batch_size = int(self.query_one("#batch-size").value or "32")
            training_steps = int(self.query_one("#training-steps").value or "1000")
            optimizer_type = self.query_one("#optimizer-select").value
            corpus_path = self.query_one("#corpus-path").value or "corpus/"
            
            # Initialize training
            self._initialize_training(
                vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len,
                learning_rate, batch_size, training_steps, optimizer_type, corpus_path
            )
            
            # Start training in background
            self.is_training = True
            self.query_one("#start-training").disabled = True
            self.query_one("#stop-training").disabled = False
            
            self.query_one("#training-log").write_line("🚀 Starting training...")
            
            # Run training asynchronously
            asyncio.create_task(self._run_training(training_steps))
            
        except Exception as e:
            self.query_one("#training-log").write_line(f"❌ Error: {str(e)}")
    
    @on(Button.Pressed, "#stop-training")
    def stop_training(self) -> None:
        """Stop model training."""
        self.is_training = False
        self.query_one("#start-training").disabled = False
        self.query_one("#stop-training").disabled = True
        self.query_one("#save-model").disabled = False
        self.query_one("#training-log").write_line("🛑 Training stopped")
    
    @on(Button.Pressed, "#save-model")
    def save_model_action(self) -> None:
        """Save the trained model."""
        if self.trainer is None:
            self.query_one("#training-log").write_line("❌ No model to save")
            return
        
        try:
            os.makedirs("models", exist_ok=True)
            model_path = f"models/model_{len(os.listdir('models')) + 1}.pkl"
            save_model(self.trainer.model, model_path)
            self.query_one("#training-log").write_line(f"💾 Model saved to {model_path}")
        except Exception as e:
            self.query_one("#training-log").write_line(f"❌ Save error: {str(e)}")
    
    def _initialize_training(self, vocab_size, d_model, num_heads, d_ff, num_layers, 
                            max_seq_len, learning_rate, batch_size, training_steps, 
                            optimizer_type, corpus_path):
        """Initialize training components."""
        # Create tokenizer and load corpus
        tokenizer = CharTokenizer()
        corpus_files = []
        
        if os.path.exists(corpus_path):
            for file in Path(corpus_path).rglob("*.txt"):
                corpus_files.append(str(file))
        
        if not corpus_files:
            # Use sample text if no corpus files found
            sample_texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is a subset of artificial intelligence.",
                "Neural networks are inspired by the human brain."
            ]
            tokenizer.build_vocab(sample_texts)
            
            dataloader = DataLoader(
                corpus_paths=None,
                tokenizer=tokenizer,
                seq_length=min(64, max_seq_len),
                batch_size=batch_size,
                sample_texts=sample_texts
            )
        else:
            # Load actual corpus files
            corpus_texts = []
            for file_path in corpus_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    corpus_texts.append(f.read())
            
            tokenizer.build_vocab(corpus_texts)
            
            dataloader = DataLoader(
                corpus_paths=corpus_files,
                tokenizer=tokenizer,
                seq_length=min(64, max_seq_len),
                batch_size=batch_size
            )
        
        # Create model with actual vocabulary size
        model = GPTModel(
            vocab_size=tokenizer.vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            max_seq_len=max_seq_len
        )
        
        # Create optimizer
        if optimizer_type == "sgd":
            optimizer = SGD(learning_rate=learning_rate)
        else:
            optimizer = Adam(learning_rate=learning_rate)
        
        # Create trainer
        self.trainer = Trainer(model, dataloader, optimizer)
        
        self.query_one("#training-log").write_line(f"✅ Model initialized with vocab_size={tokenizer.vocab_size}")
    
    async def _run_training(self, training_steps):
        """Run training loop with UI updates."""
        if self.trainer is None:
            return
        
        progress_bar = self.query_one("#training-progress")
        stats_display = self.query_one("#training-stats")
        log_display = self.query_one("#training-log")
        
        progress_bar.total = training_steps
        
        for step in range(training_steps):
            if not self.is_training:
                break
            
            try:
                # Training step
                loss = self.trainer.train_step()
                
                # Update progress
                progress_bar.advance(1)
                
                # Update stats every 10 steps
                if step % 10 == 0:
                    perplexity = self.trainer.compute_perplexity(num_batches=5)
                    stats_display.update(f"Step: {step} | Loss: {loss:.4f} | Perplexity: {perplexity:.2f}")
                    log_display.write_line(f"Step {step}: Loss={loss:.4f}, Perplexity={perplexity:.2f}")
                
                # Small delay to prevent UI blocking
                await asyncio.sleep(0.01)
                
            except Exception as e:
                log_display.write_line(f"❌ Training error at step {step}: {str(e)}")
                break
        
        # Training completed
        self.is_training = False
        self.query_one("#start-training").disabled = False
        self.query_one("#stop-training").disabled = True
        self.query_one("#save-model").disabled = False
        log_display.write_line("🎉 Training completed!")


class GenerationPanel(Container):
    """Text generation interface."""
    
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("✨ Text Generation", classes="panel-title")
            
            with Horizontal():
                yield Select(
                    [("No models available", None)],
                    id="model-select",
                    allow_blank=False
                )
                yield Button("Load Model", id="load-model", variant="primary")
                yield Button("Refresh Models", id="refresh-models")
            
            yield Label("Input Text (context):")
            yield Input(placeholder="Enter context text...", id="context-input")
            
            with Horizontal():
                yield Input(placeholder="Max Length", id="max-length", value="50")
                yield Input(placeholder="Temperature", id="temperature", value="1.0")
                yield Button("Generate", id="generate-btn", variant="success")
            
            yield Static("Generated Text", classes="section-title")
            yield TextArea(id="generated-text", read_only=True)
            
            yield Static("Generation Log", classes="section-title")
            yield Log(id="generation-log")
    
    def on_mount(self) -> None:
        """Refresh models when panel mounts."""
        self._refresh_models()
    
    @on(Button.Pressed, "#refresh-models")
    def refresh_models(self) -> None:
        """Refresh the models list."""
        self._refresh_models()
    
    @on(Button.Pressed, "#load-model")
    def load_model_action(self) -> None:
        """Load selected model."""
        model_select = self.query_one("#model-select")
        if model_select.value is None:
            self.query_one("#generation-log").write_line("❌ No model selected")
            return
        
        try:
            model_path = model_select.value
            self.current_model = load_model(model_path)
            self.query_one("#generation-log").write_line(f"✅ Model loaded from {model_path}")
        except Exception as e:
            self.query_one("#generation-log").write_line(f"❌ Load error: {str(e)}")
    
    @on(Button.Pressed, "#generate-btn")
    def generate_text(self) -> None:
        """Generate text from the loaded model."""
        if not hasattr(self, 'current_model') or self.current_model is None:
            self.query_one("#generation-log").write_line("❌ No model loaded")
            return
        
        context_text = self.query_one("#context-input").value
        if not context_text:
            self.query_one("#generation-log").write_line("❌ Please enter context text")
            return
        
        try:
            max_length = int(self.query_one("#max-length").value or "50")
            temperature = float(self.query_one("#temperature").value or "1.0")
            
            # Create a simple tokenizer for demonstration
            tokenizer = CharTokenizer()
            tokenizer.build_vocab([context_text])
            
            # Encode context
            context_tokens = tokenizer.encode(context_text)
            
            # Generate text
            generated_tokens = self.current_model.generate(
                context=context_tokens[-20:],  # Use last 20 tokens as context
                max_length=max_length,
                temperature=temperature
            )
            
            # Decode generated text
            generated_text = tokenizer.decode(generated_tokens.tolist())
            
            # Display results
            full_text = context_text + generated_text
            self.query_one("#generated-text").text = full_text
            self.query_one("#generation-log").write_line(f"✅ Generated {len(generated_tokens)} tokens")
            
        except Exception as e:
            self.query_one("#generation-log").write_line(f"❌ Generation error: {str(e)}")
    
    def _refresh_models(self) -> None:
        """Refresh the models dropdown."""
        models_dir = Path("models")
        if not models_dir.exists():
            models_dir.mkdir(exist_ok=True)
        
        model_files = list(models_dir.glob("*.pkl"))
        
        if model_files:
            options = [(f.name, str(f)) for f in model_files]
        else:
            options = [("No models available", None)]
        
        self.query_one("#model-select").set_options(options)
        self.query_one("#generation-log").write_line(f"📁 Found {len(model_files)} model(s)")


class ModelsPanel(Container):
    """Model management interface."""
    
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("📦 Model Management", classes="panel-title")
            
            with Horizontal():
                yield Button("Refresh List", id="refresh-models-list", variant="primary")
                yield Button("Delete Selected", id="delete-model", variant="error")
            
            yield DataTable(id="models-table")
            
            yield Static("Model Details", classes="section-title")
            yield TextArea(id="model-details", read_only=True)
            
            yield Static("Management Log", classes="section-title")
            yield Log(id="management-log")
    
    def on_mount(self) -> None:
        """Initialize the models table."""
        self._setup_table()
        self._refresh_models_list()
    
    def _setup_table(self) -> None:
        """Setup the models table."""
        table = self.query_one("#models-table")
        table.add_columns("Name", "Size", "Modified")
        table.cursor_type = "row"
    
    @on(Button.Pressed, "#refresh-models-list")
    def refresh_models_list(self) -> None:
        """Refresh the models list."""
        self._refresh_models_list()
    
    @on(Button.Pressed, "#delete-model")
    def delete_model(self) -> None:
        """Delete selected model."""
        table = self.query_one("#models-table")
        if table.cursor_row is None:
            self.query_one("#management-log").write_line("❌ No model selected")
            return
        
        row_data = table.get_row_at(table.cursor_row)
        model_name = row_data[0]
        model_path = Path("models") / model_name
        
        try:
            if model_path.exists():
                model_path.unlink()
                self.query_one("#management-log").write_line(f"🗑️ Deleted {model_name}")
                self._refresh_models_list()
            else:
                self.query_one("#management-log").write_line(f"❌ Model {model_name} not found")
        except Exception as e:
            self.query_one("#management-log").write_line(f"❌ Delete error: {str(e)}")
    
    @on(DataTable.RowSelected)
    def show_model_details(self, event: DataTable.RowSelected) -> None:
        """Show details of selected model."""
        model_name = event.data_table.get_row_at(event.row_index)[0]
        model_path = Path("models") / model_name
        
        if not model_path.exists():
            return
        
        try:
            # Get file stats
            stat = model_path.stat()
            size_mb = stat.st_size / (1024 * 1024)
            
            # Load model to get configuration
            model = load_model(str(model_path))
            
            details = f"""Model: {model_name}
Size: {size_mb:.2f} MB
Vocabulary Size: {model.vocab_size}
Model Dimension: {model.d_model}
Number of Heads: {model.num_heads}
Feed Forward Dimension: {model.d_ff}
Number of Layers: {model.num_layers}
Max Sequence Length: {model.max_seq_len}"""
            
            self.query_one("#model-details").text = details
            
        except Exception as e:
            self.query_one("#model-details").text = f"Error loading model details: {str(e)}"
    
    def _refresh_models_list(self) -> None:
        """Refresh the models list in the table."""
        models_dir = Path("models")
        if not models_dir.exists():
            models_dir.mkdir(exist_ok=True)
        
        table = self.query_one("#models-table")
        table.clear()
        
        model_files = list(models_dir.glob("*.pkl"))
        
        for model_file in model_files:
            stat = model_file.stat()
            size_mb = stat.st_size / (1024 * 1024)
            modified = stat.st_mtime
            
            table.add_row(
                model_file.name,
                f"{size_mb:.2f} MB",
                str(modified)
            )
        
        self.query_one("#management-log").write_line(f"📊 Listed {len(model_files)} model(s)")


class SettingsPanel(Container):
    """Settings and configuration interface."""
    
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("⚙️ Settings", classes="panel-title")
            
            with Vertical():
                yield Label("Default Model Configuration:")
                yield Input(placeholder="Default Vocabulary Size", id="default-vocab-size", value="1000")
                yield Input(placeholder="Default Model Dimension", id="default-d-model", value="128")
                yield Input(placeholder="Default Number of Heads", id="default-num-heads", value="8")
                yield Input(placeholder="Default Feed Forward Dim", id="default-d-ff", value="512")
                yield Input(placeholder="Default Number of Layers", id="default-num-layers", value="6")
                yield Input(placeholder="Default Max Sequence Length", id="default-max-seq-len", value="256")
                
                yield Label("Training Defaults:")
                yield Input(placeholder="Default Learning Rate", id="default-lr", value="0.001")
                yield Input(placeholder="Default Batch Size", id="default-batch-size", value="32")
                yield Select(
                    [("SGD", "sgd"), ("Adam", "adam")],
                    value="adam",
                    id="default-optimizer"
                )
                
                yield Label("Paths:")
                yield Input(placeholder="Default Corpus Path", id="default-corpus-path", value="corpus/")
                yield Input(placeholder="Default Models Path", id="default-models-path", value="models/")
                
                yield Checkbox("Enable training progress sounds", id="enable-sounds", value=False)
                yield Checkbox("Auto-save models after training", id="auto-save", value=True)
                
                with Horizontal():
                    yield Button("Save Settings", id="save-settings", variant="success")
                    yield Button("Reset to Defaults", id="reset-settings", variant="warning")
            
            yield Static("Settings Log", classes="section-title")
            yield Log(id="settings-log")
    
    @on(Button.Pressed, "#save-settings")
    def save_settings(self) -> None:
        """Save current settings."""
        self.query_one("#settings-log").write_line("💾 Settings saved (placeholder)")
    
    @on(Button.Pressed, "#reset-settings")
    def reset_settings(self) -> None:
        """Reset settings to defaults."""
        self.query_one("#settings-log").write_line("🔄 Settings reset to defaults")


class DumbGPTApp(App):
    """Main DumbGPT TUI Application."""
    
    TITLE = "DumbGPT - GPT Implementation from Scratch"
    SUB_TITLE = "Learn transformers by building them!"
    
    CSS_PATH = "app.css"
    
    MODES = {
        "welcome": WelcomeScreen,
        "main": MainScreen,
    }
    
    def on_mount(self) -> None:
        """Initialize the application."""
        self.switch_mode("welcome")


def main():
    """Main entry point for the TUI application."""
    app = DumbGPTApp()
    app.run()


if __name__ == "__main__":
    main()