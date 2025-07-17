"""
DumbGPT TUI Application

A rich terminal interface for training and interacting with GPT models.
Built with Textual for an interactive learning experience.
"""

import os
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Button,
    Header,
    Footer,
    Static,
    Label,
    Input,
    TextArea,
    DataTable,
    TabbedContent,
    TabPane,
    Select,
    Checkbox,
    Log,
)
from textual.screen import Screen
from textual.binding import Binding
from textual import on

from ..model.transformer import GPTModel
from ..tokenizer.tokenizer import CharTokenizer
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
            Static(
                """
┌─────────────────────────────────────────────────────────────────┐
│                        🧠 DumbGPT v1.0                         │
│                                                                 │
│         A GPT Implementation from Scratch for Learning         │
│                                                                 │
│  📚 Built with pure Python & NumPy for educational purposes    │
│  🎯 Features: Text Generation, Model Management, Interaction   │
│  💻 Rich terminal interface for model usage                    │
│                                                                 │
│              Press ENTER to continue or Q to quit              │
└─────────────────────────────────────────────────────────────────┘
            """,
                id="welcome-text",
            ),
            id="welcome-container",
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
        Binding("ctrl+g", "toggle_generation", "Generation"),
        Binding("ctrl+m", "toggle_models", "Models"),
        Binding("ctrl+s", "toggle_settings", "Settings"),
        Binding("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()

        with TabbedContent(initial="generation"):
            with TabPane("Generation", id="generation"):
                yield GenerationPanel()

            with TabPane("Models", id="models"):
                yield ModelsPanel()

            with TabPane("Settings", id="settings"):
                yield SettingsPanel()

        yield Footer()

    def action_toggle_generation(self) -> None:
        """Switch to generation tab."""
        self.query_one(TabbedContent).active = "generation"

    def action_toggle_models(self) -> None:
        """Switch to models tab."""
        self.query_one(TabbedContent).active = "models"

    def action_toggle_settings(self) -> None:
        """Switch to settings tab."""
        self.query_one(TabbedContent).active = "settings"

    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()



class GenerationPanel(Container):
    """Text generation interface."""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("✨ Text Generation", classes="panel-title")

            with Horizontal():
                yield Select(
                    [("No models available", None)],
                    id="model-select",
                    allow_blank=False,
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
            self.query_one("#generation-log").write_line(
                f"✅ Model loaded from {model_path}"
            )
        except Exception as e:
            self.query_one("#generation-log").write_line(f"❌ Load error: {str(e)}")

    @on(Button.Pressed, "#generate-btn")
    def generate_text(self) -> None:
        """Generate text from the loaded model."""
        if not hasattr(self, "current_model") or self.current_model is None:
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
                temperature=temperature,
            )

            # Decode generated text
            generated_text = tokenizer.decode(generated_tokens.tolist())

            # Display results
            full_text = context_text + generated_text
            self.query_one("#generated-text").text = full_text
            self.query_one("#generation-log").write_line(
                f"✅ Generated {len(generated_tokens)} tokens"
            )

        except Exception as e:
            self.query_one("#generation-log").write_line(
                f"❌ Generation error: {str(e)}"
            )

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
        self.query_one("#generation-log").write_line(
            f"📁 Found {len(model_files)} model(s)"
        )


class ModelsPanel(Container):
    """Model management interface."""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("📦 Model Management", classes="panel-title")

            with Horizontal():
                yield Button(
                    "Refresh List", id="refresh-models-list", variant="primary"
                )
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
                self.query_one("#management-log").write_line(
                    f"❌ Model {model_name} not found"
                )
        except Exception as e:
            self.query_one("#management-log").write_line(f"❌ Delete error: {str(e)}")

    @on(DataTable.RowSelected)
    def show_model_details(self, event: DataTable.RowSelected) -> None:
        """Show details of selected model."""
        model_name = event.data_table.get_row_at(event.cursor_row)[0]
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
            self.query_one(
                "#model-details"
            ).text = f"Error loading model details: {str(e)}"

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

            table.add_row(model_file.name, f"{size_mb:.2f} MB", str(modified))

        self.query_one("#management-log").write_line(
            f"📊 Listed {len(model_files)} model(s)"
        )


class SettingsPanel(Container):
    """Settings and configuration interface."""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("⚙️ Settings", classes="panel-title")

            with Vertical():
                yield Label("Generation Defaults:")
                yield Input(
                    placeholder="Default Max Generation Length",
                    id="default-max-length",
                    value="50",
                )
                yield Input(
                    placeholder="Default Temperature",
                    id="default-temperature",
                    value="1.0",
                )
                yield Input(
                    placeholder="Default Context Length",
                    id="default-context-length",
                    value="20",
                )

                yield Label("Paths:")
                yield Input(
                    placeholder="Models Directory",
                    id="models-path",
                    value="models/",
                )

                yield Label("Interface Preferences:")
                yield Checkbox(
                    "Auto-refresh model list", id="auto-refresh", value=True
                )
                yield Checkbox(
                    "Show generation timestamps", id="show-timestamps", value=False
                )
                yield Checkbox(
                    "Auto-scroll generation log", id="auto-scroll", value=True
                )

                with Horizontal():
                    yield Button("Save Settings", id="save-settings", variant="success")
                    yield Button(
                        "Reset to Defaults", id="reset-settings", variant="warning"
                    )

            yield Static("Settings Log", classes="section-title")
            yield Log(id="settings-log")

    @on(Button.Pressed, "#save-settings")
    def save_settings(self) -> None:
        """Save current settings."""
        self.query_one("#settings-log").write_line("💾 Settings saved")

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
