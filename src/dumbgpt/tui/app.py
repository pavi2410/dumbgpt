"""
DumbGPT TUI – Chat interface
"""

from pathlib import Path

import tiktoken
import torch
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button, Footer, Header, Input, Label, Log, Select,
)

from ..model.transformer import GPTModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
MODELS_DIR = Path("models")


def list_saved_models() -> list[tuple[str, str]]:
    """Return list of (label, path_str) for Select widget."""
    if not MODELS_DIR.exists():
        return []
    files = sorted(MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [(p.name, str(p)) for p in files]


def load_model_from_path(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    tok = tiktoken.get_encoding("gpt2")
    model = GPTModel(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        num_layers=cfg["num_layers"],
        max_seq_len=cfg["max_seq_len"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    return model, tok, cfg, params


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Chat tab
# ---------------------------------------------------------------------------
class ChatTab(Vertical):
    DEFAULT_CSS = """
    ChatTab { height: 1fr; padding: 1 2; }
    #model-bar { height: 3; margin-bottom: 1; }
    #model-select { width: 1fr; }
    #reload-btn { width: 14; }
    #chat-log { height: 1fr; border: solid $primary; }
    #gen-bar { height: 3; margin-top: 1; }
    #prompt-input { width: 1fr; }
    #gen-btn { width: 12; }
    #params-bar { height: 3; margin-top: 1; }
    #temp-input { width: 20; }
    #topk-input { width: 20; }
    #len-input  { width: 20; }
    #status-label { margin-top: 1; color: $text-muted; }
    """

    def compose(self) -> ComposeResult:
        models = list_saved_models()
        options = models if models else [("(no models found – train first)", "")]

        with Horizontal(id="model-bar"):
            yield Select(options, prompt="Select a model…", id="model-select")
            yield Button("Reload", id="reload-btn", variant="default")

        yield Log(id="chat-log", highlight=True)

        with Horizontal(id="params-bar"):
            yield Input(value="0.8",  placeholder="temperature", id="temp-input")
            yield Input(value="50",   placeholder="top-k",       id="topk-input")
            yield Input(value="100",  placeholder="max new tokens", id="len-input")

        with Horizontal(id="gen-bar"):
            yield Input(placeholder="Enter prompt and press Enter…", id="prompt-input")
            yield Button("Generate", id="gen-btn", variant="primary")

        yield Label("No model loaded.", id="status-label")

    # ---- model loading ----

    @on(Button.Pressed, "#reload-btn")
    def refresh_models(self):
        sel = self.query_one("#model-select", Select)
        models = list_saved_models()
        options = models if models else [("(no models found – train first)", "")]
        sel.set_options(options)

    @on(Select.Changed, "#model-select")
    def model_selected(self, event: Select.Changed):
        path = event.value
        if not path:
            return
        log = self.query_one("#chat-log", Log)
        status = self.query_one("#status-label", Label)
        log.write_line(f"Loading {Path(path).name}…")
        status.update("Loading…")
        self._load_model_async(path)

    @work(thread=True)
    def _load_model_async(self, path: str):
        log = self.query_one("#chat-log", Log)
        status = self.query_one("#status-label", Label)
        try:
            device = get_device()
            model, tok, cfg, params = load_model_from_path(path, device)
            self.app.model = model
            self.app.tokenizer = tok
            self.app.device = device
            info = (f"Loaded {Path(path).name}  |  "
                    f"{params:,} params  |  device={device}  |  "
                    f"d_model={cfg['d_model']}  layers={cfg['num_layers']}  "
                    f"seq={cfg['max_seq_len']}")
            self.app.call_from_thread(log.write_line, info)
            self.app.call_from_thread(status.update, f"Model: {Path(path).name} ({params:,} params)")
        except Exception as exc:
            self.app.call_from_thread(log.write_line, f"Error loading model: {exc}")
            self.app.call_from_thread(status.update, "Load failed.")

    # ---- generation ----

    @on(Button.Pressed, "#gen-btn")
    @on(Input.Submitted, "#prompt-input")
    def generate(self):
        prompt = self.query_one("#prompt-input", Input).value.strip()
        if not prompt:
            return
        if not self.app.model:
            self.query_one("#chat-log", Log).write_line(
                "No model loaded. Select one above.")
            return

        self.query_one("#prompt-input", Input).value = ""

        try:
            temperature = float(self.query_one("#temp-input", Input).value or "0.8")
            top_k       = int(self.query_one("#topk-input",  Input).value or "50")
            max_new     = int(self.query_one("#len-input",   Input).value or "100")
        except ValueError:
            temperature, top_k, max_new = 0.8, 50, 100

        log = self.query_one("#chat-log", Log)
        log.write_line(f"You: {prompt}")
        log.write_line("Generating…")
        self._generate_async(prompt, temperature, top_k, max_new)

    @work(thread=True)
    def _generate_async(self, prompt: str, temperature: float, top_k: int, max_new: int):
        log = self.query_one("#chat-log", Log)
        try:
            tok = self.app.tokenizer
            model = self.app.model
            device = self.app.device
            input_ids = tok.encode(prompt)
            ctx = torch.tensor([input_ids], dtype=torch.long)
            out = model.generate(ctx, max_new_tokens=max_new, temperature=temperature, top_k=top_k)
            new_tokens = out[0, len(input_ids):].tolist()
            generated_text = tok.decode(new_tokens)
            self.app.call_from_thread(log.write_line, f"Model: {generated_text}")
        except Exception as exc:
            self.app.call_from_thread(log.write_line, f"Generation error: {exc}")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
class DumbGPTApp(App):
    """DumbGPT Terminal Interface."""

    TITLE = "DumbGPT"
    CSS = """
    Screen { background: $surface; }
    """
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    model = None
    tokenizer = None
    device = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield ChatTab()
        yield Footer()


def main():
    app = DumbGPTApp()
    app.run()


if __name__ == "__main__":
    main()