"""
DumbGPT TUI – Chat + Training launcher
"""

import subprocess
import sys
from pathlib import Path

import torch
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button, Footer, Header, Input, Label, Log,
    Select, TabbedContent, TabPane,
)

from ..model.transformer import GPTModel
from ..tokenizer.tiktoken_tokenizer import TikTokenTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
MODELS_DIR = Path("models")

PRESETS = {
    "nano":   dict(d_model=128, num_heads=4, d_ff=256,  num_layers=3, max_seq_len=128),
    "small":  dict(d_model=256, num_heads=4, d_ff=512,  num_layers=4, max_seq_len=128),
    "medium": dict(d_model=384, num_heads=6, d_ff=768,  num_layers=6, max_seq_len=256),
}


def list_saved_models() -> list[tuple[str, str]]:
    """Return list of (label, path_str) for Select widget."""
    if not MODELS_DIR.exists():
        return []
    files = sorted(MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [(p.name, str(p)) for p in files]


def load_model_from_path(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    tok = TikTokenTokenizer()
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

        yield Log(id="chat-log", highlight=True, markup=True)

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
        log.write_line(f"[bold]Loading {Path(path).name}…[/bold]")
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
            info = (f"[green]Loaded[/green] {Path(path).name}  |  "
                    f"{params:,} params  |  device={device}  |  "
                    f"d_model={cfg['d_model']}  layers={cfg['num_layers']}  "
                    f"seq={cfg['max_seq_len']}")
            self.call_from_thread(log.write_line, info)
            self.call_from_thread(status.update, f"Model: {Path(path).name} ({params:,} params)")
        except Exception as exc:
            self.call_from_thread(log.write_line, f"[red]Error loading model: {exc}[/red]")
            self.call_from_thread(status.update, "Load failed.")

    # ---- generation ----

    @on(Button.Pressed, "#gen-btn")
    @on(Input.Submitted, "#prompt-input")
    def generate(self):
        prompt = self.query_one("#prompt-input", Input).value.strip()
        if not prompt:
            return
        if not self.app.model:
            self.query_one("#chat-log", Log).write_line(
                "[yellow]No model loaded. Select one above.[/yellow]")
            return

        self.query_one("#prompt-input", Input).value = ""

        try:
            temperature = float(self.query_one("#temp-input", Input).value or "0.8")
            top_k       = int(self.query_one("#topk-input",  Input).value or "50")
            max_new     = int(self.query_one("#len-input",   Input).value or "100")
        except ValueError:
            temperature, top_k, max_new = 0.8, 50, 100

        log = self.query_one("#chat-log", Log)
        log.write_line(f"[bold cyan]You:[/bold cyan] {prompt}")
        log.write_line("[dim]Generating…[/dim]")
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
            # Only the newly generated tokens
            new_tokens = out[0, len(input_ids):].tolist()
            generated_text = tok.decode(new_tokens)
            self.call_from_thread(
                log.write_line,
                f"[bold green]Model:[/bold green] {generated_text}"
            )
        except Exception as exc:
            self.call_from_thread(log.write_line, f"[red]Generation error: {exc}[/red]")


# ---------------------------------------------------------------------------
# Train tab
# ---------------------------------------------------------------------------
class TrainTab(Vertical):
    DEFAULT_CSS = """
    TrainTab { height: 1fr; padding: 1 2; }
    #train-options { height: auto; margin-bottom: 1; }
    #preset-select { width: 20; }
    #epochs-input  { width: 12; }
    #batch-input   { width: 12; }
    #steps-input   { width: 12; }
    #lr-input      { width: 16; }
    #train-btn     { width: 16; margin-top: 1; }
    #stop-btn      { width: 16; margin-top: 1; }
    #train-log     { height: 1fr; border: solid $accent; }
    """

    _proc = None

    def compose(self) -> ComposeResult:
        yield Label("[b]Training Configuration[/b]", markup=True)
        with Horizontal(id="train-options"):
            yield Select(
                [(k, k) for k in PRESETS],
                value="small",
                prompt="Preset…",
                id="preset-select",
            )
            yield Input(value="5",    placeholder="epochs",     id="epochs-input")
            yield Input(value="32",   placeholder="batch",      id="batch-input")
            yield Input(value="500",  placeholder="steps/epoch",id="steps-input")
            yield Input(value="3e-4", placeholder="lr",         id="lr-input")

        with Horizontal():
            yield Button("Start Training", id="train-btn", variant="success")
            yield Button("Stop",           id="stop-btn",  variant="error")

        yield Log(id="train-log", highlight=True)

    @on(Button.Pressed, "#train-btn")
    def start_training(self):
        if self._proc and self._proc.poll() is None:
            self.query_one("#train-log", Log).write_line(
                "[yellow]Training already running.[/yellow]")
            return

        preset = self.query_one("#preset-select", Select).value or "small"
        epochs = self.query_one("#epochs-input", Input).value or "5"
        batch  = self.query_one("#batch-input",  Input).value or "32"
        steps  = self.query_one("#steps-input",  Input).value or "500"
        lr     = self.query_one("#lr-input",     Input).value or "3e-4"

        log = self.query_one("#train-log", Log)
        log.write_line(f"[bold]Starting training:[/bold] preset={preset} epochs={epochs} "
                       f"batch={batch} steps={steps} lr={lr}")

        cmd = [
            sys.executable, "train.py",
            "--preset", preset,
            "--epochs", epochs,
            "--batch",  batch,
            "--steps",  steps,
            "--lr",     lr,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._stream_output()

    @work(thread=True)
    def _stream_output(self):
        log = self.query_one("#train-log", Log)
        if not self._proc:
            return
        for line in self._proc.stdout:
            self.call_from_thread(log.write_line, line.rstrip())
        rc = self._proc.wait()
        self.call_from_thread(
            log.write_line,
            f"[bold {'green' if rc == 0 else 'red'}]Training finished (exit {rc})[/bold]"
        )

    @on(Button.Pressed, "#stop-btn")
    def stop_training(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self.query_one("#train-log", Log).write_line("[red]Training stopped.[/red]")
        else:
            self.query_one("#train-log", Log).write_line("[yellow]No training process running.[/yellow]")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
class DumbGPTApp(App):
    """DumbGPT Terminal Interface."""

    TITLE = "DumbGPT"
    CSS = """
    Screen { background: $surface; }
    TabbedContent { height: 1fr; }
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
        with TabbedContent():
            with TabPane("Chat", id="chat-pane"):
                yield ChatTab()
            with TabPane("Train", id="train-pane"):
                yield TrainTab()
        yield Footer()


def main():
    app = DumbGPTApp()
    app.run()


if __name__ == "__main__":
    main()