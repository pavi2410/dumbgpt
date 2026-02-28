#!/usr/bin/env python3
"""
DumbGPT Training Script

Trains a GPT model on a corpus using TikToken BPE tokenizer.
Uses a streaming IterableDataset so the full corpus is never loaded into RAM.
"""

import math
import random
import time
import argparse
from pathlib import Path

import tiktoken
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.panel import Panel

from .model.transformer import GPTModel


# ---------------------------------------------------------------------------
# Model presets – pick one with --preset
# ---------------------------------------------------------------------------
PRESETS = {
    "micro": dict(d_model=256, num_heads=4,  d_ff=1024, num_layers=4,  max_seq_len=256),   # ~7M  — fast iteration / debugging
    "base":  dict(d_model=768, num_heads=12, d_ff=3072, num_layers=12, max_seq_len=1024),  # ~117M — main training target
}


console = Console()


# ---------------------------------------------------------------------------
# Pre-training dataset sources and interleave weights
# ---------------------------------------------------------------------------
DATASET_SOURCES = [
    # (hf_path, hf_split, text_field, mix_weight)
    ("roneneldan/TinyStories",    "train", "text", 0.3),
    ("HuggingFaceFW/fineweb-edu", "train", "text", 0.7),
]


class HFStreamingDataset(IterableDataset):
    """
    Streams tokenized windows directly from interleaved HuggingFace datasets.
    No pre-download step: HF caches parquet shards locally on first access
    and resumes automatically across runs.
    Sources and mix weights live in DATASET_SOURCES above.
    """

    def __init__(self, tokenizer, seq_len: int, steps: int, seed: int = 42):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.steps = steps
        self.seed = seed

    def _build_stream(self):
        from datasets import load_dataset, interleave_datasets
        streams = [
            load_dataset(path, split=split, streaming=True, trust_remote_code=False)
            for path, split, _, _ in DATASET_SOURCES
        ]
        probs = [w for _, _, _, w in DATASET_SOURCES]
        return interleave_datasets(streams, probabilities=probs, seed=self.seed)

    def __iter__(self):
        rng = random.Random(self.seed)
        yielded = 0
        for sample in self._build_stream():
            if yielded >= self.steps:
                break
            text = sample.get("text", "") or ""
            if not text:
                continue
            tokens = self.tokenizer.encode(text)
            if len(tokens) < self.seq_len + 1:
                continue
            start = rng.randint(0, len(tokens) - self.seq_len - 1)
            chunk = tokens[start: start + self.seq_len + 1]
            yield (
                torch.tensor(chunk[:-1], dtype=torch.long),
                torch.tensor(chunk[1:],  dtype=torch.long),
            )
            yielded += 1


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(model, config, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unwrap torch.compile'd model before saving
    raw = getattr(model, "_orig_mod", model)
    torch.save({"model_state_dict": raw.state_dict(), "config": config, "tokenizer_type": "tiktoken"}, path)


def build_lr_scheduler(optimizer, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float):
    """Linear warmup then cosine decay to min_lr."""
    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr / max_lr + (1.0 - min_lr / max_lr) * cosine
    return optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


def load_checkpoint(path: Path, device: torch.device, use_checkpointing: bool = False):
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
        use_checkpointing=use_checkpointing,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    return model, tok, cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train DumbGPT")
    parser.add_argument("--preset",     default="base",  choices=list(PRESETS), help="Model size preset")
    parser.add_argument("--checkpoint", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--compile",    action="store_true", help="torch.compile the model")
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch",      type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--warmup",     type=int,   default=100,  help="LR warmup steps")
    parser.add_argument("--steps",      type=int,   default=500,  help="Training steps per epoch")
    parser.add_argument("--val-steps",  type=int,   default=50,   help="Validation steps per epoch")
    parser.add_argument("--out",        default="models/model.pt", help="Output model path")
    parser.add_argument("--resume",     default=None, help="Resume from checkpoint path")
    args = parser.parse_args()

    device = get_device()
    # Free ~10-20% speedup on CUDA/XPU via TF32 tensor cores
    torch.set_float32_matmul_precision("high")
    console.print(f"Device: {device}  |  PyTorch {torch.__version__}")

    # ---- Tokenizer ----
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab

    # ---- Model ----
    preset = PRESETS[args.preset]
    config = {"vocab_size": vocab_size, **preset}

    if args.resume and Path(args.resume).exists():
        console.print(f"Resuming from {args.resume}")
        model, tokenizer, config = load_checkpoint(Path(args.resume), device)
    else:
        model = GPTModel(use_checkpointing=args.checkpoint, **config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(Panel(
        f"Preset: [bold]{args.preset}[/]  |  Params: [bold]{total_params:,}[/] total / {trainable:,} trainable\n"
        f"Config: {config}",
        title="Model", border_style="blue",
    ))

    # ---- Data loaders (streaming directly from HuggingFace) ----
    seq_len     = config["max_seq_len"]
    total_steps = args.epochs * args.steps
    # Use different seeds for train/val so they sample different documents
    train_ds = HFStreamingDataset(tokenizer, seq_len, steps=args.steps * args.batch, seed=42)
    val_ds   = HFStreamingDataset(tokenizer, seq_len, steps=args.val_steps * args.batch, seed=99)
    # num_workers=0: HF streaming datasets are not fork-safe; CPU overhead is minimal vs GPU
    train_loader = DataLoader(train_ds, batch_size=args.batch, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, num_workers=0)

    # ---- Optimizer + scheduler (linear warmup + cosine decay) ----
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    scheduler = build_lr_scheduler(
        optimizer,
        warmup_steps=args.warmup,
        total_steps=total_steps,
        max_lr=args.lr,
        min_lr=args.lr / 10,
    )

    # Optional: torch.compile (speeds up training after warmup, XPU support may vary)
    if args.compile:
        try:
            model = torch.compile(model)
            console.print("[green]torch.compile enabled[/]")
        except Exception as e:
            console.print(f"[yellow]torch.compile skipped:[/] {e}")

    # Mixed precision scaler (BF16 for XPU, FP16 for CUDA)
    use_amp = device.type in ("cuda", "xpu")
    amp_dtype = torch.bfloat16 if device.type == "xpu" else torch.float16
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")  # Only CUDA needs scaler
    amp_ctx = lambda: torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp)

    models_dir = Path(args.out).parent
    models_dir.mkdir(parents=True, exist_ok=True)
    best_path = models_dir / "best_model.pt"

    console.print(Panel(
        f"[bold]{args.epochs}[/] epochs  ×  [bold]{args.steps}[/] steps/epoch  ×  batch [bold]{args.batch}[/]",
        title="Training", border_style="blue",
    ))

    best_val_loss = float("inf")
    global_step = 0
    t0 = time.time()

    with Progress(
        TextColumn("{task.description}"),
        BarColumn(bar_width=38),
        MofNCompleteColumn(),
        TextColumn("[dim]{task.fields[stats]}"),
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=4,
    ) as progress:
        for epoch in range(1, args.epochs + 1):
            # -- Train --
            model.train()
            train_loss = 0.0
            n = 0
            train_task = progress.add_task(
                f"[cyan]Ep {epoch:2d}/{args.epochs} train", total=args.steps, stats=""
            )
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)

                with amp_ctx():
                    loss = model.get_loss(inputs, targets)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                train_loss += loss.item()
                n += 1
                global_step += 1
                progress.update(train_task, advance=1,
                                stats=f"loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}")

            # -- Validate --
            model.eval()
            val_loss = 0.0
            vn = 0
            val_task = progress.add_task(
                f"[yellow]Ep {epoch:2d}/{args.epochs} val  ", total=args.val_steps, stats=""
            )
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    with amp_ctx():
                        loss = model.get_loss(inputs, targets)
                    val_loss += loss.item()
                    vn += 1
                    progress.update(val_task, advance=1)

            avg_train = train_loss / max(n, 1)
            avg_val   = val_loss   / max(vn, 1)
            lr_now    = scheduler.get_last_lr()[0]
            elapsed   = time.time() - t0

            saved_marker = ""
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                save_checkpoint(model, config, best_path)
                saved_marker = "  [green]✓ best[/]"

            progress.console.print(
                f"[bold]Ep {epoch}/{args.epochs}[/]  "
                f"train=[red]{avg_train:.4f}[/]  val=[green]{avg_val:.4f}[/]  "
                f"lr={lr_now:.2e}  steps={global_step}  t={elapsed:.0f}s{saved_marker}"
            )

    save_checkpoint(model, config, Path(args.out))
    elapsed_total = time.time() - t0
    console.print(Panel(
        f"Best val loss: [bold green]{best_val_loss:.4f}[/]  |  "
        f"Time: [bold]{elapsed_total:.0f}s[/]  |  "
        f"Saved: [dim]{args.out}[/]",
        title="Done", border_style="green",
    ))


if __name__ == "__main__":
    main()