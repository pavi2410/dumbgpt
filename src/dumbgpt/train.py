#!/usr/bin/env python3
"""
DumbGPT Training Script

Trains a GPT model on a corpus using TikToken BPE tokenizer.
Uses a streaming IterableDataset so the full corpus is never loaded into RAM.
"""

import sys
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
from rich.table import Table
from rich.panel import Panel

from .model.transformer import GPTModel


# ---------------------------------------------------------------------------
# Model presets – pick one with --preset
# ---------------------------------------------------------------------------
PRESETS = {
    "small": dict(d_model=512, num_heads=8,  d_ff=2048, num_layers=8,  max_seq_len=512),   # ~10M
    "base":  dict(d_model=768, num_heads=12, d_ff=3072, num_layers=12, max_seq_len=1024), # ~100M
}


# ---------------------------------------------------------------------------
# Corpus helpers
# ---------------------------------------------------------------------------
console = Console()

def iter_corpus_files(corpus_dir: Path):
    """Yield (path, kind) for every .txt batch file in corpus/tinystories and corpus/fineweb."""
    corpus_dir = Path(corpus_dir)
    for ts_file in sorted((corpus_dir / "tinystories").glob("*.txt")) if (corpus_dir / "tinystories").exists() else []:
        yield ts_file, "tinystories"
    for fw_file in sorted((corpus_dir / "fineweb").glob("*.txt")) if (corpus_dir / "fineweb").exists() else []:
        yield fw_file, "fineweb"


def read_file_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Streaming dataset – samples random windows from random files on the fly
# ---------------------------------------------------------------------------
class StreamingCorpusDataset(IterableDataset):
    """
    Yields (input_ids, target_ids) pairs without loading the whole corpus.
    
    Handles both single-item files and batched files (with ---SEPARATOR---).
    Each worker independently shuffles its file list and draws random windows
    of length seq_len+1 from each file's token stream.
    """

    def __init__(self, file_paths: list[Path], tokenizer, seq_len: int, steps_per_epoch: int):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.steps_per_epoch = steps_per_epoch

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files = self.file_paths.copy()

        # Split files across workers to avoid duplicate samples
        if worker_info is not None:
            per_worker = max(1, len(files) // worker_info.num_workers)
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(files)
            files = files[start:end]
            steps = max(1, self.steps_per_epoch // worker_info.num_workers)
            seed = worker_info.id
        else:
            steps = self.steps_per_epoch
            seed = 0

        rng = random.Random(seed)
        rng.shuffle(files)

        yielded = 0
        while yielded < steps:
            rng.shuffle(files)
            for path in files:
                if yielded >= steps:
                    break
                text = read_file_safe(path)
                if not text:
                    continue

                # Handle batched files: split by separator and pick one item randomly
                if "---STORY_SEPARATOR---" in text:
                    items = [s.strip() for s in text.split("---STORY_SEPARATOR---") if s.strip()]
                    text = rng.choice(items)
                elif "---DOC_SEPARATOR---" in text:
                    items = [s.strip() for s in text.split("---DOC_SEPARATOR---") if s.strip()]
                    text = rng.choice(items)

                tokens = self.tokenizer.encode(text)
                if len(tokens) < self.seq_len + 1:
                    continue
                start = rng.randint(0, len(tokens) - self.seq_len - 1)
                chunk = tokens[start: start + self.seq_len + 1]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                target_ids = torch.tensor(chunk[1:],  dtype=torch.long)
                yield input_ids, target_ids
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
    torch.save({"model_state_dict": model.state_dict(), "config": config, "tokenizer_type": "tiktoken"}, path)


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
    parser.add_argument("--preset",     default="small",  choices=list(PRESETS), help="Model size preset")
    parser.add_argument("--checkpoint", action="store_true", help="Enable gradient checkpointing (saves VRAM, slower)")
    parser.add_argument("--compile",    action="store_true", help="torch.compile the model (faster after warmup)")
    parser.add_argument("--epochs",    type=int,   default=5)
    parser.add_argument("--batch",     type=int,   default=32)
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--steps",     type=int,   default=500,  help="Training steps per epoch")
    parser.add_argument("--val-steps", type=int,   default=50,   help="Validation steps per epoch")
    parser.add_argument("--corpus",    default="corpus", help="Path to corpus directory")
    parser.add_argument("--out",       default="models/model.pt", help="Output model path")
    parser.add_argument("--resume",    default=None, help="Resume from checkpoint path")
    args = parser.parse_args()

    device = get_device()
    console.print(f"Device: {device}  |  PyTorch {torch.__version__}")

    # ---- Corpus ----
    corpus_dir = Path(args.corpus)
    all_files = list(iter_corpus_files(corpus_dir))
    if not all_files:
        console.print(f"No corpus files found in {corpus_dir}")
        sys.exit(1)

    console.print(f"Corpus: {len(all_files)} files")

    # Split files 90/10 for train/val
    random.shuffle(all_files)
    split = int(len(all_files) * 0.9)
    train_files = [p for p, _ in all_files[:split]]
    val_files   = [p for p, _ in all_files[split:]] or train_files[:max(1, len(train_files)//10)]

    # ---- Tokenizer ----
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab
    console.print(f"Vocab size: {vocab_size:,}")

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
    console.print(f"Preset: {args.preset}  |  Params: {total_params:,} total / {trainable:,} trainable")
    console.print(f"Config: {config}")

    # ---- Data loaders ----
    seq_len = config["max_seq_len"]
    # Total samples per epoch = steps * batch_size
    train_samples = args.steps * args.batch
    val_samples   = args.val_steps * args.batch

    train_ds = StreamingCorpusDataset(train_files, tokenizer, seq_len, train_samples)
    val_ds   = StreamingCorpusDataset(val_files,   tokenizer, seq_len, val_samples)

    train_loader = DataLoader(train_ds, batch_size=args.batch, num_workers=2, prefetch_factor=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, num_workers=2, prefetch_factor=2)

    # ---- Optimizer + scheduler ----
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    total_steps = args.epochs * args.steps
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr / 10)

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