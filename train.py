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

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dumbgpt.model.transformer import GPTModel
from dumbgpt.tokenizer.tiktoken_tokenizer import TikTokenTokenizer


# ---------------------------------------------------------------------------
# Model presets – pick one with --preset
# ---------------------------------------------------------------------------
PRESETS = {
    "nano": dict(d_model=128, num_heads=4, d_ff=256,  num_layers=3, max_seq_len=128),
    "small": dict(d_model=256, num_heads=4, d_ff=512,  num_layers=4, max_seq_len=128),
    "medium": dict(d_model=384, num_heads=6, d_ff=768,  num_layers=6, max_seq_len=256),
}


# ---------------------------------------------------------------------------
# Corpus helpers
# ---------------------------------------------------------------------------
CODE_EXTENSIONS = {".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs"}


def iter_corpus_files(corpus_dir: Path):
    """Yield (path, kind) for every readable file in the corpus."""
    corpus_dir = Path(corpus_dir)

    for txt in (corpus_dir / "novels").glob("*.txt") if (corpus_dir / "novels").exists() else []:
        yield txt, "novel"

    for ext_file in (corpus_dir / "code").glob("*") if (corpus_dir / "code").exists() else []:
        yield ext_file, "code"

    nm = corpus_dir / "node_modules"
    if nm.exists():
        for p in nm.rglob("*"):
            if p.suffix in CODE_EXTENSIONS and p.is_file():
                yield p, "node_modules"


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

    Each worker independently shuffles its file list and draws random windows
    of length seq_len+1 from each file's token stream.
    """

    def __init__(self, file_paths: list[Path], tokenizer, seq_len: int, steps_per_epoch: int):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.steps_per_epoch = steps_per_epoch

    def __iter__(self):
        rng = random.Random()
        files = self.file_paths.copy()
        rng.shuffle(files)

        yielded = 0
        while yielded < self.steps_per_epoch:
            rng.shuffle(files)
            for path in files:
                if yielded >= self.steps_per_epoch:
                    break
                text = read_file_safe(path)
                if not text:
                    continue
                tokens = self.tokenizer.encode(text)
                if len(tokens) < self.seq_len + 1:
                    continue
                # Random window
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


def load_checkpoint(path: Path, device: torch.device):
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
    return model, tok, cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train DumbGPT")
    parser.add_argument("--preset",    default="small",  choices=list(PRESETS), help="Model size preset")
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
    print(f"Device: {device}  |  PyTorch {torch.__version__}")

    # ---- Corpus ----
    corpus_dir = Path(args.corpus)
    all_files = list(iter_corpus_files(corpus_dir))
    if not all_files:
        print(f"No corpus files found in {corpus_dir}")
        sys.exit(1)

    print(f"Corpus: {len(all_files)} files")

    # Split files 90/10 for train/val
    random.shuffle(all_files)
    split = int(len(all_files) * 0.9)
    train_files = [p for p, _ in all_files[:split]]
    val_files   = [p for p, _ in all_files[split:]] or train_files[:max(1, len(train_files)//10)]

    # ---- Tokenizer ----
    tokenizer = TikTokenTokenizer()
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size:,}")

    # ---- Model ----
    preset = PRESETS[args.preset]
    config = {"vocab_size": vocab_size, **preset}

    if args.resume and Path(args.resume).exists():
        print(f"Resuming from {args.resume}")
        model, tokenizer, config = load_checkpoint(Path(args.resume), device)
    else:
        model = GPTModel(**config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Preset: {args.preset}  |  Params: {total_params:,} total / {trainable:,} trainable")
    print(f"Config: {config}")

    # ---- Data loaders ----
    seq_len = config["max_seq_len"]
    # Total samples per epoch = steps * batch_size
    train_samples = args.steps * args.batch
    val_samples   = args.val_steps * args.batch

    train_ds = StreamingCorpusDataset(train_files, tokenizer, seq_len, train_samples)
    val_ds   = StreamingCorpusDataset(val_files,   tokenizer, seq_len, val_samples)

    train_loader = DataLoader(train_ds, batch_size=args.batch, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, num_workers=0)

    # ---- Optimizer + scheduler ----
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    total_steps = args.epochs * args.steps
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr / 10)

    # Mixed precision scaler (only for CUDA)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    amp_ctx = lambda: torch.amp.autocast(device_type=device.type, enabled=use_amp)

    models_dir = Path(args.out).parent
    models_dir.mkdir(parents=True, exist_ok=True)
    best_path = models_dir / "best_model.pt"

    print(f"\nTraining: {args.epochs} epochs x {args.steps} steps/epoch  (batch={args.batch})")
    print("-" * 60)

    best_val_loss = float("inf")
    global_step = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        # -- Train --
        model.train()
        train_loss = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:2d}/{args.epochs} [train]", leave=True)
        for inputs, targets in pbar:
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
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        # -- Validate --
        model.eval()
        val_loss = 0.0
        vn = 0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch:2d}/{args.epochs} [val]  ", leave=True):
                inputs, targets = inputs.to(device), targets.to(device)
                with amp_ctx():
                    loss = model.get_loss(inputs, targets)
                val_loss += loss.item()
                vn += 1

        avg_train = train_loss / max(n, 1)
        avg_val   = val_loss   / max(vn, 1)
        lr_now    = scheduler.get_last_lr()[0]
        elapsed   = time.time() - t0

        print(f"Epoch {epoch:2d}/{args.epochs}  train={avg_train:.4f}  val={avg_val:.4f}  "
              f"lr={lr_now:.2e}  steps={global_step}  t={elapsed:.0f}s")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_checkpoint(model, config, best_path)
            print(f"  -> saved best model ({best_val_loss:.4f})")

    save_checkpoint(model, config, Path(args.out))
    print(f"\nDone in {time.time()-t0:.0f}s  |  Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {args.out}")


if __name__ == "__main__":
    main()