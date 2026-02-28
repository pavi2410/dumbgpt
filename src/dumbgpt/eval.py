#!/usr/bin/env python3
"""
DumbGPT Evaluation Script

Measures:
  - Perplexity on held-out corpus (lower is better; GPT-2 ~30 on WebText)
  - Generation throughput (tokens/sec)
  - Qualitative generation samples
"""

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn

from .train import HFStreamingDataset, load_checkpoint, get_device

console = Console()


def compute_perplexity(
    model: torch.nn.Module,
    tokenizer,
    seq_len: int,
    num_batches: int = 50,
    batch_size: int = 8,
    device: str = "cpu",
) -> tuple[float, float]:
    """Estimate perplexity by streaming held-out samples from HuggingFace datasets.

    Returns:
        (perplexity, avg_cross_entropy_loss)
    """
    model.eval()
    ds = HFStreamingDataset(tokenizer, seq_len, steps=num_batches * batch_size, seed=777)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=0)

    total_loss = 0.0
    n = 0
    with Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Perplexity eval", total=num_batches)
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                total_loss += model.get_loss(inputs, targets).item()
                n += 1
                progress.advance(task)
                if n >= num_batches:
                    break

    avg_loss = total_loss / max(n, 1)
    return math.exp(avg_loss), avg_loss


def generate_sample(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.3,
) -> tuple[str, float]:
    """Generate text and return (generated_text, tokens_per_sec)."""
    tokens = tokenizer.encode(prompt)
    context = torch.tensor(tokens).unsqueeze(0)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            context, max_new_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
    elapsed = time.perf_counter() - t0

    new_tokens = out[0, len(tokens):].tolist()
    return tokenizer.decode(new_tokens), max_new_tokens / elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a DumbGPT model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",       default="models/best_model.pt", help="Checkpoint path")
    parser.add_argument("--ppl-batches", type=int, default=50,           help="Batches used for perplexity")
    parser.add_argument("--tokens",      type=int,   default=100,  help="Tokens to generate per prompt")
    parser.add_argument("--temperature", type=float, default=0.8,  help="Sampling temperature")
    parser.add_argument("--rep-penalty", type=float, default=1.3,  help="Repetition penalty (1.0=off, 1.3=moderate)")
    args = parser.parse_args()

    # --- Load ---
    path = Path(args.model)
    if not path.exists():
        path = Path("models/model.pt")
    if not path.exists():
        console.print("[bold red]No model checkpoint found.[/] Train first: [bold]uv run train --preset small[/]")
        return

    device = get_device()
    model, tokenizer, config = load_checkpoint(path, device)
    model.eval()

    params = sum(p.numel() for p in model.parameters())

    info = Table.grid(padding=(0, 2))
    info.add_row("[bold]Device[/]",   str(device))
    info.add_row("[bold]Checkpoint[/]", str(path))
    info.add_row("[bold]Params[/]",   f"{params:,}")
    info.add_row("[bold]Config[/]",   f"d_model={config['d_model']}  heads={config['num_heads']}  "
                                      f"layers={config['num_layers']}  seq_len={config['max_seq_len']}")
    console.print(Panel(info, title="Model", border_style="blue"))

    # --- Perplexity ---
    ppl, loss = compute_perplexity(
        model, tokenizer,
        seq_len=config["max_seq_len"],
        num_batches=args.ppl_batches,
        device=device,
    )
    ppl_color = "green" if ppl < 100 else "yellow" if ppl < 500 else "red"
    console.print(Panel(
        f"Loss: [bold]{loss:.4f}[/]   Perplexity: [bold {ppl_color}]{ppl:.1f}[/]\n"
        f"[dim](Untrained baseline ~{tokenizer.n_vocab:,}  │  GPT-2 small ~30 on WebText)[/]",
        title=f"Perplexity  ({args.ppl_batches} batches × 8)", border_style=ppl_color,
    ))

    # --- Generation quality + throughput ---
    # Prompts cover both TinyStories (narrative) and Fineweb-edu (factual/instructional) styles
    prompt_groups: list[tuple[str, list[str]]] = [
        ("Narrative (TinyStories style)", [
            "Once upon a time, there was a little girl named",
            "Tom and his dog went to the park. Suddenly",
            "One day, a small bunny found a",
        ]),
        ("Factual / Science", [
            "The solar system consists of the Sun and",
            "Water is made up of hydrogen and oxygen atoms. When",
            "Photosynthesis is the process by which plants",
        ]),
        ("Educational / How-to", [
            "To solve a quadratic equation, you need to",
            "The most important thing to remember when learning a new language is",
            "Scientists use the scientific method to",
        ]),
        ("General Web / News", [
            "Researchers at the university have discovered that",
            "The new technology allows users to",
            "According to recent studies, the best way to",
        ]),
    ]

    table = Table(title="Generation Samples", border_style="cyan", show_lines=True)
    table.add_column("Category",  style="bold yellow",  max_width=20, overflow="fold")
    table.add_column("Prompt",    style="bold",         max_width=30, overflow="fold")
    table.add_column("Output",    style="dim",          max_width=50, overflow="fold")
    table.add_column("tok/s",     justify="right",      style="green", width=7)

    throughputs = []
    for category, prompts in prompt_groups:
        for prompt in prompts:
            text, tps = generate_sample(
                model, tokenizer, prompt,
                max_new_tokens=args.tokens,
                temperature=args.temperature,
                repetition_penalty=args.rep_penalty,
            )
            throughputs.append(tps)
            table.add_row(category, prompt, text.strip(), f"{tps:.1f}")
            category = ""  # only show category label on first row of each group

    console.print(table)
    avg_tps = sum(throughputs) / len(throughputs)
    console.print(Panel(
        f"Avg throughput: [bold green]{avg_tps:.1f} tok/s[/]",
        title="Done", border_style="green",
    ))


if __name__ == "__main__":
    main()