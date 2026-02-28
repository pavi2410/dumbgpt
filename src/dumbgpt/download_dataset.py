#!/usr/bin/env python3
"""
Dataset download utility for DumbGPT.

Downloads and prepares training datasets from Hugging Face.
Supports TinyStories, Fineweb-edu, and other small high-quality datasets.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.panel import Panel
from rich.table import Table

console = Console()

def download_tinystories(output_dir: Path, max_stories: Optional[int] = None) -> int:
    """
    Download TinyStories dataset.

    Args:
        output_dir: Directory to save stories
        max_stories: Maximum number of stories to download (None = all)

    Returns:
        Number of stories saved
    """
    from datasets import load_dataset

    console.print("[bold blue]TinyStories[/] — loading from Hugging Face...")
    ds = load_dataset("roneneldan/TinyStories", split="train")

    total = min(max_stories, len(ds)) if max_stories else len(ds)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 1000
    num_batches = (total + batch_size - 1) // batch_size

    with Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Writing batches", total=num_batches)
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total)
            stories = [ds[i]["text"] for i in range(start_idx, end_idx)]
            filepath = output_dir / f"stories_{batch_idx:05d}.txt"
            filepath.write_text("\n\n---STORY_SEPARATOR---\n\n".join(stories), encoding="utf-8")
            progress.advance(task)

    return total


def download_fineweb_sample(output_dir: Path, num_docs: int = 100_000) -> int:
    """
    Download a sample from Fineweb-edu dataset (streaming, no full download required).

    Args:
        output_dir: Directory to save documents
        num_docs: Number of documents to sample

    Returns:
        Number of documents saved
    """
    from datasets import load_dataset

    console.print(f"[bold blue]Fineweb-edu[/] — streaming {num_docs:,} docs from Hugging Face...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 1000
    batch: list[str] = []
    batch_idx = 0
    total_saved = 0

    with Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        dl_task    = progress.add_task("Downloading docs",   total=num_docs)
        write_task = progress.add_task("Writing batches",    total=(num_docs + batch_size - 1) // batch_size)

        for doc in ds:
            if total_saved >= num_docs:
                break
            batch.append(doc["text"])
            total_saved += 1
            progress.advance(dl_task)

            if len(batch) >= batch_size:
                out = output_dir / f"fineweb_{batch_idx:05d}.txt"
                out.write_text("\n\n---DOC_SEPARATOR---\n\n".join(batch), encoding="utf-8")
                batch_idx += 1
                batch = []
                progress.advance(write_task)

        # flush remaining
        if batch:
            out = output_dir / f"fineweb_{batch_idx:05d}.txt"
            out.write_text("\n\n---DOC_SEPARATOR---\n\n".join(batch), encoding="utf-8")
            batch_idx += 1
            progress.advance(write_task)

    return total_saved


def main():
    parser = argparse.ArgumentParser(
        description="Download training datasets for DumbGPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download TinyStories (recommended for first run)
  uv run download_dataset.py tinystories
  
  # Download Fineweb sample
  uv run download_dataset.py fineweb
  
  # Download only 10k TinyStories for quick testing
  uv run download_dataset.py tinystories --max-stories 10000
        """
    )
    
    parser.add_argument(
        "dataset",
        choices=["tinystories", "fineweb"],
        help="Dataset to download"
    )
    parser.add_argument(
        "--output",
        default="corpus",
        help="Output directory (default: corpus)"
    )
    parser.add_argument(
        "--max-stories",
        type=int,
        default=None,
        help="Maximum number of stories (TinyStories only)"
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=100_000,
        help="Number of documents (Fineweb only, default: 100000)"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    try:
        from datasets import load_dataset  # noqa: F401
    except ImportError:
        console.print("[bold red]Error:[/] 'datasets' library not installed.")
        console.print("Install with: [bold]uv pip install datasets[/]")
        sys.exit(1)
    
    corpus_dir = Path(args.output)
    
    try:
        if args.dataset == "tinystories":
            output_dir = corpus_dir / "tinystories"
            count = download_tinystories(output_dir, args.max_stories)
            size_mb = sum(f.stat().st_size for f in output_dir.glob("*.txt")) / 1024 / 1024
        elif args.dataset == "fineweb":
            output_dir = corpus_dir / "fineweb"
            count = download_fineweb_sample(output_dir, args.num_docs)
            size_mb = sum(f.stat().st_size for f in output_dir.glob("*.txt")) / 1024 / 1024
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        sys.exit(1)

    num_files = len(list(output_dir.glob("*.txt")))
    table = Table(title="Download Complete", border_style="green")
    table.add_column("Field",  style="bold")
    table.add_column("Value")
    table.add_row("Dataset",   args.dataset)
    table.add_row("Documents", f"{count:,}")
    table.add_row("Files",     f"{num_files:,}  (1000 docs/file)")
    table.add_row("Size",      f"{size_mb:.1f} MB")
    table.add_row("Location",  str(output_dir))
    console.print(table)
    console.print()
    console.print("Next: [bold]uv run train --preset small --epochs 5 --steps 1000 --batch 8 --checkpoint[/]")


if __name__ == "__main__":
    main()
