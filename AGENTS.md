# AGENTS.md

## Cursor Cloud specific instructions

DumbGPT is a small PyTorch GPT (decoder-only transformer) with four entry points,
all run via `uv` (see `[project.scripts]` in `pyproject.toml`):

- `uv run train` — train a model (streams data from the HuggingFace Hub)
- `uv run eval` — perplexity + generation eval against a checkpoint
- `uv run tui` — Textual terminal chat UI for a trained model
- `uv run pytest` — test suite (see `tests/`)

There is no linter/formatter configured in this repo; `uv run pytest` is the only check.

### Environment / hardware notes

- Requires Python 3.14; `uv sync` provisions it automatically. `uv` is installed to
  `~/.local/bin` (the update script does not assume it is on `PATH`).
- `torch` is pinned to the Intel XPU wheel index. On a CPU-only VM there is no GPU,
  so the code falls back to `device=cpu`. The `XPU device count is zero!` warning at
  startup is expected and harmless.

### Data streaming gotcha (important)

- `train` and `eval` stream `roneneldan/TinyStories` and `HuggingFaceFW/fineweb-edu`
  directly from the HuggingFace Hub — there is no pre-download step and network access
  is required on first run. The **first** run can spend several minutes caching parquet
  shards into `~/.cache/huggingface`; subsequent runs reuse the cache and are fast
  (a `micro` smoke train finishes in ~10s once cached). Do NOT pipe these commands
  through `tail`/`head`, as that buffers the `rich` progress output and makes them look
  hung — redirect to a log file and read it instead.
- Requests are unauthenticated by default (a `Set a HF_TOKEN` warning prints); this is
  fine for smoke tests but a `HF_TOKEN` secret raises rate limits.

### Quick smoke commands

- Fast train: `uv run train --preset micro --epochs 1 --steps 5 --batch 2 --val-steps 2 --warmup 2`
  (writes `models/model.pt` and `models/best_model.pt`).
- Fast eval: `uv run eval --model models/model.pt --ppl-batches 3 --tokens 20`
  (builds `data/eval_holdout/` on first run via `save_to_disk`).
- `models/` and `data/` are gitignored; checkpoints from a tiny smoke run produce
  gibberish text and near-baseline perplexity (~50k) — that is expected.
