# Mech Interp Puzzles — May 2026

*Inspired by Callum McDougall's [ARENA Monthly Algorithmic Challenges](https://learn.arena.education/chapter1_transformer_interp/monthly_algorithmic/).*

Monthly algorithmic mechanistic interpretability challenge. Each puzzle is a toy model trained on a toy algorithmic task. Your goal: reverse-engineer the algorithm the model learned.

**Starter notebook**: [Open in Colab](https://colab.research.google.com/github/andyrdt/puzzles/blob/main/05_2026/starter_notebook.ipynb)

## Puzzle 1: Count Unique Tokens

Given a sequence of 10 tokens drawn from 10 symbols, predict the number of distinct symbols.

- **Input format**: `[BOS] t1 t2 ... t10 [ANS]`
- **Output**: at `[ANS]`, predict count token `#k` where `k = |{t1, ..., t10}|`
- **Vocab**: input symbols `a`..`j` (ids 0..9), `BOS` (10), `ANS` (11), count tokens `#1`..`#10` (ids 12..21)
- **Model**: 2-layer attention-only transformer; no MLP, no LayerNorm, no positional embeddings; causal masking
- **Architecture**: `d_model=32`, `n_heads=4`, 9,600 parameters
- **Accuracy**: 100% on every count on a held-out test set
- **HuggingFace**: [`andyrdt/05_2026_puzzle_1`](https://huggingface.co/andyrdt/05_2026_puzzle_1)

## Getting started

### Setup

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Training

```bash
# Reproduces the released model (deterministic, ~5 min on a modern GPU)
python 05_2026/puzzle1/train.py

# With wandb logging
python 05_2026/puzzle1/train.py --wandb
```

### Pushing to HuggingFace

```bash
python 05_2026/push_to_hf.py --local_dir 05_2026/puzzle1/checkpoints --repo_id your-username/05_2026_puzzle_1
```

### Loading the released model

```python
import json, importlib, torch
from pathlib import Path
from huggingface_hub import hf_hub_download

model_py = hf_hub_download("andyrdt/05_2026_puzzle_1", "model.py")
spec = importlib.util.spec_from_file_location("model", model_py)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

config = json.loads(Path(hf_hub_download("andyrdt/05_2026_puzzle_1", "config.json")).read_text())
model = mod.AttentionOnlyTransformer.from_config(config["model"])
model.load_state_dict(torch.load(
    hf_hub_download("andyrdt/05_2026_puzzle_1", "model.pt"),
    weights_only=True
))
model.eval()

# [a, b, a, c, a, b, d, a, c, e] -> 5 distinct symbols
BOS, ANS = 10, 11
COUNT_BASE = 12  # count k -> COUNT_BASE + (k - 1)
seq = [BOS, 0, 1, 0, 2, 0, 1, 3, 0, 2, 4, ANS]
x = torch.tensor([seq])
logits, attns = model(x)
pred = logits[0, -1].argmax().item() - COUNT_BASE + 1
print(f"Predicted count: {pred}")  # -> 5
```

See `starter_notebook.ipynb` for a full starter ([Open in Colab](https://colab.research.google.com/github/andyrdt/puzzles/blob/main/05_2026/starter_notebook.ipynb)).

## Wandb metrics

When `--wandb` is enabled:

| Metric | Description |
|--------|-------------|
| `train/loss` | Cross-entropy loss at the count-prediction position (per step) |
| `train/lr` | Learning rate (per step) |
| `train/examples_seen` | Cumulative training examples (per step) |
| `train/epoch` | Current epoch (per step) |
| `eval/loss` | Test set loss (per eval) |
| `eval/acc` | Test set accuracy (per eval) |
| `eval/train_loss` | Train-subset loss for memorization check (per eval) |
| `eval/train_acc` | Train-subset accuracy for memorization check (per eval) |
| `eval/acc_count_{k}` | Per-count accuracy for unique-count = k (per eval) |

## File structure

```
05_2026/
├── README.md
├── model.py              # Shared attention-only transformer
├── push_to_hf.py         # Push checkpoints to HuggingFace
├── starter_notebook.ipynb
└── puzzle1/
    ├── train.py
    └── checkpoints/      # Saved model, config, plot (gitignored)
```
