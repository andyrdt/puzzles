# Mech Interp Puzzles — October 2026

*Inspired by Callum McDougall's [ARENA Monthly Algorithmic Challenges](https://learn.arena.education/chapter1_transformer_interp/monthly_algorithmic/).*

Monthly algorithmic mechanistic interpretability challenge. Each puzzle is a toy model trained on a toy algorithmic task. Your goal: reverse-engineer the algorithm the model learned.

**Starter notebook**: [Open in Colab](https://colab.research.google.com/github/andyrdt/puzzles/blob/main/10_2026/starter_notebook.ipynb)

## Puzzle 1: Set Difference, Part 2

Given a set of symbols and a permutation of it, read one symbol at a time, and predict which symbols could come next.

- **Input format**: `[BOS] x1 … xK [SEP] y1 … yK [EOS]`, where `X = {x1, …, xK}` is a set of `K` distinct symbols (`2 ≤ K ≤ 8`) and `Y` is an independent uniformly random permutation of `X`.
- **Output**: at `SEP` and at each `yi` with `i < K`, the uniform distribution over the symbols of `X` not among `y1 … yi`; at `yK`, `EOS`. No other positions are scored.
- **Vocab**: symbols `a`..`p` (ids 0..15), `BOS` (16), `SEP` (17), `EOS` (18)
- **Model**: 2-layer attention-only transformer, 1 head per layer; pre-norm RMSNorm before each attention layer and before the unembedding; rotary position embeddings (RoPE); causal masking; no MLPs, no biases
- **Architecture**: `d_model=64`, `n_heads=1`, 35,392 parameters
- **Training**: soft-target cross-entropy against the distributions above, AdamW (weight decay 0.1), 60k steps
- **Accuracy**: at every scored position, every valid next token has a higher logit than every invalid token, on all held-out sets (each in 1,024 random orderings). The train/val/test split is over the underlying sets `X`.
- **HuggingFace**: [`andyrdt/10_2026_puzzle_1`](https://huggingface.co/andyrdt/10_2026_puzzle_1)

Example: `X = {b, c, e, j, n}`

```text
[BOS] j e b c n [SEP] b j e n c [EOS]

after SEP -> uniform over {b, c, e, j, n}
after b   -> uniform over {c, e, j, n}
after j   -> uniform over {c, e, n}
after e   -> uniform over {c, n}
after n   -> c
after c   -> EOS
```

## Getting started

### Setup

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Training

```bash
# Trains the released configuration (~3 min on a modern GPU). Kernels are not bit-deterministic,
# so weights differ slightly from the released checkpoint, but behaviour matches.
python 10_2026/puzzle1/train.py
```

### Evaluating

```bash
# Every held-out test set, in many random and structured orderings
python 10_2026/puzzle1/evaluate_checkpoint.py 10_2026/puzzle1/checkpoints/<run_name> --split test
```

### Pushing to HuggingFace

```bash
python 10_2026/push_to_hf.py --local_dir 10_2026/puzzle1/checkpoints/<run_name> --repo_id your-username/10_2026_puzzle_1
```

### Loading the released model

```python
import json, importlib, torch
from pathlib import Path
from huggingface_hub import hf_hub_download

model_py = hf_hub_download("andyrdt/10_2026_puzzle_1", "model.py")
spec = importlib.util.spec_from_file_location("model", model_py)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

config = json.loads(Path(hf_hub_download("andyrdt/10_2026_puzzle_1", "config.json")).read_text())
model = mod.Transformer.from_config(config["model"])
model.load_state_dict(torch.load(
    hf_hub_download("andyrdt/10_2026_puzzle_1", "model.pt"),
    weights_only=True
))
model.eval()

# X = {b, c, e, j, n}
BOS, SEP, EOS = 16, 17, 18
sym = lambda s: ord(s) - ord("a")
x = torch.tensor([[BOS, *map(sym, "jebcn"), SEP, *map(sym, "bjenc"), EOS]])
logits, attns = model(x)
probs = logits[0].softmax(-1)
print(probs[7].topk(4))   # after "b": c, e, j, n (~0.25 each)
```

See `starter_notebook.ipynb` for a full starter ([Open in Colab](https://colab.research.google.com/github/andyrdt/puzzles/blob/main/10_2026/starter_notebook.ipynb)).

## File structure

```
10_2026/
├── README.md
├── model.py                  # Transformer (attention-only, RoPE, RMSNorm)
├── push_to_hf.py             # Push checkpoint to HuggingFace
├── starter_notebook.ipynb
└── puzzle1/
    ├── task.py               # Data protocol, train/val/test split, metrics
    ├── train.py
    ├── evaluate_checkpoint.py
    ├── test_task.py
    └── checkpoints/          # Saved model, config, plot (gitignored)
```
