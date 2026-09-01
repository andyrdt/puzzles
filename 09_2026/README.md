# Mech Interp Puzzles — September 2026

*Inspired by Callum McDougall's [ARENA Monthly Algorithmic Challenges](https://learn.arena.education/chapter1_transformer_interp/monthly_algorithmic/).*

Monthly algorithmic mechanistic interpretability challenge. Each puzzle is a toy model trained on a toy algorithmic task. The model is as simple as possible while achieving perfect accuracy. Your goal: reverse-engineer the algorithm the model learned.

**Starter notebook**: [Open in Colab](https://colab.research.google.com/github/andyrdt/puzzles/blob/main/09_2026/starter_notebook.ipynb)

## Puzzle 1: Set Difference

Given a set of four symbols, and the same set with one symbol removed, predict the missing symbol.

- **Input format**: `[BOS] x1 x2 x3 x4 [SEP] y1 y2 y3 [SEP]`, where `X = {x1, x2, x3, x4}` is a set of four distinct symbols and `Y = X \ {z}` is `X` with one symbol `z` removed. Both halves are independently shuffled.
- **Output**: at the final `[SEP]`, predict `z`
- **Vocab**: symbols `a`..`p` (ids 0..15), `BOS` (16), `SEP` (17)
- **Model**: 1-layer attention-only transformer; no MLP, no LayerNorm, no biases; learned positional embeddings; causal masking
- **Architecture**: `d_model=2`, `n_heads=2`, 108 parameters
- **Accuracy**: 100% on all 1,048,320 valid prompts (train/val/test split is over the underlying sets `X`)
- **HuggingFace**: [`andyrdt/09_2026_puzzle_1`](https://huggingface.co/andyrdt/09_2026_puzzle_1)

Example: `X = {b, e, j, n}`, `z = j`, `Y = {b, e, n}`

```text
[BOS] n b j e [SEP] e n b [SEP] -> j
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
# Reproduces the released model (deterministic, ~2 min on a modern GPU)
python 09_2026/puzzle1/train.py
```

### Evaluating

```bash
# Every ordering of every held-out test problem
python 09_2026/puzzle1/evaluate_checkpoint.py 09_2026/puzzle1/checkpoints --split test
```

### Pushing to HuggingFace

```bash
python 09_2026/push_to_hf.py --local_dir 09_2026/puzzle1/checkpoints --repo_id your-username/09_2026_puzzle_1
```

### Loading the released model

```python
import json, importlib, torch
from pathlib import Path
from huggingface_hub import hf_hub_download

model_py = hf_hub_download("andyrdt/09_2026_puzzle_1", "model.py")
spec = importlib.util.spec_from_file_location("model", model_py)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

config = json.loads(Path(hf_hub_download("andyrdt/09_2026_puzzle_1", "config.json")).read_text())
model = mod.AttentionOnlyTransformer.from_config(config["model"])
model.load_state_dict(torch.load(
    hf_hub_download("andyrdt/09_2026_puzzle_1", "model.pt"),
    weights_only=True
))
model.eval()

# X = {b, e, j, n}, Y = {b, e, n}  ->  missing symbol j
BOS, SEP = 16, 17
sym = lambda s: ord(s) - ord("a")
x = torch.tensor([[BOS, sym("n"), sym("b"), sym("j"), sym("e"), SEP, sym("e"), sym("n"), sym("b"), SEP]])
logits, attns = model(x)
print(f"Predicted: {chr(ord('a') + logits[0, -1].argmax().item())}")  # -> j
```

See `starter_notebook.ipynb` for a full starter ([Open in Colab](https://colab.research.google.com/github/andyrdt/puzzles/blob/main/09_2026/starter_notebook.ipynb)).

## File structure

```
09_2026/
├── README.md
├── model.py                  # Attention-only transformer
├── push_to_hf.py             # Push checkpoint to HuggingFace
├── starter_notebook.ipynb
└── puzzle1/
    ├── train.py
    ├── evaluate_checkpoint.py
    ├── test_protocol.py
    └── checkpoints/          # Saved model, config, plot (gitignored)
```
