# Mech Interp Puzzles — April 2026

*Inspired by Callum McDougall's [ARENA Monthly Algorithmic Challenges](https://learn.arena.education/chapter1_transformer_interp/monthly_algorithmic/).*

Monthly algorithmic mechanistic interpretability challenge. Each puzzle is a toy model trained on a toy algorithmic task. The model is as simple as possible while achieving perfect accuracy. Your goal: reverse-engineer the algorithm the model learned.

**Starter notebook**: [Open in Colab](https://colab.research.google.com/github/andyrdt/puzzles/blob/main/04_2026/starter_notebook.ipynb)

## Puzzle 1: Max of List

Given a list of numbers, predict the maximum.

### Part (a): Distinct number tokens, 0-9

- **Model**: 1-layer attention-only transformer (no MLP)
- **Tokenization**: Each number 0-9 is its own token, plus `BOS`, `SEP`, `ANS`, `EOS`
- **Vocab size**: 14
- **Architecture**: `d_model=64`, `n_heads=4`, 18,944 parameters
- **Input format**: `[BOS] n1 [SEP] n2 [SEP] ... nk [ANS]`
- **Output**: The model predicts `max [EOS]` after `[ANS]`
- **Accuracy**: 100% on held-out test set
- **HuggingFace**: [`andyrdt/04_2026_puzzle_1a`](https://huggingface.co/andyrdt/04_2026_puzzle_1a)

**Key question**: How does a single attention layer find the maximum?

### Part (b): Digit-level tokenization, 0-99

- **Model**: 2-layer attention-only transformer (no MLP)
- **Tokenization**: Each number is two digit tokens (tens, ones), e.g. 42 → `4 2`
- **Vocab size**: 14 (digits 0-9 + BOS/SEP/ANS/EOS)
- **Architecture**: `d_model=64`, `n_heads=4`, 35,712 parameters
- **Input format**: `[BOS] d1t d1o [SEP] d2t d2o [SEP] ... dkt dko [ANS]`
- **Output**: The model predicts `max_tens max_ones [EOS]` after `[ANS]`
- **Accuracy**: 100% on held-out test set
- **HuggingFace**: [`andyrdt/04_2026_puzzle_1b`](https://huggingface.co/andyrdt/04_2026_puzzle_1b)

**Key questions**:
- How does layer 1 compose tens and ones digits into a number representation?
- How does layer 2 compare these composed representations?
- The tens digit is learned almost instantly (~1k steps) while the ones digit takes ~7k steps. Why?
- A 1-layer model gets tens digit 100% but plateaus at ~40% for ones digit. What exactly can't it do?

## Getting started

### Setup

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Training

```bash
# Puzzle 1a (takes ~2 minutes on GPU)
python 04_2026/puzzle1a/train.py

# Puzzle 1b (takes ~5 minutes on GPU)
python 04_2026/puzzle1b/train.py

# With wandb logging
python 04_2026/puzzle1a/train.py --wandb
```

### Pushing to HuggingFace

```bash
python 04_2026/push_to_hf.py --local_dir 04_2026/puzzle1a/checkpoints --repo_id your-username/04_2026_puzzle_1a
python 04_2026/push_to_hf.py --local_dir 04_2026/puzzle1b/checkpoints --repo_id your-username/04_2026_puzzle_1b
```

### Loading a trained model (from HuggingFace)

```python
import json, importlib, torch
from pathlib import Path
from huggingface_hub import hf_hub_download

# Download model definition and weights
model_py = hf_hub_download("andyrdt/04_2026_puzzle_1a", "model.py")
spec = importlib.util.spec_from_file_location("model", model_py)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

config = json.loads(Path(hf_hub_download("andyrdt/04_2026_puzzle_1a", "config.json")).read_text())
model = mod.AttentionOnlyTransformer.from_config(config["model"])
model.load_state_dict(torch.load(
    hf_hub_download("andyrdt/04_2026_puzzle_1a", "model.pt"),
    weights_only=True
))
model.eval()

# Run inference: max of [3, 7, 2, 5, 1]
BOS, SEP, ANS = 10, 11, 12
x = torch.tensor([[BOS, 3, SEP, 7, SEP, 2, SEP, 5, SEP, 1, ANS]])
logits, attns = model(x)
print(f"Predicted max: {logits[0, -1].argmax().item()}")  # → 7
```

See `starter_notebook.ipynb` for a full starter notebook ([Open in Colab](https://colab.research.google.com/github/andyrdt/puzzles/blob/main/04_2026/starter_notebook.ipynb)).

## Wandb metrics

When `--wandb` is enabled, the following are logged:

| Metric | Description |
|--------|-------------|
| `train/loss` | Cross-entropy loss on masked positions (per step) |
| `train/lr` | Current learning rate (per step) |
| `train/examples_seen` | Cumulative training examples (per step) |
| `train/epoch` | Current epoch (per step) |
| `eval/loss` | Test set loss (per eval) |
| `eval/acc` | Test set accuracy (per eval) |
| `eval/acc_max_{v}` | Per-value accuracy for max=v (puzzle 1a, per eval) |
| `eval/acc_tens` | Tens digit accuracy (puzzle 1b, per eval) |
| `eval/acc_ones` | Ones digit accuracy (puzzle 1b, per eval) |

## File structure

```
04_2026/
├── README.md
├── model.py              # Shared attention-only transformer
├── push_to_hf.py         # Push checkpoints to HuggingFace
├── starter_notebook.ipynb # Starter notebook (Open in Colab)
├── puzzle1a/
│   ├── train.py          # Training script for part (a)
│   └── checkpoints/      # Saved model, config, plots (gitignored)
└── puzzle1b/
    ├── train.py          # Training script for part (b)
    └── checkpoints/      # Saved model, config, plots (gitignored)
```
