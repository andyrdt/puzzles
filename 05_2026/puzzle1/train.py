"""
Puzzle 1: Count Unique Tokens — 2-Layer Attention-Only Transformer

Task: Given a sequence of N tokens drawn from V symbols, predict the number
of unique symbols in the sequence.

Input:  [BOS] t1 t2 ... tN [ANS]
Target: count_k   (where k is the number of unique tokens, 1..N)

Vocabulary (combined input + output, with disjoint roles):
  - 0 .. V-1            : input symbols (rendered as letters a, b, c, ...)
  - V                    : BOS
  - V+1                  : ANS
  - V+2 .. V+1+N         : output count tokens (count_1, count_2, ..., count_N)

Model: 2-layer, 4-head attention-only causal transformer (no MLP, no positional
embeddings, no layernorm). The task is order-invariant, so positional info
isn't required — the model solves it purely from token identities.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Required by torch.use_deterministic_algorithms(True) for matmul-style ops
# on CUDA. Must be set BEFORE the first CUDA context is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import AttentionOnlyTransformer


# ── Vocab ───────────────────────────────────────────────────────────────────

class Vocab:
    """
    Layout:
        [0, V-1]            input symbols (rendered as letters a, b, c, ...)
        V                   BOS
        V + 1               ANS
        [V+2, V+1+N]        count tokens (count k -> token V + 1 + k)
    """

    def __init__(self, num_symbols: int, seq_len: int):
        self.num_symbols = num_symbols
        self.seq_len = seq_len
        self.BOS = num_symbols
        self.ANS = num_symbols + 1
        self.COUNT_BASE = num_symbols + 2  # count k -> COUNT_BASE + (k - 1)
        self.size = num_symbols + 2 + seq_len

    def count_token(self, k: int) -> int:
        assert 1 <= k <= self.seq_len
        return self.COUNT_BASE + (k - 1)

    def token_name(self, tok: int) -> str:
        if 0 <= tok < self.num_symbols:
            return chr(ord("a") + tok)
        if tok == self.BOS:
            return "BOS"
        if tok == self.ANS:
            return "ANS"
        if self.COUNT_BASE <= tok < self.COUNT_BASE + self.seq_len:
            return f"#{tok - self.COUNT_BASE + 1}"
        raise ValueError(f"Unknown token id {tok}")

    def to_dict(self):
        return {"type": "unique_count", "num_symbols": self.num_symbols, "seq_len": self.seq_len}


# ── Dataset ─────────────────────────────────────────────────────────────────

class UniqueCountDataset(Dataset):
    """
    Sequence: BOS t1 t2 ... tN ANS count_k
    Length:   N + 3
    Input:    drop last token (length N + 2)
    Loss:     only at position predicting count_k (= position of ANS in input)
    """

    def __init__(self, vocab: Vocab, sequences: np.ndarray):
        self.vocab = vocab
        self.size = len(sequences)
        N = vocab.seq_len
        seq_len = N + 3

        unique_counts = np.array([len(set(row.tolist())) for row in sequences], dtype=np.int64)
        seqs = np.empty((self.size, seq_len), dtype=np.int64)
        seqs[:, 0] = vocab.BOS
        seqs[:, 1 : 1 + N] = sequences
        seqs[:, 1 + N] = vocab.ANS
        for i, c in enumerate(unique_counts):
            seqs[i, 2 + N] = vocab.count_token(int(c))

        self.inputs = torch.tensor(seqs[:, :-1])
        self.targets = torch.tensor(seqs[:, 1:])

        self.loss_mask = torch.zeros_like(self.targets, dtype=torch.bool)
        ans_pos_in_input = 1 + N  # this position predicts count_k as next token
        self.loss_mask[:, ans_pos_in_input] = True

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.loss_mask[idx]


# ── Data generation ─────────────────────────────────────────────────────────

def generate_data(num_symbols, seq_len, samples_per_count, seed):
    """
    Stratified by target unique-count c in 1..N. For each c:
      1. pick c symbols from V (without replacement)
      2. seed the sequence with one copy of each of the c symbols
      3. fill the remaining N-c positions by sampling from those c symbols
      4. shuffle
    Dedupe within each c. For very small c there may be fewer than `samples_per_count`
    distinct sequences (e.g. c=1 has exactly V).
    """
    rng = np.random.default_rng(seed)
    all_seqs = []
    counts_actual = {}

    for c in range(1, seq_len + 1):
        seqs_for_c = set()
        # Cap total attempts so very-small-c doesn't loop forever.
        max_attempts = samples_per_count * 50
        attempts = 0
        while len(seqs_for_c) < samples_per_count and attempts < max_attempts:
            symbols = rng.choice(num_symbols, size=c, replace=False)
            extras = rng.choice(symbols, size=seq_len - c, replace=True)
            seq = np.concatenate([symbols, extras])
            rng.shuffle(seq)
            seqs_for_c.add(tuple(int(x) for x in seq))
            attempts += 1
        counts_actual[c] = len(seqs_for_c)
        all_seqs.extend([list(s) for s in seqs_for_c])

    all_seqs = np.array(all_seqs, dtype=np.int64)
    rng.shuffle(all_seqs)
    return all_seqs, counts_actual


def stratified_split(sequences, vocab, test_frac, seed):
    """
    Split into train/test such that each unique-count value is represented in
    both splits (when possible — c=1 has only V sequences, so test may be tiny
    or empty for it; we accept that).
    """
    rng = np.random.default_rng(seed + 1)
    counts = np.array([len(set(row.tolist())) for row in sequences])
    train_idx, test_idx = [], []
    for c in np.unique(counts):
        idx = np.where(counts == c)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_frac))) if len(idx) >= 5 else 0
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return sequences[np.array(train_idx)], sequences[np.array(test_idx)] if test_idx else sequences[:0]


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(model, loader, vocab, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    total_correct = 0
    n_samples = 0
    per_count_correct = {c: 0 for c in range(1, vocab.seq_len + 1)}
    per_count_total = {c: 0 for c in range(1, vocab.seq_len + 1)}

    ans_idx = 1 + vocab.seq_len  # position in *input* whose next-token is count

    with torch.no_grad():
        for inputs, targets, loss_mask in loader:
            inputs, targets, loss_mask = (
                inputs.to(device), targets.to(device), loss_mask.to(device)
            )
            logits, _ = model(inputs)
            loss = F.cross_entropy(
                logits.view(-1, vocab.size), targets.view(-1), reduction="none"
            ).view_as(targets)
            total_loss += (loss * loss_mask).sum().item()
            total_count += loss_mask.sum().item()

            preds = logits[:, ans_idx].argmax(dim=-1)
            true_count_token = targets[:, ans_idx]
            correct = preds == true_count_token
            total_correct += correct.sum().item()
            n_samples += inputs.shape[0]

            true_count = (true_count_token - vocab.COUNT_BASE + 1).cpu().numpy()
            corr_np = correct.cpu().numpy()
            for c in range(1, vocab.seq_len + 1):
                m = true_count == c
                per_count_correct[c] += int(corr_np[m].sum())
                per_count_total[c] += int(m.sum())

    model.train()
    per_count_acc = {
        c: per_count_correct[c] / per_count_total[c]
        for c in per_count_total
        if per_count_total[c] > 0
    }
    return {
        "loss": total_loss / max(total_count, 1),
        "acc": total_correct / max(n_samples, 1),
        "per_count_acc": per_count_acc,
        "per_count_total": per_count_total,
    }


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_training(history, per_count_acc, save_dir, args):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["step"], history.get("train_eval_loss", history["train_loss"]), label="train")
    axes[0].plot(history["step"], history["test_loss"], label="test")
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Loss"); axes[0].set_title("Loss")
    axes[0].legend()

    if "train_acc" in history and history["train_acc"]:
        axes[1].plot(history["step"], history["train_acc"], label="train")
    axes[1].plot(history["step"], history["test_acc"], label="test")
    axes[1].set_xlabel("Step"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy"); axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    cs = sorted(per_count_acc.keys())
    accs = [per_count_acc[c] for c in cs]
    axes[2].bar(cs, accs)
    axes[2].set_xlabel("Unique count"); axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Accuracy by unique count"); axes[2].set_ylim(0, 1.05)

    plt.suptitle(
        f"Puzzle (May 2026): unique-count  "
        f"V={args.num_symbols}, N={args.seq_len}, "
        f"L={args.n_layers}, h={args.n_heads}, d={args.d_model}, pos={args.pos_embed}"
    )
    plt.tight_layout()
    plt.savefig(save_dir / "training.png", dpi=150)
    plt.close()


# ── Training ────────────────────────────────────────────────────────────────

def set_deterministic(seed: int):
    """Make a training run bit-exactly reproducible across reruns on the same hardware."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def train(args):
    set_deterministic(args.seed)

    vocab = Vocab(args.num_symbols, args.seq_len)
    input_seq_len = args.seq_len + 2  # BOS + N + ANS

    print(f"Vocab size: {vocab.size} ({args.num_symbols} symbols + BOS/ANS + {args.seq_len} count tokens)")
    print(f"Sequence length (input): {input_seq_len}")
    print(f"Model: {args.n_layers}L attention-only, d_model={args.d_model}, n_heads={args.n_heads}")

    # Data
    all_seqs, counts_actual = generate_data(
        args.num_symbols, args.seq_len, args.samples_per_count, args.seed
    )
    print(f"Generated {len(all_seqs)} unique sequences across all counts.")
    print(f"Per-count sample counts: {counts_actual}")

    train_seqs, test_seqs = stratified_split(all_seqs, vocab, args.test_frac, args.seed)
    print(f"Data: {len(train_seqs)} train, {len(test_seqs)} test")

    train_ds = UniqueCountDataset(vocab, train_seqs)
    test_ds = UniqueCountDataset(vocab, test_seqs)
    # Pin the DataLoader's shuffle to its own seeded generator, decoupled from the
    # global torch RNG (which is also consumed by model init).
    loader_gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, generator=loader_gen)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Fixed train-eval subset: same size as test set for fair train-vs-test comparison.
    rng_eval = np.random.default_rng(args.seed + 7)
    train_eval_idx = rng_eval.choice(len(train_ds), size=min(len(test_ds), len(train_ds)), replace=False)
    train_eval_ds = torch.utils.data.Subset(train_ds, train_eval_idx.tolist())
    train_eval_loader = DataLoader(train_eval_ds, batch_size=args.batch_size, shuffle=False)

    model = AttentionOnlyTransformer(
        vocab_size=vocab.size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        max_seq_len=input_seq_len,
        n_layers=args.n_layers,
        pos_embed_type=args.pos_embed,
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or (
                f"puzzle_may2026_L{args.n_layers}_d{args.d_model}_h{args.n_heads}_{args.pos_embed}"
            ),
            config={
                "puzzle": "may_2026_unique_count",
                "num_symbols": args.num_symbols,
                "seq_len": args.seq_len,
                "samples_per_count": args.samples_per_count,
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "pos_embed": args.pos_embed,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "steps": args.steps,
                "seed": args.seed,
                "n_params": n_params,
                "n_train": len(train_seqs),
                "n_test": len(test_seqs),
                "vocab_size": vocab.size,
                "input_seq_len": input_seq_len,
            },
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    step = 0
    examples_seen = 0
    epoch = 0
    history = {"step": [], "train_loss": [], "test_loss": [], "test_acc": [], "train_acc": [], "train_eval_loss": []}
    pbar = tqdm(total=args.steps, desc="Training")

    while step < args.steps:
        epoch += 1
        for inputs, targets, loss_mask in train_loader:
            if step >= args.steps:
                break
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            loss_mask = loss_mask.to(args.device)

            logits, _ = model(inputs)
            loss = F.cross_entropy(
                logits.view(-1, vocab.size), targets.view(-1), reduction="none"
            ).view_as(targets)
            loss = (loss * loss_mask).sum() / loss_mask.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            step += 1
            examples_seen += inputs.shape[0]
            pbar.set_postfix(loss=f"{loss.item():.4f}", epoch=epoch)
            pbar.update(1)

            if args.wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/examples_seen": examples_seen,
                    "train/epoch": epoch,
                    "step": step,
                })

            if step % args.eval_every == 0 or step == args.steps:
                metrics = evaluate(model, test_loader, vocab, args.device)
                train_metrics = evaluate(model, train_eval_loader, vocab, args.device)
                history["step"].append(step)
                history["train_loss"].append(loss.item())
                history["test_loss"].append(metrics["loss"])
                history["test_acc"].append(metrics["acc"])
                history["train_acc"].append(train_metrics["acc"])
                history["train_eval_loss"].append(train_metrics["loss"])
                pbar.write(
                    f"Step {step} (epoch {epoch}, {examples_seen:,} ex): "
                    f"train_loss={loss.item():.4f}, "
                    f"train_acc={train_metrics['acc']:.4f}, "
                    f"test_loss={metrics['loss']:.4f}, test_acc={metrics['acc']:.4f}"
                )
                if args.wandb:
                    log = {
                        "eval/loss": metrics["loss"],
                        "eval/acc": metrics["acc"],
                        "eval/train_loss": train_metrics["loss"],
                        "eval/train_acc": train_metrics["acc"],
                        "step": step,
                    }
                    for c, acc in metrics["per_count_acc"].items():
                        log[f"eval/acc_count_{c}"] = acc
                    wandb.log(log)

    pbar.close()

    metrics = evaluate(model, test_loader, vocab, args.device)
    print(f"\nFinal: test_loss={metrics['loss']:.4f}, test_acc={metrics['acc']:.4f}")
    print(f"Per-count accuracy: {metrics['per_count_acc']}")
    print(f"Total examples seen: {examples_seen:,}, epochs: {epoch}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "model.pt")
    config = {
        "model": model.config_dict(),
        "vocab": vocab.to_dict(),
        "training": {
            "num_symbols": args.num_symbols,
            "seq_len": args.seq_len,
            "samples_per_count": args.samples_per_count,
            "steps": args.steps,
            "seed": args.seed,
            "final_test_acc": metrics["acc"],
            "final_test_loss": metrics["loss"],
            "examples_seen": examples_seen,
            "epochs": epoch,
        },
        "puzzle": "may_2026_unique_count",
    }
    (save_dir / "config.json").write_text(json.dumps(config, indent=2))
    torch.save(history, save_dir / "history.pt")
    plot_training(history, metrics["per_count_acc"], save_dir, args)
    print(f"Saved to {save_dir}")

    if args.wandb:
        wandb.finish()

    return model, vocab, history


# ── CLI ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Train May 2026 puzzle: unique count")
    # Task
    p.add_argument("--num_symbols", type=int, default=10, help="Input vocab size V")
    p.add_argument("--seq_len", type=int, default=10, help="Sequence length N")
    p.add_argument("--samples_per_count", type=int, default=10000)
    p.add_argument("--test_frac", type=float, default=0.1)
    # Model
    p.add_argument("--d_model", type=int, default=32)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--pos_embed", type=str, default="none",
                   choices=["learned", "none"],
                   help="Positional encoding type (the released puzzle uses 'none')")
    # Training
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Save
    p.add_argument("--save_dir", type=str, default=str(Path(__file__).parent / "checkpoints"))
    # Wandb
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="mech-interp-puzzles")
    p.add_argument("--wandb_name", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args)
