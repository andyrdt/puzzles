"""
Puzzle 1a: Max of List — 1-Layer Attention-Only Transformer

Task: Given a list of numbers (0-9), predict the maximum.
Input:  [BOS] n1 [SEP] n2 [SEP] ... nk [ANS]
Target: max [EOS]

Vocabulary: numbers 0..num_range-1, plus BOS, SEP, ANS, EOS.
Model: 1-layer, attention-only (no MLP), causal transformer.
"""

import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow importing from parent directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import AttentionOnlyTransformer


# ── Vocab ───────────────────────────────────────────────────────────────────

class Vocab:
    def __init__(self, num_range: int):
        self.num_range = num_range
        self.BOS = num_range
        self.SEP = num_range + 1
        self.ANS = num_range + 2
        self.EOS = num_range + 3
        self.size = num_range + 4

    def token_name(self, tok: int) -> str:
        if 0 <= tok < self.num_range:
            return str(tok)
        return {self.BOS: "BOS", self.SEP: "SEP", self.ANS: "ANS", self.EOS: "EOS"}[tok]

    def to_dict(self):
        return {"type": "number", "num_range": self.num_range}


# ── Dataset ─────────────────────────────────────────────────────────────────

class MaxOfListDataset(Dataset):
    """
    Sequence: BOS n1 SEP n2 SEP ... nk ANS max EOS
    Length:   1 + k + (k-1) + 1 + 1 + 1 = 2k + 3

    Loss only at ANS position (predict max) and max position (predict EOS).
    """

    def __init__(self, vocab: Vocab, list_len: int, numbers: np.ndarray):
        self.vocab = vocab
        self.list_len = list_len
        self.size = len(numbers)
        maxes = numbers.max(axis=1)

        seq_len = 2 * list_len + 3
        seqs = np.full((self.size, seq_len), vocab.SEP, dtype=np.int64)

        for i in range(self.size):
            pos = 0
            seqs[i, pos] = vocab.BOS; pos += 1
            for j in range(list_len):
                seqs[i, pos] = numbers[i, j]; pos += 1
                if j < list_len - 1:
                    seqs[i, pos] = vocab.SEP; pos += 1
            seqs[i, pos] = vocab.ANS; pos += 1
            seqs[i, pos] = maxes[i]; pos += 1
            seqs[i, pos] = vocab.EOS

        self.inputs = torch.tensor(seqs[:, :-1])
        self.targets = torch.tensor(seqs[:, 1:])

        self.loss_mask = torch.zeros_like(self.targets, dtype=torch.bool)
        ans_pos = 2 * list_len
        self.loss_mask[:, ans_pos] = True
        self.loss_mask[:, ans_pos + 1] = True

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.loss_mask[idx]


# ── Data generation ─────────────────────────────────────────────────────────

def generate_data(num_range, list_len, min_per_value, seed):
    rng = np.random.default_rng(seed)

    # Stratified: for each max value v, generate examples where max=v
    stratified = []
    for v in range(num_range):
        candidates = set()
        for _ in range(min_per_value * 10):
            row = rng.integers(0, v + 1, size=list_len)
            row[rng.integers(list_len)] = v
            candidates.add(tuple(row))
        candidates = [list(c) for c in candidates]
        if len(candidates) < min_per_value:
            reps = (min_per_value // len(candidates)) + 1
            candidates = (candidates * reps)[:min_per_value]
        else:
            candidates = candidates[:min_per_value]
        stratified.extend(candidates)
    stratified = np.array(stratified)

    # Random data
    random_data = rng.integers(0, num_range, size=(55_000, list_len))
    _, unique_idx = np.unique(random_data, axis=0, return_index=True)
    random_data = random_data[np.sort(unique_idx)]

    all_numbers = np.concatenate([stratified, random_data], axis=0)
    rng.shuffle(all_numbers)

    n_test = min(5_000, len(all_numbers) // 10)
    train_numbers = all_numbers[:-n_test]
    test_numbers = all_numbers[-n_test:]
    return train_numbers, test_numbers


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(model, loader, vocab, device, list_len):
    model.eval()
    total_loss = 0.0
    total_count = 0
    total_correct = 0
    n_samples = 0
    per_value_correct = {v: 0 for v in range(vocab.num_range)}
    per_value_total = {v: 0 for v in range(vocab.num_range)}

    ans_idx = 2 * list_len

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
            true_max = targets[:, ans_idx]
            correct = preds == true_max
            total_correct += correct.sum().item()
            n_samples += inputs.shape[0]

            for v in range(vocab.num_range):
                mask_v = true_max == v
                per_value_correct[v] += (correct & mask_v).sum().item()
                per_value_total[v] += mask_v.sum().item()

    model.train()
    per_value_acc = {}
    for v in range(vocab.num_range):
        if per_value_total[v] > 0:
            per_value_acc[v] = per_value_correct[v] / per_value_total[v]
    return {
        "loss": total_loss / total_count,
        "acc": total_correct / n_samples,
        "per_value_acc": per_value_acc,
    }


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_training(history, per_value_acc, save_dir, args):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["step"], history["train_loss"], label="train")
    axes[0].plot(history["step"], history["test_loss"], label="test")
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Loss"); axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history["step"], history["test_acc"])
    axes[1].set_xlabel("Step"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Test Accuracy"); axes[1].set_ylim(0, 1.05)

    vals = sorted(per_value_acc.keys())
    accs = [per_value_acc[v] for v in vals]
    axes[2].bar(vals, accs)
    axes[2].set_xlabel("Max Value"); axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Accuracy by Max Value"); axes[2].set_ylim(0, 1.05)

    plt.suptitle(f"Puzzle 1a: Max of List (range=0-{args.num_range-1}, list_len={args.list_len})")
    plt.tight_layout()
    plt.savefig(save_dir / "training.png", dpi=150)
    plt.close()


# ── Training ────────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    vocab = Vocab(args.num_range)
    input_seq_len = 2 * args.list_len + 2  # full seq is 2k+3, input drops last

    print(f"Vocab size: {vocab.size} (numbers 0-{args.num_range-1} + BOS/SEP/ANS/EOS)")
    print(f"Sequence length (input): {input_seq_len}")
    print(f"Model: 1L attention-only, d_model={args.d_model}, n_heads={args.n_heads}")

    # Data
    train_numbers, test_numbers = generate_data(
        args.num_range, args.list_len, args.min_per_value, args.seed
    )
    n_train, n_test = len(train_numbers), len(test_numbers)
    print(f"Data: {n_train} train, {n_test} test")

    train_ds = MaxOfListDataset(vocab, args.list_len, train_numbers)
    test_ds = MaxOfListDataset(vocab, args.list_len, test_numbers)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = AttentionOnlyTransformer(
        vocab_size=vocab.size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        max_seq_len=input_seq_len,
        n_layers=1,
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Wandb
    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"puzzle1a_range{args.num_range}_len{args.list_len}",
            config={
                "puzzle": "1a",
                "task": "max_of_list",
                "num_range": args.num_range,
                "list_len": args.list_len,
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "n_layers": 1,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "steps": args.steps,
                "seed": args.seed,
                "n_params": n_params,
                "n_train": n_train,
                "n_test": n_test,
                "vocab_size": vocab.size,
                "input_seq_len": input_seq_len,
            },
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    step = 0
    examples_seen = 0
    epoch = 0
    history = {"step": [], "train_loss": [], "test_loss": [], "test_acc": []}
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

            # Per-step wandb logging
            if args.wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/examples_seen": examples_seen,
                    "train/epoch": epoch,
                    "step": step,
                })

            # Eval
            if step % args.eval_every == 0 or step == args.steps:
                metrics = evaluate(model, test_loader, vocab, args.device, args.list_len)
                history["step"].append(step)
                history["train_loss"].append(loss.item())
                history["test_loss"].append(metrics["loss"])
                history["test_acc"].append(metrics["acc"])
                pbar.write(
                    f"Step {step} (epoch {epoch}, {examples_seen:,} examples): "
                    f"train_loss={loss.item():.4f}, test_loss={metrics['loss']:.4f}, "
                    f"test_acc={metrics['acc']:.4f}"
                )
                if args.wandb:
                    log = {
                        "eval/loss": metrics["loss"],
                        "eval/acc": metrics["acc"],
                        "step": step,
                    }
                    for v, acc in metrics["per_value_acc"].items():
                        log[f"eval/acc_max_{v}"] = acc
                    wandb.log(log)

    pbar.close()

    # Final eval
    metrics = evaluate(model, test_loader, vocab, args.device, args.list_len)
    print(f"\nFinal: test_loss={metrics['loss']:.4f}, test_acc={metrics['acc']:.4f}")
    print(f"Per-value accuracy: {metrics['per_value_acc']}")
    print(f"Total examples seen: {examples_seen:,}, epochs: {epoch}")

    # Save
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "model.pt")
    config = {
        "model": model.config_dict(),
        "vocab": vocab.to_dict(),
        "training": {
            "list_len": args.list_len,
            "num_range": args.num_range,
            "steps": args.steps,
            "seed": args.seed,
            "final_test_acc": metrics["acc"],
            "final_test_loss": metrics["loss"],
            "examples_seen": examples_seen,
            "epochs": epoch,
        },
        "puzzle": "1a",
    }
    (save_dir / "config.json").write_text(json.dumps(config, indent=2))
    torch.save(history, save_dir / "history.pt")
    plot_training(history, metrics["per_value_acc"], save_dir, args)
    print(f"Saved to {save_dir}")

    if args.wandb:
        wandb.finish()

    return model, vocab, history


# ── CLI ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Train Puzzle 1a: Max of List")
    # Task
    p.add_argument("--num_range", type=int, default=10)
    p.add_argument("--list_len", type=int, default=5)
    p.add_argument("--min_per_value", type=int, default=200)
    # Model
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=4)
    # Training
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Save
    p.add_argument("--save_dir", type=str, default=str(Path(__file__).parent / "checkpoints"))
    # Wandb
    p.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    p.add_argument("--wandb_project", type=str, default="mech-interp-puzzles")
    p.add_argument("--wandb_name", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args)
