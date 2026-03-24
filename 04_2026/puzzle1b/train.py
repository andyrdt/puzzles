"""
Puzzle 1b: Max of List — 2-Layer Attention-Only Transformer, Digit-Level Tokenization

Task: Given a list of numbers (0-99), predict the maximum.
Each number is tokenized as two digits (tens, ones), always zero-padded.

Input:  [BOS] d1_tens d1_ones [SEP] d2_tens d2_ones [SEP] ... dk_tens dk_ones [ANS]
Target: max_tens max_ones [EOS]

Vocabulary: digits 0-9, plus BOS, SEP, ANS, EOS (14 tokens total).
Model: 2-layer, attention-only (no MLP), causal transformer.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import AttentionOnlyTransformer


# ── Vocab ───────────────────────────────────────────────────────────────────

class Vocab:
    """Tokens 0-9: digits. 10: BOS, 11: SEP, 12: ANS, 13: EOS."""

    BOS = 10
    SEP = 11
    ANS = 12
    EOS = 13
    size = 14

    @staticmethod
    def number_to_digits(n):
        return n // 10, n % 10

    @staticmethod
    def digits_to_number(tens, ones):
        return tens * 10 + ones

    @staticmethod
    def token_name(tok: int) -> str:
        if 0 <= tok <= 9:
            return str(tok)
        return {10: "BOS", 11: "SEP", 12: "ANS", 13: "EOS"}[tok]

    @staticmethod
    def to_dict():
        return {"type": "digit", "num_range": 100}


# ── Dataset ─────────────────────────────────────────────────────────────────

class MaxOfListDigitsDataset(Dataset):
    """
    Sequence: BOS d1t d1o SEP d2t d2o SEP ... dkt dko ANS mt mo EOS
    Length:   1 + 2k + (k-1) + 1 + 2 + 1 = 3k + 4

    Loss at: ANS (predict mt), mt (predict mo), mo (predict EOS).
    """

    def __init__(self, vocab, list_len: int, numbers: np.ndarray):
        self.vocab = vocab
        self.list_len = list_len
        self.size = len(numbers)
        maxes = numbers.max(axis=1)

        seq_len = 3 * list_len + 4
        seqs = np.full((self.size, seq_len), vocab.SEP, dtype=np.int64)

        for i in range(self.size):
            pos = 0
            seqs[i, pos] = vocab.BOS; pos += 1
            for j in range(list_len):
                tens, ones = vocab.number_to_digits(numbers[i, j])
                seqs[i, pos] = tens; pos += 1
                seqs[i, pos] = ones; pos += 1
                if j < list_len - 1:
                    seqs[i, pos] = vocab.SEP; pos += 1
            seqs[i, pos] = vocab.ANS; pos += 1
            mt, mo = vocab.number_to_digits(maxes[i])
            seqs[i, pos] = mt; pos += 1
            seqs[i, pos] = mo; pos += 1
            seqs[i, pos] = vocab.EOS

        self.inputs = torch.tensor(seqs[:, :-1])
        self.targets = torch.tensor(seqs[:, 1:])

        self.loss_mask = torch.zeros_like(self.targets, dtype=torch.bool)
        ans_pos = 3 * list_len
        self.loss_mask[:, ans_pos] = True
        self.loss_mask[:, ans_pos + 1] = True
        self.loss_mask[:, ans_pos + 2] = True

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.loss_mask[idx]


# ── Data generation ─────────────────────────────────────────────────────────

def generate_data(num_range, list_len, min_per_value, seed):
    rng = np.random.default_rng(seed)

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

    random_data = rng.integers(0, num_range, size=(100_000, list_len))
    _, unique_idx = np.unique(random_data, axis=0, return_index=True)
    random_data = random_data[np.sort(unique_idx)]

    all_numbers = np.concatenate([stratified, random_data], axis=0)
    rng.shuffle(all_numbers)

    n_test = min(10_000, len(all_numbers) // 10)
    train_numbers = all_numbers[:-n_test]
    test_numbers = all_numbers[-n_test:]
    return train_numbers, test_numbers


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(model, loader, vocab, device, list_len):
    model.eval()
    total_loss = 0.0
    total_count = 0
    total_correct = 0
    total_tens_correct = 0
    total_ones_correct = 0
    n_samples = 0

    ans_pos = 3 * list_len

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

            pred_tens = logits[:, ans_pos].argmax(dim=-1)
            pred_ones = logits[:, ans_pos + 1].argmax(dim=-1)
            true_tens = targets[:, ans_pos]
            true_ones = targets[:, ans_pos + 1]

            tens_ok = pred_tens == true_tens
            ones_ok = pred_ones == true_ones
            total_correct += (tens_ok & ones_ok).sum().item()
            total_tens_correct += tens_ok.sum().item()
            total_ones_correct += ones_ok.sum().item()
            n_samples += inputs.shape[0]

    model.train()
    return {
        "loss": total_loss / total_count,
        "acc": total_correct / n_samples,
        "acc_tens": total_tens_correct / n_samples,
        "acc_ones": total_ones_correct / n_samples,
    }


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_training(history, save_dir, args):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["step"], history["train_loss"], label="train")
    axes[0].plot(history["step"], history["test_loss"], label="test")
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Loss"); axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history["step"], history["test_acc"], label="both digits")
    axes[1].set_xlabel("Step"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Test Accuracy"); axes[1].set_ylim(0, 1.05); axes[1].legend()

    axes[2].plot(history["step"], history["test_acc_tens"], label="tens digit")
    axes[2].plot(history["step"], history["test_acc_ones"], label="ones digit")
    axes[2].set_xlabel("Step"); axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Per-Digit Accuracy"); axes[2].set_ylim(0, 1.05); axes[2].legend()

    plt.suptitle(
        f"Puzzle 1b: Max of List (0-{args.num_range-1}, digit tokenization, {args.n_layers}L)"
    )
    plt.tight_layout()
    plt.savefig(save_dir / "training.png", dpi=150)
    plt.close()


# ── Training ────────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    vocab = Vocab()
    input_seq_len = 3 * args.list_len + 3  # full seq is 3k+4, input drops last

    print(f"Vocab size: {vocab.size} (digits 0-9 + BOS/SEP/ANS/EOS)")
    print(f"Number range: 0-{args.num_range - 1}")
    print(f"Sequence length (input): {input_seq_len}")
    print(f"Model: {args.n_layers}L attention-only, d_model={args.d_model}, n_heads={args.n_heads}")

    # Data
    train_numbers, test_numbers = generate_data(
        args.num_range, args.list_len, args.min_per_value, args.seed
    )
    n_train, n_test = len(train_numbers), len(test_numbers)
    print(f"Data: {n_train} train, {n_test} test")

    train_ds = MaxOfListDigitsDataset(vocab, args.list_len, train_numbers)
    test_ds = MaxOfListDigitsDataset(vocab, args.list_len, test_numbers)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = AttentionOnlyTransformer(
        vocab_size=vocab.size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        max_seq_len=input_seq_len,
        n_layers=args.n_layers,
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Wandb
    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"puzzle1b_{args.n_layers}L_range{args.num_range}_len{args.list_len}",
            config={
                "puzzle": "1b",
                "task": "max_of_list_digits",
                "num_range": args.num_range,
                "list_len": args.list_len,
                "n_layers": args.n_layers,
                "d_model": args.d_model,
                "n_heads": args.n_heads,
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
    history = {
        "step": [], "train_loss": [], "test_loss": [],
        "test_acc": [], "test_acc_tens": [], "test_acc_ones": [],
    }
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
                metrics = evaluate(model, test_loader, vocab, args.device, args.list_len)
                history["step"].append(step)
                history["train_loss"].append(loss.item())
                history["test_loss"].append(metrics["loss"])
                history["test_acc"].append(metrics["acc"])
                history["test_acc_tens"].append(metrics["acc_tens"])
                history["test_acc_ones"].append(metrics["acc_ones"])
                pbar.write(
                    f"Step {step} (epoch {epoch}, {examples_seen:,} examples): "
                    f"train_loss={loss.item():.4f}, test_loss={metrics['loss']:.4f}, "
                    f"test_acc={metrics['acc']:.4f} "
                    f"(tens={metrics['acc_tens']:.4f}, ones={metrics['acc_ones']:.4f})"
                )
                if args.wandb:
                    wandb.log({
                        "eval/loss": metrics["loss"],
                        "eval/acc": metrics["acc"],
                        "eval/acc_tens": metrics["acc_tens"],
                        "eval/acc_ones": metrics["acc_ones"],
                        "step": step,
                    })

    pbar.close()

    # Final eval
    metrics = evaluate(model, test_loader, vocab, args.device, args.list_len)
    print(f"\nFinal: test_loss={metrics['loss']:.4f}, test_acc={metrics['acc']:.4f}")
    print(f"  tens_acc={metrics['acc_tens']:.4f}, ones_acc={metrics['acc_ones']:.4f}")
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
            "n_layers": args.n_layers,
            "steps": args.steps,
            "seed": args.seed,
            "final_test_acc": metrics["acc"],
            "final_test_loss": metrics["loss"],
            "examples_seen": examples_seen,
            "epochs": epoch,
        },
        "puzzle": "1b",
    }
    (save_dir / "config.json").write_text(json.dumps(config, indent=2))
    torch.save(history, save_dir / "history.pt")
    plot_training(history, save_dir, args)
    print(f"Saved to {save_dir}")

    if args.wandb:
        wandb.finish()

    return model, vocab, history


# ── CLI ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Train Puzzle 1b: Max of List (digit tokenization)")
    # Task
    p.add_argument("--num_range", type=int, default=100)
    p.add_argument("--list_len", type=int, default=5)
    p.add_argument("--min_per_value", type=int, default=100)
    # Model
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
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
