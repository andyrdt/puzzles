"""Train the September 2026 set-difference puzzle.

Task
----
Choose a K-element subset X of an N-symbol vocabulary and a missing symbol
z in X. The model sees independently shuffled renderings of X and X \\ {z}:

    [BOS] shuffle(X) [SEP] shuffle(X \\ {z}) [SEP]

It must predict z autoregressively at the final separator. Train, validation,
and test splits are made over whole underlying sets X *before* any rendering,
so a held-out example cannot leak through a different choice of z or shuffle.
"""

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

# Required for deterministic CUDA matmuls. This must precede importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import AttentionOnlyTransformer


class Vocab:
    """N ordinary symbols followed by BOS and a shared separator token."""

    def __init__(self, num_symbols: int):
        self.num_symbols = num_symbols
        self.BOS = num_symbols
        self.SEP = num_symbols + 1
        self.size = num_symbols + 2

    def token_name(self, token: int) -> str:
        if 0 <= token < self.num_symbols:
            return chr(ord("a") + token) if token < 26 else f"t{token}"
        if token == self.BOS:
            return "BOS"
        if token == self.SEP:
            return "SEP"
        raise ValueError(f"unknown token id {token}")

    def to_dict(self):
        return {"type": "set_difference", "num_symbols": self.num_symbols}


def enumerate_sets(num_symbols: int, set_size: int, max_sets: int) -> torch.Tensor:
    if not 2 <= set_size < num_symbols:
        raise ValueError("set_size must satisfy 2 <= K < N")
    count = math.comb(num_symbols, set_size)
    if count > max_sets:
        raise ValueError(
            f"C({num_symbols}, {set_size})={count:,} exceeds --max_sets={max_sets:,}"
        )
    return torch.tensor(
        list(itertools.combinations(range(num_symbols), set_size)),
        dtype=torch.long,
    )


def _covers_vocabulary(sets: torch.Tensor, num_symbols: int) -> bool:
    return len(sets) > 0 and torch.unique(sets).numel() == num_symbols


def split_sets(
    all_sets: torch.Tensor,
    num_symbols: int,
    val_frac: float,
    test_frac: float,
    seed: int,
):
    """Split whole X sets, retrying until every symbol occurs in every split."""
    if val_frac <= 0 or test_frac <= 0 or val_frac + test_frac >= 1:
        raise ValueError("val_frac and test_frac must be positive and sum to < 1")

    total = len(all_sets)
    n_val = max(1, round(total * val_frac))
    n_test = max(1, round(total * test_frac))
    n_train = total - n_val - n_test
    if n_train < 1:
        raise ValueError("not enough underlying sets for the requested split")

    for attempt in range(1000):
        generator = torch.Generator().manual_seed(seed + attempt)
        order = torch.randperm(total, generator=generator)
        train_sets = all_sets[order[:n_train]]
        val_sets = all_sets[order[n_train : n_train + n_val]]
        test_sets = all_sets[order[n_train + n_val :]]
        if all(
            _covers_vocabulary(part, num_symbols)
            for part in (train_sets, val_sets, test_sets)
        ):
            return {
                "train": train_sets,
                "val": val_sets,
                "test": test_sets,
                "split_seed": seed + attempt,
            }

    raise ValueError(
        "could not make splits that each cover every symbol; use larger N/K pools "
        "or larger validation/test fractions"
    )


def split_fingerprint(sets: torch.Tensor) -> str:
    return hashlib.sha256(sets.numpy().tobytes()).hexdigest()[:16]


def load_checkpoint_splits(
    checkpoint_dir: Path,
    split_config: dict,
    num_symbols: int,
    set_size: int,
):
    """Load authoritative saved splits, with a legacy regeneration fallback."""
    split_path = checkpoint_dir / "splits.pt"
    if split_path.exists():
        raw_splits = torch.load(split_path, map_location="cpu", weights_only=True)
    else:
        all_sets = enumerate_sets(num_symbols, set_size, max_sets=10_000_000)
        raw_splits = split_sets(
            all_sets,
            num_symbols,
            split_config["val_frac"],
            split_config["test_frac"],
            split_config["requested_seed"],
        )

    splits = {}
    for name in ("train", "val", "test"):
        if name not in raw_splits:
            raise RuntimeError(f"saved splits are missing {name!r}")
        sets = raw_splits[name].detach().cpu().to(torch.long).contiguous()
        expected_count = split_config["counts"][name]
        if sets.shape != (expected_count, set_size):
            raise RuntimeError(
                f"{name} split has shape {tuple(sets.shape)}, expected "
                f"({expected_count}, {set_size})"
            )
        if sets.numel() and (
            sets.min() < 0
            or sets.max() >= num_symbols
            or (sets[:, 1:] <= sets[:, :-1]).any()
        ):
            raise RuntimeError(f"{name} split contains an invalid canonical set")
        actual_fingerprint = split_fingerprint(sets)
        expected_fingerprint = split_config["fingerprints"][name]
        if actual_fingerprint != expected_fingerprint:
            raise RuntimeError(
                f"{name} split fingerprint mismatch: "
                f"{actual_fingerprint} != {expected_fingerprint}"
            )
        splits[name] = sets

    keys = {
        name: {tuple(row.tolist()) for row in sets}
        for name, sets in splits.items()
    }
    for name, unique_sets in keys.items():
        if len(unique_sets) != len(splits[name]):
            raise RuntimeError(f"{name} split contains duplicate underlying sets")
    if not (
        keys["train"].isdisjoint(keys["val"])
        and keys["train"].isdisjoint(keys["test"])
        and keys["val"].isdisjoint(keys["test"])
    ):
        raise RuntimeError("saved whole-set splits are not pairwise disjoint")
    if len(set.union(*keys.values())) != math.comb(num_symbols, set_size):
        raise RuntimeError("saved splits do not partition the full underlying set space")
    return splits


def _row_permutations(
    values: torch.Tensor,
    generator: torch.Generator,
    permutation_table: torch.Tensor | None = None,
) -> torch.Tensor:
    if permutation_table is not None:
        indices = torch.randint(
            len(permutation_table),
            (len(values),),
            device=values.device,
            generator=generator,
        )
        return values.gather(1, permutation_table[indices])
    order = torch.rand(
        values.shape, device=values.device, generator=generator
    ).argsort(dim=1)
    return values.gather(1, order)


def build_permutation_table(length: int, device: torch.device):
    """Precompute small permutation spaces; fall back to argsort when large."""
    if math.factorial(length) > 100_000:
        return None
    return torch.tensor(
        list(itertools.permutations(range(length))),
        dtype=torch.long,
        device=device,
    )


def render_training_batch(
    set_pool: torch.Tensor,
    batch_size: int,
    vocab: Vocab,
    generator: torch.Generator,
    x_permutations: torch.Tensor | None = None,
    y_permutations: torch.Tensor | None = None,
):
    """Sample underlying problems and render both set segments on the GPU."""
    device = set_pool.device
    set_size = set_pool.shape[1]
    set_indices = torch.randint(
        len(set_pool), (batch_size,), device=device, generator=generator
    )
    sets = set_pool[set_indices]
    missing_indices = torch.randint(
        set_size, (batch_size,), device=device, generator=generator
    )
    targets = sets.gather(1, missing_indices[:, None]).squeeze(1)

    shuffled_x = _row_permutations(sets, generator, x_permutations)
    positions = torch.arange(set_size, device=device).expand(batch_size, -1)
    remaining = sets[positions != missing_indices[:, None]].view(
        batch_size, set_size - 1
    )
    shuffled_y = _row_permutations(remaining, generator, y_permutations)

    inputs = torch.full(
        (batch_size, 2 * set_size + 2),
        vocab.SEP,
        device=device,
        dtype=torch.long,
    )
    inputs[:, 0] = vocab.BOS
    inputs[:, 1 : 1 + set_size] = shuffled_x
    inputs[:, set_size + 2 : 2 * set_size + 1] = shuffled_y
    return inputs, targets


def build_eval_examples(
    sets: torch.Tensor,
    permutations: int,
    vocab: Vocab,
    seed: int,
    max_eval_sets: int | None = None,
):
    """Render every z for each selected X.

    ``permutations=0`` enumerates the full Cartesian product of X and Y orders;
    a positive value draws that many fixed independent shuffles per (X, z).
    """
    if permutations < 0:
        raise ValueError("permutations must be >= 0")
    generator = torch.Generator().manual_seed(seed)
    if max_eval_sets is not None and len(sets) > max_eval_sets:
        chosen = torch.randperm(len(sets), generator=generator)[:max_eval_sets]
        sets = sets[chosen]

    num_sets, set_size = sets.shape
    if permutations == 0:
        permutations = math.factorial(set_size) * math.factorial(set_size - 1)
        num_examples = num_sets * set_size * permutations
        if num_examples > 2_000_000:
            raise ValueError(
                f"exhaustive evaluation would create {num_examples:,} examples; "
                "sample permutations instead"
            )
        x_orders = torch.tensor(
            list(itertools.permutations(range(set_size))), dtype=torch.long
        )
        y_orders = torch.tensor(
            list(itertools.permutations(range(set_size - 1))), dtype=torch.long
        )
        problems = sets.repeat_interleave(set_size, dim=0)
        missing_indices = torch.arange(set_size).repeat(num_sets)
        targets_by_problem = problems.gather(
            1, missing_indices[:, None]
        ).squeeze(1)
        positions = torch.arange(set_size).expand(len(problems), -1)
        remaining_by_problem = problems[
            positions != missing_indices[:, None]
        ].view(len(problems), set_size - 1)

        x_order_grid = x_orders.repeat_interleave(len(y_orders), dim=0)
        y_order_grid = y_orders.repeat(len(x_orders), 1)
        expanded = problems.repeat_interleave(permutations, dim=0)
        remaining = remaining_by_problem.repeat_interleave(permutations, dim=0)
        shuffled_x = expanded.gather(
            1, x_order_grid.repeat(len(problems), 1)
        )
        shuffled_y = remaining.gather(
            1, y_order_grid.repeat(len(problems), 1)
        )
        targets = targets_by_problem.repeat_interleave(permutations)

        inputs = torch.full(
            (len(expanded), 2 * set_size + 2), vocab.SEP, dtype=torch.long
        )
        inputs[:, 0] = vocab.BOS
        inputs[:, 1 : 1 + set_size] = shuffled_x
        inputs[:, set_size + 2 : 2 * set_size + 1] = shuffled_y
        return {
            "inputs": inputs,
            "targets": targets,
            "num_sets": num_sets,
            "num_problems": num_sets * set_size,
            "permutations": permutations,
            "set_size": set_size,
            "exhaustive": True,
        }

    repeats_per_set = set_size * permutations
    expanded = sets.repeat_interleave(repeats_per_set, dim=0)
    missing_indices = (
        torch.arange(set_size)
        .repeat_interleave(permutations)
        .repeat(num_sets)
    )
    targets = expanded.gather(1, missing_indices[:, None]).squeeze(1)

    shuffled_x = _row_permutations(expanded, generator)
    positions = torch.arange(set_size).expand(len(expanded), -1)
    remaining = expanded[
        positions != missing_indices[:, None]
    ].view(len(expanded), set_size - 1)
    shuffled_y = _row_permutations(remaining, generator)

    inputs = torch.full(
        (len(expanded), 2 * set_size + 2), vocab.SEP, dtype=torch.long
    )
    inputs[:, 0] = vocab.BOS
    inputs[:, 1 : 1 + set_size] = shuffled_x
    inputs[:, set_size + 2 : 2 * set_size + 1] = shuffled_y
    return {
        "inputs": inputs,
        "targets": targets,
        "num_sets": num_sets,
        "num_problems": num_sets * set_size,
        "permutations": permutations,
        "set_size": set_size,
        "exhaustive": False,
    }


def evaluate(
    model,
    examples,
    vocab: Vocab,
    device: torch.device,
    batch_size: int,
    precision: str,
):
    model.eval()
    predictions = []
    margins = []
    total_loss = 0.0
    total = len(examples["targets"])
    use_bf16 = precision == "bf16" and device.type == "cuda"

    with torch.inference_mode():
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            inputs = examples["inputs"][start:stop].to(device)
            targets = examples["targets"][start:stop].to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                logits, _ = model(inputs)
                final_logits = logits[:, -1]
                loss = F.cross_entropy(final_logits, targets, reduction="sum")
            total_loss += loss.item()
            batch_predictions = final_logits.argmax(dim=-1)
            predictions.append(batch_predictions.cpu())
            top_values, top_indices = final_logits.topk(2, dim=-1)
            best_other = torch.where(
                top_indices[:, 0] == targets,
                top_values[:, 1],
                top_values[:, 0],
            )
            target_values = final_logits.gather(1, targets[:, None]).squeeze(1)
            margins.append((target_values - best_other).float().cpu())

    predictions = torch.cat(predictions)
    margins = torch.cat(margins)
    targets = examples["targets"]
    correct = predictions == targets
    repetitions = examples["permutations"]
    correct_by_problem = correct.view(-1, repetitions)
    pred_by_problem = predictions.view(-1, repetitions)
    set_size = examples["set_size"]
    correct_by_set = correct.view(examples["num_sets"], set_size * repetitions)

    x_values = examples["inputs"][:, 1 : 1 + set_size]
    y_values = examples["inputs"][:, set_size + 2 : 2 * set_size + 1]
    wrong = ~correct
    wrong_in_y = wrong & (predictions[:, None] == y_values).any(dim=1)
    wrong_outside_x = wrong & ~(predictions[:, None] == x_values).any(dim=1)
    num_errors = wrong.sum().item()

    token_total = torch.bincount(targets, minlength=vocab.num_symbols)
    token_correct = torch.bincount(
        targets[correct], minlength=vocab.num_symbols
    )
    per_token_acc = {
        str(token): token_correct[token].item() / token_total[token].item()
        for token in range(vocab.num_symbols)
        if token_total[token] > 0
    }
    metrics = {
        "loss": total_loss / total,
        "acc": correct.float().mean().item(),
        "strict_problem_acc": correct_by_problem.all(dim=1).float().mean().item(),
        "strict_set_acc": correct_by_set.all(dim=1).float().mean().item(),
        "permutation_consistency": (
            pred_by_problem == pred_by_problem[:, :1]
        ).all(dim=1).float().mean().item(),
        "mean_correct_margin": margins.mean().item(),
        "min_correct_margin": margins.min().item(),
        "per_token_acc": per_token_acc,
        "min_token_acc": min(per_token_acc.values()),
        "num_errors": num_errors,
        "errors_in_y": wrong_in_y.sum().item(),
        "errors_outside_x": wrong_outside_x.sum().item(),
        "num_examples": total,
        "num_sets": examples["num_sets"],
        "num_problems": examples["num_problems"],
        "permutations_per_problem": repetitions,
        "exhaustive_permutations": examples["exhaustive"],
    }
    model.train()
    return metrics


def plot_history(history, save_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    steps = history["step"]
    axes[0].plot(steps, history["train_loss"], label="batch train")
    axes[0].plot(steps, history["train_eval_loss"], label="fixed train")
    axes[0].plot(steps, history["val_loss"], label="held-out X")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Step")
    axes[0].legend()

    axes[1].plot(steps, history["train_acc"], label="train")
    axes[1].plot(steps, history["val_acc"], label="held-out X")
    axes[1].set_title("Example accuracy")
    axes[1].set_xlabel("Step")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend()

    axes[2].plot(steps, history["val_strict_problem_acc"])
    axes[2].set_title("Held-out problems: all shuffles correct")
    axes[2].set_xlabel("Step")
    axes[2].set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def set_deterministic(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)


def train(args):
    run_started = time.perf_counter()
    set_deterministic(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.precision == "bf16" and device.type != "cuda":
        raise RuntimeError("bf16 precision is currently supported only on CUDA")
    if (
        args.precision == "bf16"
        and device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("bf16 requested but unsupported by this GPU")

    vocab = Vocab(args.num_symbols)
    all_sets = enumerate_sets(args.num_symbols, args.set_size, args.max_sets)
    splits = split_sets(
        all_sets,
        args.num_symbols,
        args.val_frac,
        args.test_frac,
        args.split_seed,
    )
    train_sets_device = splits["train"].to(device)
    input_seq_len = 2 * args.set_size + 2

    train_examples = build_eval_examples(
        splits["train"],
        args.train_eval_permutations,
        vocab,
        args.seed + 101,
        args.train_eval_sets,
    )
    val_examples = build_eval_examples(
        splits["val"],
        args.eval_permutations,
        vocab,
        args.seed + 202,
    )
    test_examples = None
    if not args.skip_test:
        test_examples = build_eval_examples(
            splits["test"],
            args.test_permutations,
            vocab,
            args.seed + 303,
        )

    model = AttentionOnlyTransformer(
        vocab_size=vocab.size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        max_seq_len=input_seq_len,
        n_layers=args.n_layers,
        pos_embed_std=args.pos_embed_std,
    ).to(device)
    train_model = (
        torch.compile(model, mode=args.compile_mode) if args.compile else model
    )
    num_parameters = sum(parameter.numel() for parameter in model.parameters())

    print(
        f"Task: N={args.num_symbols}, K={args.set_size}; "
        f"C(N,K)={len(all_sets):,} underlying X sets"
    )
    print(
        "Split by whole X: "
        f"{len(splits['train']):,} train / {len(splits['val']):,} val / "
        f"{len(splits['test']):,} test (seed={splits['split_seed']})"
    )
    print(
        f"Model: {args.n_layers}L, {args.n_heads} heads, d_model={args.d_model}, "
        f"pos=learned (std={args.pos_embed_std:g}), parameters={num_parameters:,}"
    )
    print(
        f"GPU path: device={device}, batch={args.batch_size:,}, "
        f"precision={args.precision}, compile={args.compile}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.min_lr
    )
    batch_generator = torch.Generator(device=device).manual_seed(args.seed + 404)
    x_permutations = build_permutation_table(args.set_size, device)
    y_permutations = build_permutation_table(args.set_size - 1, device)
    use_bf16 = args.precision == "bf16" and device.type == "cuda"

    history = {
        "step": [],
        "train_loss": [],
        "train_eval_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_strict_problem_acc": [],
        "examples_per_second": [],
    }
    training_started = time.perf_counter()
    last_eval_time = training_started
    last_eval_examples = 0
    final_batch_loss = float("nan")
    progress = tqdm(
        range(1, args.steps + 1), desc="Training", disable=args.no_progress
    )

    for step in progress:
        inputs, targets = render_training_batch(
            train_sets_device,
            args.batch_size,
            vocab,
            batch_generator,
            x_permutations,
            y_permutations,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            logits, attention_patterns = train_model(inputs)
            prediction_loss = F.cross_entropy(logits[:, -1], targets)
            if args.attention_entropy_reg > 0:
                entropies = []
                for attention in attention_patterns:
                    final_attention = attention[:, :, -1].float().clamp_min(1e-12)
                    entropies.append(
                        -(final_attention * final_attention.log()).sum(dim=-1).mean()
                    )
                attention_entropy = torch.stack(entropies).mean()
                loss = prediction_loss - args.attention_entropy_reg * attention_entropy
            else:
                loss = prediction_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        final_batch_loss = prediction_loss.item()
        progress.set_postfix(loss=f"{final_batch_loss:.4f}")

        if step % args.eval_every == 0 or step == args.steps:
            if device.type == "cuda":
                torch.cuda.synchronize()
            now = time.perf_counter()
            examples_seen = step * args.batch_size
            throughput = (examples_seen - last_eval_examples) / (now - last_eval_time)
            train_metrics = evaluate(
                model,
                train_examples,
                vocab,
                device,
                args.eval_batch_size,
                args.precision,
            )
            val_metrics = evaluate(
                model,
                val_examples,
                vocab,
                device,
                args.eval_batch_size,
                args.precision,
            )
            history["step"].append(step)
            history["train_loss"].append(final_batch_loss)
            history["train_eval_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["acc"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["acc"])
            history["val_strict_problem_acc"].append(
                val_metrics["strict_problem_acc"]
            )
            history["examples_per_second"].append(throughput)
            progress.write(
                f"step {step:6d}: train={train_metrics['acc']:.4f}, "
                f"heldout-X={val_metrics['acc']:.4f}, "
                f"strict={val_metrics['strict_problem_acc']:.4f}, "
                f"{throughput:,.0f} examples/s"
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            last_eval_examples = examples_seen
            last_eval_time = time.perf_counter()

    training_loop_seconds = time.perf_counter() - training_started
    final_train_metrics = evaluate(
        model, train_examples, vocab, device, args.eval_batch_size, args.precision
    )
    final_val_metrics = evaluate(
        model, val_examples, vocab, device, args.eval_batch_size, args.precision
    )
    final_test_metrics = None
    if test_examples is not None:
        final_test_metrics = evaluate(
            model, test_examples, vocab, device, args.eval_batch_size, args.precision
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_and_evaluation_seconds = time.perf_counter() - run_started

    max_memory_gb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        if device.type == "cuda"
        else 0.0
    )
    results = {
        "train_sample": final_train_metrics,
        "validation_heldout_x": final_val_metrics,
        "test_heldout_x": final_test_metrics,
        "training_loop_seconds": training_loop_seconds,
        "training_and_evaluation_seconds": training_and_evaluation_seconds,
        "examples_seen": args.steps * args.batch_size,
        "training_examples_per_second_including_periodic_eval": (
            args.steps * args.batch_size / training_loop_seconds
        ),
        "max_cuda_memory_gb": max_memory_gb,
    }

    print("Final train sample:", json.dumps(final_train_metrics, indent=2))
    print("Final held-out-X validation:", json.dumps(final_val_metrics, indent=2))
    if final_test_metrics is not None:
        print("Final held-out-X test:", json.dumps(final_test_metrics, indent=2))
    print(
        f"Training loop {training_loop_seconds:.1f}s; training + all evaluation "
        f"{training_and_evaluation_seconds:.1f}s; peak CUDA memory "
        f"{max_memory_gb:.2f} GiB"
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model": model.config_dict(),
        "vocab": vocab.to_dict(),
        "task": {
            "name": "set_difference",
            "set_size": args.set_size,
            "input_format": "[BOS] shuffle(X) [SEP] shuffle(X_without_z) [SEP]",
            "target": "z",
        },
        "splits": {
            "unit": "whole underlying X set",
            "requested_seed": args.split_seed,
            "actual_seed": splits["split_seed"],
            "val_frac": args.val_frac,
            "test_frac": args.test_frac,
            "counts": {name: len(splits[name]) for name in ("train", "val", "test")},
            "fingerprints": {
                name: split_fingerprint(splits[name])
                for name in ("train", "val", "test")
            },
            "pairwise_disjoint": True,
        },
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "min_lr": args.min_lr,
            "weight_decay": args.weight_decay,
            "attention_entropy_reg": args.attention_entropy_reg,
            "seed": args.seed,
            "precision": args.precision,
            "compile": args.compile,
            "compile_mode": args.compile_mode if args.compile else None,
            "eval_every": args.eval_every,
            "grad_clip": args.grad_clip,
            "examples_seen": args.steps * args.batch_size,
        },
        "evaluation": {
            "train_sample_sets": args.train_eval_sets,
            "train_sample_permutations": args.train_eval_permutations,
            "validation_permutations": args.eval_permutations,
            "test_permutations": args.test_permutations,
            "batch_size": args.eval_batch_size,
            "test_skipped": args.skip_test,
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "tf32_matmul_enabled": torch.backends.cuda.matmul.allow_tf32,
        },
        "results": results,
        "puzzle": "september_2026_set_difference",
    }
    torch.save(model.state_dict(), save_dir / "model.pt")
    torch.save(history, save_dir / "history.pt")
    torch.save(
        {name: splits[name] for name in ("train", "val", "test")},
        save_dir / "splits.pt",
    )
    (save_dir / "config.json").write_text(json.dumps(config, indent=2))
    (save_dir / "results.json").write_text(json.dumps(results, indent=2))
    plot_history(history, save_dir / "training.png")
    print(f"Saved artifacts to {save_dir}")
    return model, vocab, history, results


def get_args():
    parser = argparse.ArgumentParser(
        description="Train September 2026 puzzle: set difference"
    )
    parser.add_argument("--num_symbols", type=int, default=16, help="vocabulary size N")
    parser.add_argument("--set_size", type=int, default=4, help="set size K")
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--split_seed", type=int, default=1729)
    parser.add_argument("--max_sets", type=int, default=1_000_000)

    parser.add_argument("--d_model", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--pos_embed_std", type=float, default=0.02)

    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument(
        "--attention_entropy_reg",
        type=float,
        default=0.0,
        help="training-only reward for broad final-query attention",
    )
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--eval_every", type=int, default=5000)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument(
        "--compile_mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="default",
    )

    parser.add_argument(
        "--train_eval_permutations", type=int, default=16,
        help="fixed shuffles per train (X,z); 0 means exhaustive",
    )
    parser.add_argument(
        "--eval_permutations", type=int, default=0,
        help="fixed shuffles per validation (X,z); 0 means exhaustive",
    )
    parser.add_argument(
        "--test_permutations", type=int, default=0,
        help="fixed shuffles per test (X,z); 0 means exhaustive",
    )
    parser.add_argument("--train_eval_sets", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=32768)
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument(
        "--run_test",
        dest="skip_test",
        action="store_false",
        help="run the final held-out test audit after training",
    )
    test_group.add_argument(
        "--skip_test",
        dest="skip_test",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(skip_test=True)
    parser.add_argument(
        "--save_dir",
        default=str(Path(__file__).parent / "checkpoints"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(get_args())
