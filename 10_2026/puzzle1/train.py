"""October 2026, puzzle 1: train the released model (defaults = released configuration).

    [BOS] x1 .. xK [SEP] y1 .. yK [EOS],   Y = random permutation of X

Each scored position is trained with soft-target cross-entropy against the
uniform distribution over the symbols of X not yet seen in Y (EOS once Y is
complete). See task.py for the exact protocol.

Data is generated fresh every step from the training sets; validation uses a
fixed batch covering every validation set in several random orderings.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Transformer
from task import Sampler, Vocab, build_batch, position_stats, seq_len_for, split_sets, summarize


def fixed_eval_batch(sets_by_k, vocab, k_max, orderings, device, seed):
    """Every set in `sets_by_k`, each in `orderings` random (X order, Y order) pairs."""
    members = torch.cat([sets_by_k[k] for k in sorted(sets_by_k)]).repeat(orderings, 1).to(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    return build_batch(members, vocab, k_max, generator=gen)


def soft_ce_loss(logits, targets, scored):
    """Mean soft-target cross-entropy over scored positions (mask-weighted: static shapes for compile)."""
    ce = -(targets * logits.log_softmax(-1)).sum(-1)
    return (ce * scored).sum() / scored.sum()


class BatchStream:
    """Generates `chunk` training batches per call to amortize the many small data-gen kernels."""

    def __init__(self, sampler, vocab, k_max, batch_size, chunk):
        self.sampler, self.vocab, self.k_max = sampler, vocab, k_max
        self.batch_size, self.chunk = batch_size, chunk
        self.buffer, self.i = None, chunk

    def next(self):
        if self.i == self.chunk:
            members = self.sampler.sample_members(self.batch_size * self.chunk)
            self.buffer = build_batch(members, self.vocab, self.k_max, generator=self.sampler.gen)
            self.i = 0
        sl = slice(self.i * self.batch_size, (self.i + 1) * self.batch_size)
        self.i += 1
        return {k: v[sl] for k, v in self.buffer.items()}


@torch.no_grad()
def evaluate(model, batch, vocab, prefix, chunk=8192):
    model.eval()
    parts = []
    for i in range(0, len(batch["tokens"]), chunk):
        logits, _ = model(batch["tokens"][i : i + chunk])
        parts.append({k: v for k, v in position_stats(logits, {"targets": batch["targets"][i : i + chunk]}).items()})
    model.train()
    stats = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    return summarize(stats, batch, vocab, prefix)


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress)))


def param_norms(model):
    out = {"weights/total_norm": math.sqrt(sum(p.pow(2).sum().item() for p in model.parameters()))}
    for name, p in model.named_parameters():
        out[f"weights/{name}"] = p.norm().item()
    return out


def attention_figure(model, vocab, k_max, device):
    """Attention patterns for one fixed example (every layer/head)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = min(6, k_max)
    members = torch.zeros(1, vocab.num_symbols, dtype=torch.bool, device=device)
    members[0, [1, 4, 7, 9, 12, 14][:k]] = True
    gen = torch.Generator(device=device).manual_seed(0)
    batch = build_batch(members, vocab, k_max, generator=gen)
    T = 2 * k + 3
    tokens = batch["tokens"][:, :T]
    model.eval()
    with torch.no_grad():
        _, patterns = model(tokens)
    model.train()
    labels = vocab.render(tokens[0]).split()
    L, H = len(patterns), patterns[0].shape[1]
    fig, axes = plt.subplots(L, H, figsize=(2.6 * H, 2.6 * L), squeeze=False)
    for l in range(L):
        for h in range(H):
            ax = axes[l][h]
            ax.imshow(patterns[l][0, h].cpu().numpy(), vmin=0, vmax=1, cmap="Blues")
            ax.set_xticks(range(T), labels, fontsize=5, rotation=90)
            ax.set_yticks(range(T), labels, fontsize=5)
            ax.set_title(f"L{l}H{h}", fontsize=8)
    fig.tight_layout()
    return fig


def run_name(args):
    return f"L{args.n_layers}_d{args.d_model}_h{args.n_heads}_wd{args.weight_decay}_lr{args.lr}_s{args.seed}"


def train(args):
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")  # TF32 matmuls
    np.random.seed(args.seed)
    device = args.device

    vocab = Vocab(args.num_symbols)
    sets = split_sets(args.num_symbols, args.k_min, args.k_max, args.val_frac, args.test_frac, args.split_seed)
    sampler = Sampler(sets["train"], device, args.seed)
    val_batch = fixed_eval_batch(sets["val"], vocab, args.k_max, args.eval_orderings, device, seed=1)
    train_eval_members = Sampler(sets["train"], device, seed=12345).sample_members(len(val_batch["tokens"]))
    train_eval_batch = build_batch(train_eval_members, vocab, args.k_max,
                                   generator=torch.Generator(device=device).manual_seed(2))

    model = Transformer(
        vocab_size=vocab.size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=seq_len_for(args.k_max),
        rope_base=args.rope_base,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    name = args.wandb_name or run_name(args)
    save_dir = Path(args.save_dir) / name
    save_dir.mkdir(parents=True, exist_ok=True)

    n_sets = {s: sum(len(v) for v in sets[s].values()) for s in sets}
    print(f"{name}: {n_params:,} params | sets train/val/test = "
          f"{n_sets['train']}/{n_sets['val']}/{n_sets['test']} | val seqs {len(val_batch['tokens'])}")

    config = {
        "puzzle": "oct_2026_permutation",
        "model": model.config_dict(),
        "task": {
            "num_symbols": args.num_symbols, "k_min": args.k_min, "k_max": args.k_max,
            "val_frac": args.val_frac, "test_frac": args.test_frac, "split_seed": args.split_seed,
        },
        "training": {k: v for k, v in vars(args).items()
                     if k not in ("device", "save_dir") and not k.startswith("wandb")},
        "n_params": n_params,
    }

    wandb = None
    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group,
            name=name, tags=args.wandb_tags, config={**vars(args), "n_params": n_params,
                                                    **{f"n_sets_{s}": n for s, n in n_sets.items()}},
            dir=str(save_dir),
        )
        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")

    # Standard practice: no weight decay on norm gains (1-D params).
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2),
        fused=device.startswith("cuda"),
    )

    def forward_loss(tokens, targets, scored):
        logits, _ = model(tokens)
        return soft_ce_loss(logits, targets, scored), logits

    train_step_fn = torch.compile(forward_loss) if args.compile else forward_loss
    stream = BatchStream(sampler, vocab, args.k_max, args.batch_size, args.data_chunk)

    history = []
    t0 = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            metrics = {"step": step, "examples_seen": step * args.batch_size}
            metrics.update(evaluate(model, val_batch, vocab, "val/"))
            metrics.update(evaluate(model, train_eval_batch, vocab, "train_eval/"))
            metrics.update(param_norms(model))
            metrics["time/elapsed_s"] = time.time() - t0
            history.append(metrics)
            print(f"step {step:>6} | val kl {metrics['val/kl']:.5f} seq_acc {metrics['val/seq_acc']:.4f} "
                  f"min_margin {metrics['val/min_margin']:+.3f} | train kl {metrics['train_eval/kl']:.5f} "
                  f"| |W| {metrics['weights/total_norm']:.1f} | {metrics['time/elapsed_s']:.0f}s", flush=True)
            if wandb:
                wandb.log(metrics)
            if args.save_every and step % args.save_every == 0:
                (save_dir / "snapshots").mkdir(exist_ok=True)
                torch.save(model.state_dict(), save_dir / "snapshots" / f"step{step:07d}.pt")
            if step == args.steps:
                break

        lr = lr_at(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch = stream.next()
        loss, logits = train_step_fn(batch["tokens"], batch["targets"], batch["scored"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip or float("inf"))
        optimizer.step()

        if wandb and step % args.log_every == 0:
            with torch.no_grad():
                stats = position_stats(logits.detach(), batch)
                kl = stats["kl"][batch["scored"]].mean().item()
            wandb.log({"step": step, "train/ce": loss.item(), "train/kl": kl,
                       "train/grad_norm": grad_norm.item(), "train/lr": lr})

    final = history[-1]
    torch.save(model.state_dict(), save_dir / "model.pt")
    (save_dir / "config.json").write_text(json.dumps(config, indent=2))
    (save_dir / "history.json").write_text(json.dumps(history))
    (save_dir / "final_metrics.json").write_text(json.dumps(final, indent=2))
    fig = attention_figure(model, vocab, args.k_max, device)
    fig.savefig(save_dir / "attention.png", dpi=150)
    if wandb:
        wandb.log({"step": args.steps, "attention": wandb.Image(fig)})
        for key in ("val/kl", "val/seq_acc", "val/min_margin", "val/n_seq_errors", "val/max_tv"):
            wandb.run.summary[f"final_{key}"] = final[key]
        wandb.finish()
    print(f"saved {save_dir}")
    return final


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Task
    p.add_argument("--num_symbols", type=int, default=16)
    p.add_argument("--k_min", type=int, default=2)
    p.add_argument("--k_max", type=int, default=8)
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--test_frac", type=float, default=0.10)
    p.add_argument("--split_seed", type=int, default=0)
    # Model
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=1)
    p.add_argument("--rope_base", type=float, default=10000.0)
    # Optimization
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr_frac", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.98)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                   help="torch.compile the training forward + loss")
    p.add_argument("--data_chunk", type=int, default=32, help="training batches generated per data-gen call")
    # Eval / logging
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--eval_orderings", type=int, default=8)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--save_every", type=int, default=0, help="save weight snapshots every N steps (0 = off)")
    p.add_argument("--save_dir", type=str, default=str(Path(__file__).parent / "checkpoints"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="mech-interp-puzzles")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default=None)
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--wandb_tags", type=str, nargs="*", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    train(get_args())
