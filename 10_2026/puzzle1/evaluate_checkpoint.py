"""Evaluate a checkpoint on every set of a split, in many orderings.

Orderings per set: `--orderings` random (X order, Y order) pairs, plus structured
ones: Y in the same order as X, Y reversed, and both sorted / reverse-sorted.

    python 10_2026/puzzle1/evaluate_checkpoint.py 10_2026/puzzle1/checkpoints/<run> --split test
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Transformer
from task import Vocab, build_batch, position_stats, split_sets, summarize


def structured_orders(B, V, device, gen):
    """Yield (x_keys, y_keys) pairs for structured orderings."""
    ramp = torch.arange(V, device=device, dtype=torch.float).expand(B, V)
    rand = torch.rand(B, V, device=device, generator=gen)
    yield "sorted_same", ramp, ramp
    yield "sorted_reversed", ramp, -ramp
    yield "same_order", rand, rand
    yield "reversed_order", rand, -rand


@torch.no_grad()
def run(model, members, vocab, k_max, x_keys, y_keys, chunk=32768):
    stats_parts, batch_parts = [], []
    for i in range(0, len(members), chunk):
        b = build_batch(members[i : i + chunk], vocab, k_max,
                        x_order=None if x_keys is None else x_keys[i : i + chunk],
                        y_order=None if y_keys is None else y_keys[i : i + chunk])
        logits, _ = model(b["tokens"])
        stats_parts.append(position_stats(logits, b))
        batch_parts.append({k: b[k] for k in ("targets", "scored", "k", "tokens")})
    cat = lambda parts: {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    return cat(stats_parts), cat(batch_parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--orderings", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    task = config["task"]
    vocab = Vocab(task["num_symbols"])
    model = Transformer.from_config(config["model"]).to(args.device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=args.device, weights_only=True))
    model.eval()

    sets = split_sets(task["num_symbols"], task["k_min"], task["k_max"],
                      task["val_frac"], task["test_frac"], task["split_seed"])[args.split]
    members = torch.cat([sets[k] for k in sorted(sets)]).to(args.device)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)

    results = {}
    worst = []
    reps = members.repeat(args.orderings, 1)
    xk = torch.rand(reps.shape, device=args.device, generator=gen)
    yk = torch.rand(reps.shape, device=args.device, generator=gen)
    evals = [("random", reps, xk, yk)]
    evals += [(name, members, x, y) for name, x, y in
              structured_orders(len(members), vocab.num_symbols, args.device, gen)]

    for name, mem, x, y in evals:
        stats, batch = run(model, mem, vocab, task["k_max"], x, y)
        results[name] = summarize(stats, batch, vocab)
        results[name]["n_sequences"] = len(mem)
        bad = ((stats["margin"] <= 0) & batch["scored"]).any(-1).nonzero().squeeze(-1)[:5]
        worst += [f"[{name}] {vocab.render(batch['tokens'][i])}" for i in bad.tolist()]
        r = results[name]
        print(f"{name:>16}: {r['n_sequences']:>9,} seqs | errors {r['n_seq_errors']:>6} | "
              f"min_margin {r['min_margin']:+.3f} | kl {r['kl']:.5f} | max_tv {r['max_tv']:.4f}")

    total_errors = sum(r["n_seq_errors"] for r in results.values())
    print(f"TOTAL sequence errors on {args.split}: {total_errors}")
    for w in worst[:10]:
        print("  error:", w)
    out = run_dir / f"eval_{args.split}.json"
    out.write_text(json.dumps({"split": args.split, "orderings": args.orderings,
                               "total_errors": total_errors, "results": results}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
