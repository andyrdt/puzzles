"""Evaluate a saved September checkpoint on a reproducible whole-set split."""

import argparse
import json
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent))

from model import AttentionOnlyTransformer
from train import (
    Vocab,
    build_eval_examples,
    evaluate,
    load_checkpoint_splits,
    set_deterministic,
    split_fingerprint,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument(
        "--split", choices=["train", "val", "test"], default="test"
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=0,
        help="shuffles per (X,z); 0 exhausts all X/Y order pairs",
    )
    parser.add_argument("--max_eval_sets", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=65536)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "bf16"],
        default=None,
        help="evaluation precision; defaults to the checkpoint training precision",
    )
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = json.loads((args.checkpoint_dir / "config.json").read_text())
    precision = args.precision or config["training"]["precision"]
    set_deterministic(config["training"]["seed"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if precision == "bf16" and device.type != "cuda":
        raise RuntimeError("bf16 evaluation requires CUDA")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 requested but unsupported by this GPU")
    num_symbols = config["vocab"]["num_symbols"]
    set_size = config["task"]["set_size"]
    vocab = Vocab(num_symbols)

    split_config = config["splits"]
    splits = load_checkpoint_splits(
        args.checkpoint_dir,
        split_config,
        num_symbols,
        set_size,
    )
    actual_fingerprint = split_fingerprint(splits[args.split])
    expected_fingerprint = split_config["fingerprints"][args.split]
    if actual_fingerprint != expected_fingerprint:
        raise RuntimeError(
            "split fingerprint mismatch: "
            f"{actual_fingerprint} != {expected_fingerprint}"
        )

    examples = build_eval_examples(
        splits[args.split],
        args.permutations,
        vocab,
        args.seed,
        args.max_eval_sets,
    )
    model = AttentionOnlyTransformer.from_config(config["model"]).to(device)
    model.load_state_dict(
        torch.load(
            args.checkpoint_dir / "model.pt",
            map_location=device,
            weights_only=True,
        )
    )
    metrics = evaluate(
        model, examples, vocab, device, args.batch_size, precision
    )
    report = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "split": args.split,
        "split_fingerprint": actual_fingerprint,
        "precision": precision,
        "metrics": metrics,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
