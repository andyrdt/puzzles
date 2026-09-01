"""Fast invariants for the September task and evaluation protocol."""

import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent))

from model import AttentionOnlyTransformer
from train import (
    Vocab,
    build_permutation_table,
    build_eval_examples,
    enumerate_sets,
    evaluate,
    load_checkpoint_splits,
    render_training_batch,
    split_fingerprint,
    split_sets,
)


class SetDifferenceOracle(nn.Module):
    def __init__(self, vocab: Vocab, set_size: int):
        super().__init__()
        self.vocab = vocab
        self.set_size = set_size

    def forward(self, inputs):
        batch, seq_len = inputs.shape
        logits = torch.full(
            (batch, seq_len, self.vocab.size),
            -100.0,
            device=inputs.device,
        )
        x = inputs[:, 1 : 1 + self.set_size]
        y = inputs[:, self.set_size + 2 : 2 * self.set_size + 1]
        missing = x[~(x[:, :, None] == y[:, None, :]).any(dim=2)]
        logits[torch.arange(batch, device=inputs.device), -1, missing] = 100.0
        return logits, []


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.num_symbols = 8
        self.set_size = 3
        self.vocab = Vocab(self.num_symbols)
        all_sets = enumerate_sets(self.num_symbols, self.set_size, max_sets=1000)
        self.splits = split_sets(
            all_sets,
            self.num_symbols,
            val_frac=0.2,
            test_frac=0.2,
            seed=123,
        )

    def test_whole_set_splits_are_disjoint(self):
        keys = {
            name: {tuple(row.tolist()) for row in rows}
            for name, rows in self.splits.items()
            if name in ("train", "val", "test")
        }
        self.assertTrue(keys["train"].isdisjoint(keys["val"]))
        self.assertTrue(keys["train"].isdisjoint(keys["test"]))
        self.assertTrue(keys["val"].isdisjoint(keys["test"]))

    def test_checkpoint_splits_are_loaded_as_source_of_truth(self):
        tensors = {
            name: self.splits[name] for name in ("train", "val", "test")
        }
        config = {
            "requested_seed": 9999,
            "val_frac": 0.2,
            "test_frac": 0.2,
            "counts": {name: len(sets) for name, sets in tensors.items()},
            "fingerprints": {
                name: split_fingerprint(sets) for name, sets in tensors.items()
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            torch.save(tensors, checkpoint_dir / "splits.pt")
            loaded = load_checkpoint_splits(
                checkpoint_dir,
                config,
                self.num_symbols,
                self.set_size,
            )
        for name in tensors:
            self.assertTrue(torch.equal(loaded[name], tensors[name]))

    def test_exhaustive_rendering_is_complete_unique_and_valid(self):
        sets = self.splits["val"][:2]
        examples = build_eval_examples(
            sets, permutations=0, vocab=self.vocab, seed=1
        )
        expected_per_problem = 12  # 3! * 2!
        self.assertEqual(examples["permutations"], expected_per_problem)
        self.assertEqual(len(examples["inputs"]), 2 * 3 * expected_per_problem)
        self.assertEqual(torch.unique(examples["inputs"], dim=0).shape[0], 72)

        for inputs, target in zip(examples["inputs"], examples["targets"]):
            x = set(inputs[1 : 1 + self.set_size].tolist())
            y = set(inputs[self.set_size + 2 : 2 * self.set_size + 1].tolist())
            self.assertEqual(x - y, {target.item()})
            self.assertEqual(y, x - {target.item()})

    def test_oracle_scores_perfectly_under_grouped_metrics(self):
        examples = build_eval_examples(
            self.splits["val"], permutations=0, vocab=self.vocab, seed=1
        )
        oracle = SetDifferenceOracle(self.vocab, self.set_size)
        metrics = evaluate(
            oracle,
            examples,
            self.vocab,
            torch.device("cpu"),
            batch_size=1024,
            precision="fp32",
        )
        self.assertEqual(metrics["acc"], 1.0)
        self.assertEqual(metrics["strict_problem_acc"], 1.0)
        self.assertEqual(metrics["strict_set_acc"], 1.0)
        self.assertEqual(metrics["num_errors"], 0)

    def test_dynamic_gpu_style_renderer_obeys_task(self):
        pool = self.splits["train"]
        permutation_options = (
            (None, None),
            (
                build_permutation_table(self.set_size, torch.device("cpu")),
                build_permutation_table(self.set_size - 1, torch.device("cpu")),
            ),
        )
        for x_permutations, y_permutations in permutation_options:
            generator = torch.Generator().manual_seed(99)
            inputs, targets = render_training_batch(
                pool,
                256,
                self.vocab,
                generator,
                x_permutations,
                y_permutations,
            )
            for row, target in zip(inputs, targets):
                x = set(row[1 : 1 + self.set_size].tolist())
                y = set(row[self.set_size + 2 : 2 * self.set_size + 1].tolist())
                self.assertEqual(x - y, {target.item()})

    def test_model_attention_is_causal(self):
        model = AttentionOnlyTransformer(
            vocab_size=self.vocab.size,
            d_model=16,
            n_heads=2,
            n_layers=2,
            max_seq_len=2 * self.set_size + 2,
        )
        examples = build_eval_examples(
            self.splits["val"][:1], permutations=1, vocab=self.vocab, seed=5
        )
        _, attentions = model(examples["inputs"])
        upper_triangle = torch.triu(
            torch.ones(model.max_seq_len, model.max_seq_len, dtype=torch.bool),
            diagonal=1,
        )
        for layer_attention in attentions:
            self.assertTrue(torch.equal(
                layer_attention[:, :, upper_triangle],
                torch.zeros_like(layer_attention[:, :, upper_triangle]),
            ))

    def test_released_architecture_round_trips_and_has_108_parameters(self):
        model = AttentionOnlyTransformer(
            vocab_size=18,
            d_model=2,
            n_heads=2,
            n_layers=1,
            max_seq_len=10,
        )
        restored = AttentionOnlyTransformer.from_config(model.config_dict())
        self.assertEqual(restored.config_dict(), model.config_dict())
        self.assertEqual(sum(p.numel() for p in restored.parameters()), 108)


if __name__ == "__main__":
    unittest.main()
