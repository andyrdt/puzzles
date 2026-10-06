"""Checks of the data protocol in task.py.   python 10_2026/puzzle1/test_task.py"""

from itertools import combinations

import torch

from task import SPLITS, Sampler, Vocab, build_batch, position_stats, split_sets, summarize


def test_splits_are_disjoint_and_complete():
    sets = split_sets(16, 2, 8, 0.05, 0.10, 0)
    for k in range(2, 9):
        rows = [set(map(tuple, sets[s][k].int().tolist())) for s in SPLITS]
        assert all(len(r) == len(sets[s][k]) for r, s in zip(rows, SPLITS))
        assert not (rows[0] & rows[1] or rows[0] & rows[2] or rows[1] & rows[2])
        n = sum(len(r) for r in rows)
        assert n == len(list(combinations(range(16), k)))


def test_batch_layout_and_targets():
    vocab = Vocab(16)
    sets = split_sets(16, 2, 8, 0.05, 0.10, 0)
    members = Sampler(sets["train"], "cpu", seed=0).sample_members(2048)
    b = build_batch(members, vocab, 8, generator=torch.Generator().manual_seed(0))
    tok, tgt = b["tokens"], b["targets"]
    for i in range(len(tok)):
        K = int(b["k"][i])
        x, y = tok[i, 1 : 1 + K].tolist(), tok[i, K + 2 : 2 * K + 2].tolist()
        assert tok[i, 0] == vocab.BOS and tok[i, K + 1] == vocab.SEP and (tok[i, 2 * K + 2 :] == vocab.EOS).all()
        assert sorted(x) == sorted(y) == members[i].nonzero().squeeze(-1).tolist()
        assert b["scored"][i].nonzero().squeeze(-1).tolist() == list(range(K + 1, 2 * K + 2))
        for j, pos in enumerate(range(K + 1, 2 * K + 2)):
            remaining = [s for s in x if s not in y[:j]]
            expect = torch.zeros(vocab.size)
            if remaining:
                expect[remaining] = 1 / len(remaining)
            else:
                expect[vocab.EOS] = 1.0
            assert torch.allclose(tgt[i, pos], expect)


def test_perfect_predictor_scores_perfectly():
    vocab = Vocab(16)
    members = Sampler(split_sets(16, 2, 8, 0.05, 0.10, 0)["val"], "cpu", seed=1).sample_members(512)
    b = build_batch(members, vocab, 8, generator=torch.Generator().manual_seed(1))
    logits = torch.where(b["targets"] > 0, 0.0, -30.0)
    m = summarize(position_stats(logits, b), b, vocab)
    assert m["seq_acc"] == 1.0 and abs(m["kl"]) < 1e-6 and m["min_margin"] == 30.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
