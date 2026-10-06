"""Permutation / running set-difference task.

Sequence:  [BOS] x1 .. xK [SEP] y1 .. yK [EOS]
  X = {x1..xK}: K distinct symbols from V, in random order.
  Y: an independent uniformly random permutation of X.

Scored positions (input position -> soft target over the next token):
  SEP        -> uniform over X
  y_i, i<K   -> uniform over X \\ {y1..yi}
  y_K        -> EOS
Nothing else is scored. Batches mixing K are right-padded with EOS, which only
appears after every scored position, so causal attention never sees it.

Train/val/test are split over the underlying sets X (as bitmasks), so held-out
sets never appear in training in any ordering.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch


@dataclass(frozen=True)
class Vocab:
    num_symbols: int = 16

    @property
    def BOS(self):
        return self.num_symbols

    @property
    def SEP(self):
        return self.num_symbols + 1

    @property
    def EOS(self):
        return self.num_symbols + 2

    @property
    def size(self):
        return self.num_symbols + 3

    def name(self, tok: int) -> str:
        if tok < self.num_symbols:
            return chr(ord("a") + tok)
        return {self.BOS: "BOS", self.SEP: "SEP", self.EOS: "EOS"}[tok]

    def render(self, tokens) -> str:
        return " ".join(self.name(int(t)) for t in tokens)


def seq_len_for(k_max: int) -> int:
    return 2 * k_max + 3


SPLITS = ("train", "val", "test")


def split_sets(num_symbols, k_min, k_max, val_frac, test_frac, seed):
    """Deterministically partition every K-subset (K in [k_min, k_max]) into splits.

    Returns {split: {K: LongTensor of member masks [n, V] as bool}}.
    The split is stratified per K so every K appears in every split.
    """
    rng = np.random.default_rng(seed)
    out = {s: {} for s in SPLITS}
    for k in range(k_min, k_max + 1):
        subsets = np.array(list(combinations(range(num_symbols), k)), dtype=np.int64)
        perm = rng.permutation(len(subsets))
        n_test = max(1, round(len(subsets) * test_frac))
        n_val = max(1, round(len(subsets) * val_frac))
        idx = {
            "test": perm[:n_test],
            "val": perm[n_test : n_test + n_val],
            "train": perm[n_test + n_val :],
        }
        for s in SPLITS:
            members = np.zeros((len(idx[s]), num_symbols), dtype=bool)
            np.put_along_axis(members, subsets[idx[s]], True, axis=1)
            out[s][k] = torch.from_numpy(members)
    return out


def build_batch(members, vocab: Vocab, k_max: int, generator=None, x_order=None, y_order=None):
    """Turn set masks into sequences with soft targets.

    members: bool [B, V], each row a set X (row sums are the K values).
    x_order / y_order: optional float [B, V] sort keys; default is random.
    Returns dict with tokens [B, T], targets [B, T, vocab], scored [B, T] bool, k [B].
    """
    device = members.device
    B, V = members.shape
    T = seq_len_for(k_max)
    k = members.sum(-1)

    def order(keys):
        if keys is None:
            keys = torch.rand(B, V, device=device, generator=generator)
        keys = keys.masked_fill(~members, float("inf"))
        return keys.argsort(dim=-1)  # first K entries are the members, in key order

    xs, ys = order(x_order), order(y_order)

    pos = torch.arange(T, device=device).expand(B, T)
    kk = k[:, None]
    is_x = (pos >= 1) & (pos <= kk)
    is_y = (pos >= kk + 2) & (pos <= 2 * kk + 1)

    tokens = torch.full((B, T), vocab.EOS, device=device, dtype=torch.long)
    tokens[:, 0] = vocab.BOS
    x_idx = (pos - 1).clamp(0, V - 1)
    y_idx = (pos - kk - 2).clamp(0, V - 1)
    tokens = torch.where(is_x, xs.gather(1, x_idx), tokens)
    tokens = torch.where(pos == kk + 1, vocab.SEP, tokens)
    tokens = torch.where(is_y, ys.gather(1, y_idx), tokens)

    # used[b, t, v]: symbol v has appeared in Y at or before position t
    y_onehot = torch.zeros(B, T, V, device=device)
    y_onehot.scatter_(2, tokens.clamp(max=V - 1)[..., None], 1.0)
    y_onehot *= is_y[..., None]
    used = y_onehot.cumsum(1) > 0
    remaining = members[:, None, :] & ~used

    scored = (pos >= kk + 1) & (pos <= 2 * kk + 1)
    n_rem = remaining.sum(-1, keepdim=True)
    targets = torch.zeros(B, T, vocab.size, device=device)
    targets[..., :V] = remaining.float() / n_rem.clamp(min=1)
    targets[..., vocab.EOS] = (n_rem.squeeze(-1) == 0).float()
    targets *= scored[..., None]

    return {"tokens": tokens, "targets": targets, "scored": scored, "k": k}


class Sampler:
    """Samples training batches: K uniform over [k_min, k_max], then a set from the split."""

    def __init__(self, sets_by_k, device, seed):
        self.ks = sorted(sets_by_k)
        self.sets = {k: v.to(device) for k, v in sets_by_k.items()}
        self.device = device
        self.gen = torch.Generator(device=device).manual_seed(seed)

    def sample_members(self, batch_size):
        k_choice = torch.randint(len(self.ks), (batch_size,), device=self.device, generator=self.gen)
        members = torch.empty(batch_size, next(iter(self.sets.values())).shape[1],
                              dtype=torch.bool, device=self.device)
        for i, k in enumerate(self.ks):
            rows = (k_choice == i).nonzero().squeeze(-1)
            pick = torch.randint(len(self.sets[k]), (len(rows),), device=self.device, generator=self.gen)
            members[rows] = self.sets[k][pick]
        return members


def position_stats(logits, batch):
    """Per-position loss and correctness. All outputs are [B, T]; only `scored` entries matter.

    ce:     soft-target cross-entropy  -sum_t p_t log q_t
    kl:     ce - H(p); zero iff the model's distribution equals the target
    margin: min logit over valid tokens - max logit over invalid tokens
    tv:     total variation distance between softmax and target
    """
    targets = batch["targets"]
    logp = logits.float().log_softmax(-1)
    ce = -(targets * logp).sum(-1)
    entropy = -(targets * targets.clamp(min=1e-30).log()).sum(-1)
    valid = targets > 0
    min_valid = logits.float().masked_fill(~valid, float("inf")).min(-1).values
    max_invalid = logits.float().masked_fill(valid, float("-inf")).max(-1).values
    tv = 0.5 * (logp.exp() - targets).abs().sum(-1)
    return {"ce": ce, "kl": ce - entropy, "margin": min_valid - max_invalid, "tv": tv}


def summarize(stats, batch, vocab: Vocab, prefix=""):
    """Scalar metrics from `position_stats`; `prefix` namespaces the keys."""
    scored = batch["scored"]
    is_eos = scored & (batch["targets"][..., vocab.EOS] > 0)
    is_sym = scored & ~is_eos
    correct = stats["margin"] > 0
    seq_correct = (correct | ~scored).all(-1)
    m = lambda x, mask: x[mask].float().mean().item()
    out = {
        "ce": m(stats["ce"], scored),
        "kl": m(stats["kl"], scored),
        "kl_symbol": m(stats["kl"], is_sym),
        "kl_eos": m(stats["kl"], is_eos),
        "pos_acc": m(correct, scored),
        "seq_acc": seq_correct.float().mean().item(),
        "eos_acc": m(correct, is_eos),
        "min_margin": stats["margin"][scored].min().item(),
        "mean_tv": m(stats["tv"], scored),
        "max_tv": stats["tv"][scored].max().item(),
        "n_seq_errors": int((~seq_correct).sum().item()),
    }
    for k in batch["k"].unique().tolist():
        rows = batch["k"] == k
        out[f"seq_acc_k{k}"] = seq_correct[rows].float().mean().item()
        out[f"kl_k{k}"] = m(stats["kl"], scored & rows[:, None])
    return {f"{prefix}{key}": v for key, v in out.items()}
