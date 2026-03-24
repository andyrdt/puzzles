"""
Shared attention-only transformer model for mech interp puzzles.

This is intentionally simple and transparent — no libraries like TransformerLens,
just raw PyTorch so participants can inspect every weight matrix directly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum


class AttentionHead(nn.Module):
    def __init__(self, d_model: int, d_head: int):
        super().__init__()
        self.d_head = d_head
        self.W_Q = nn.Linear(d_model, d_head, bias=False)
        self.W_K = nn.Linear(d_model, d_head, bias=False)
        self.W_V = nn.Linear(d_model, d_head, bias=False)

    def forward(self, x, mask=None):
        q = self.W_Q(x)
        k = self.W_K(x)
        v = self.W_V(x)
        scores = einsum(q, k, "b i d, b j d -> b i j") / (self.d_head ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = einsum(attn, v, "b i j, b j d -> b i d")
        return out, attn


class AttentionOnlyLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.heads = nn.ModuleList(
            [AttentionHead(d_model, self.d_head) for _ in range(n_heads)]
        )
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        head_outs = []
        attns = []
        for head in self.heads:
            out, attn = head(x, mask)
            head_outs.append(out)
            attns.append(attn)
        concat = torch.cat(head_outs, dim=-1)
        return self.W_O(concat), torch.stack(attns, dim=1)


class AttentionOnlyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 max_seq_len: int, n_layers: int = 1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList(
            [AttentionOnlyLayer(d_model, n_heads) for _ in range(n_layers)]
        )
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

        nn.init.normal_(self.tok_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        nn.init.normal_(self.unembed.weight, std=0.02)

    def forward(self, x):
        b, s = x.shape
        positions = torch.arange(s, device=x.device).unsqueeze(0)
        h = self.tok_embed(x) + self.pos_embed(positions)

        mask = torch.tril(torch.ones(s, s, device=x.device)).unsqueeze(0)

        all_attns = []
        for layer in self.layers:
            residual = h
            out, attns = layer(h, mask)
            h = residual + out
            all_attns.append(attns)

        logits = self.unembed(h)
        return logits, all_attns

    def config_dict(self):
        """Return a dict of model config for serialization."""
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "max_seq_len": self.max_seq_len,
        }

    @classmethod
    def from_config(cls, config):
        return cls(
            vocab_size=config["vocab_size"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            max_seq_len=config["max_seq_len"],
            n_layers=config["n_layers"],
        )
