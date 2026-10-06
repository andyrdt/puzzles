"""The transformer used for October 2026.

Two causal attention-only layers with one head each, pre-norm RMSNorm before
every attention layer and before the unembedding, and rotary position
embeddings (RoPE) on queries and keys. No MLPs and no biases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_cos_sin(seq_len, d_head, base, device):
    """Rotary tables (GPT-NeoX / Llama half-split layout), each [seq_len, d_head]."""
    inv_freq = 1.0 / base ** (torch.arange(0, d_head, 2, device=device).float() / d_head)
    angles = torch.outer(torch.arange(seq_len, device=device).float(), inv_freq)
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()


class Attention(nn.Module):
    """Multi-head causal self-attention with RoPE and an output projection."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask, rope):
        split = lambda t: rearrange(t, "b s (h d) -> b h s d", h=self.n_heads)
        q, k, v = split(self.W_Q(x)), split(self.W_K(x)), split(self.W_V(x))
        cos, sin = rope
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        scores = einsum(q, k, "b h i d, b h j d -> b h i j") / self.d_head**0.5
        scores = scores.masked_fill(~mask, float("-inf"))
        pattern = F.softmax(scores, dim=-1)
        z = einsum(pattern, v, "b h i j, b h j d -> b h i d")
        return self.W_O(rearrange(z, "b h s d -> b s (h d)")), pattern


class Block(nn.Module):
    """Pre-norm attention block: x + attn(RMSNorm(x))."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln_attn = nn.RMSNorm(d_model, eps=1e-5)
        self.attn = Attention(d_model, n_heads)

    def forward(self, x, mask, rope):
        attn_out, pattern = self.attn(self.ln_attn(x), mask, rope)
        return x + attn_out, pattern


class Transformer(nn.Module):
    """Causal attention-only transformer with RoPE and pre-norm RMSNorm."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_seq_len: int,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        if (d_model // n_heads) % 2:
            raise ValueError("RoPE needs an even d_head")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.rope_base = rope_base

        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_final = nn.RMSNorm(d_model, eps=1e-5)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.normal_(self.tok_embed.weight, std=0.02)

    def forward(self, tokens):
        """Returns (logits [b, s, vocab], list of attention patterns [b, h, s, s] per layer)."""
        _, seq_len = tokens.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"sequence length {seq_len} > max_seq_len={self.max_seq_len}")
        x = self.tok_embed(tokens)
        rope = rope_cos_sin(seq_len, self.d_model // self.n_heads, self.rope_base, tokens.device)
        mask = torch.ones(seq_len, seq_len, device=tokens.device, dtype=torch.bool).tril()
        patterns = []
        for block in self.blocks:
            x, pattern = block(x, mask, rope)
            patterns.append(pattern)
        return self.unembed(self.ln_final(x)), patterns

    def config_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "max_seq_len": self.max_seq_len,
            "rope_base": self.rope_base,
        }

    @classmethod
    def from_config(cls, config):
        return cls(
            vocab_size=config["vocab_size"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            max_seq_len=config["max_seq_len"],
            rope_base=config.get("rope_base", 10000.0),
        )
