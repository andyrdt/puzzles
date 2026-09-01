"""The 108-parameter attention-only model used for September 2026."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum


class AttentionHead(nn.Module):
    """One causal self-attention head."""

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
        scores = einsum(q, k, "b i d, b j d -> b i j") / self.d_head**0.5
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attention = F.softmax(scores, dim=-1)
        output = einsum(attention, v, "b i j, b j d -> b i d")
        return output, attention


class AttentionOnlyLayer(nn.Module):
    """Multi-head attention followed by an output projection."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.heads = nn.ModuleList(
            [AttentionHead(d_model, self.d_head) for _ in range(n_heads)]
        )
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        head_outputs = []
        attention_patterns = []
        for head in self.heads:
            output, attention = head(x, mask)
            head_outputs.append(output)
            attention_patterns.append(attention)
        concatenated = torch.cat(head_outputs, dim=-1)
        return self.W_O(concatenated), torch.stack(attention_patterns, dim=1)


class AttentionOnlyTransformer(nn.Module):
    """Causal transformer with learned positions and attention only.

    The released checkpoint has one layer, two width-one heads, and a
    two-dimensional residual stream. The implementation remains parameterized
    so contestants can run small controls with the same architecture.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        n_layers: int = 1,
        pos_embed_type: str = "learned",
        pos_embed_std: float = 0.02,
    ):
        super().__init__()
        if pos_embed_type != "learned":
            raise ValueError("the released architecture uses learned positions")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.pos_embed_type = pos_embed_type
        self.pos_embed_std = pos_embed_std

        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList(
            [AttentionOnlyLayer(d_model, n_heads) for _ in range(n_layers)]
        )
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

        nn.init.normal_(self.tok_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=pos_embed_std)
        nn.init.normal_(self.unembed.weight, std=0.02)

    def forward(self, tokens):
        _, seq_len = tokens.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        residual = self.tok_embed(tokens) + self.pos_embed(positions)
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=tokens.device, dtype=torch.bool)
        ).unsqueeze(0)

        all_attention_patterns = []
        for layer in self.layers:
            attention_output, attention_patterns = layer(residual, causal_mask)
            residual = residual + attention_output
            all_attention_patterns.append(attention_patterns)

        return self.unembed(residual), all_attention_patterns

    def config_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "max_seq_len": self.max_seq_len,
            "pos_embed_type": self.pos_embed_type,
            "pos_embed_std": self.pos_embed_std,
        }

    @classmethod
    def from_config(cls, config):
        return cls(
            vocab_size=config["vocab_size"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            max_seq_len=config["max_seq_len"],
            pos_embed_type=config.get("pos_embed_type", "learned"),
            pos_embed_std=config.get("pos_embed_std", 0.02),
        )
