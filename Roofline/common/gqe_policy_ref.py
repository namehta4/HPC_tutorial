"""Reference GQE (Generative Quantum Eigensolver) policy network, in PyTorch on CPU.

GQE (Nakaji et al., "The generative quantum eigensolver (GQE) and its
application for ground state search", 2023-2024) replaces gradient-based
VQE parameter optimization with a generative model that learns to *sample*
good gate sequences (or good ansatz parameter sets) directly, trained via a
reward signal derived from the resulting energy. The GPU-performance-relevant
piece of GQE is not the training loop's chemistry -- it is the policy
network's forward pass, run many times per training step to sample a batch
of candidate circuits, which is exactly the kind of small-batch,
memory-bound-then-compute-bound inference workload that GPU roofline
analysis is meant to expose.

This module defines a small causal transformer policy: given a partial
sequence of "tokens" (each token = one discretized gate-parameter choice),
predict a distribution over the next token. It is intentionally small
(the tutorial is about GPU performance methodology, not about achieving
state-of-the-art GQE results) but structurally real: multi-head self
-attention + MLP blocks, exactly the shape whose forward pass every stage
re-implements with progressively better GPU utilization.

This CPU/PyTorch version is the correctness oracle: stage1-4 GPU
implementations (hand-rolled CuPy/CUDA, no autograd) must reproduce this
module's forward-pass logits for the same weights and input, within
floating-point tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GQEConfig:
    vocab_size: int = 16  # number of discretized gate-parameter "tokens"
    seq_len: int = 8  # gate-sequence length to generate (e.g. one token per ansatz layer per qubit-group)
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 64


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GQEConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, H, Dh)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, H, T, Dh)

        scores = (q @ k.transpose(-2, -1)) / np.sqrt(self.d_head)  # (B, H, T, T)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v  # (B, H, T, Dh)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, cfg: GQEConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GQEPolicy(nn.Module):
    """Autoregressive policy over gate-sequence tokens."""

    def __init__(self, cfg: GQEConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (batch, seq_len) int64 -> logits: (batch, seq_len, vocab_size)."""
        B, T = tokens.shape
        x = self.tok_emb(tokens) + self.pos_emb[:, :T, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


def deterministic_weights(cfg: GQEConfig, seed: int = 0) -> GQEPolicy:
    """Build a policy with fixed, reproducible weights for cross-stage comparison."""
    torch.manual_seed(seed)
    model = GQEPolicy(cfg)
    model.eval()
    return model


def sample_tokens(batch_size: int, cfg: GQEConfig, seed: int = 0) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    return torch.randint(0, cfg.vocab_size, (batch_size, cfg.seq_len), generator=rng)


def export_weights_numpy(model: GQEPolicy) -> dict[str, np.ndarray]:
    """Flatten all weights to a plain dict of NumPy arrays, for loading into
    hand-rolled CuPy/CUDA implementations that don't use PyTorch's autograd
    or module system at all."""
    return {name: p.detach().cpu().numpy().astype(np.float32) for name, p in model.state_dict().items()}
