#!/usr/bin/env python3
"""Stage 2: GPU-native restructuring of the GQE policy forward pass.

Same network, same weights, same math as stage1_naive/gqe_naive.py -- what
changes is that the ENTIRE BATCH is processed in one set of kernel calls,
the way a real transformer forward pass should look:

  1. Batched matmuls. Every linear layer operates on the full
     (batch, seq_len, d_model) tensor at once via a single `@` (cuBLAS
     batched/strided GEMM under the hood) instead of stage1's Python loop
     over samples, each with its own (seq_len, d_model) matmul. This turns
     B tiny GEMMs into 1 GEMM with B times the arithmetic intensity.

  2. Batched attention. `cp.matmul` on tensors with a leading batch
     dimension is a native batched-GEMM operation -- QK^T and attn@V for
     ALL samples and ALL heads happen in a couple of kernel launches
     instead of stage1's nested per-sample, per-head Python loops.

  3. No synchronization between layers. Only the final forward pass result
     is synced/copied to host.

This is the 'policy_forward' kernel, tracked across all four stages. In
stage2 it is "as fused as a straightforward batched-matmul rewrite gets";
stage3/4 push further into custom kernels and tensor-core GEMMs.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cupy as cp
import numpy as np

from common.config import add_gqe_args
from common.gqe_policy_ref import GQEConfig, deterministic_weights, export_weights_numpy, sample_tokens


def layer_norm_batched(x: cp.ndarray, weight: cp.ndarray, bias: cp.ndarray, eps: float = 1e-5) -> cp.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / cp.sqrt(var + eps) * weight + bias


def gelu_batched(x: cp.ndarray) -> cp.ndarray:
    return 0.5 * x * (1.0 + cp.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def attention_batched(x: cp.ndarray, w: dict, cfg: GQEConfig, layer_idx: int) -> cp.ndarray:
    """x: (B, T, D) -- the whole batch at once."""
    B, T, D = x.shape
    H, Dh = cfg.n_heads, D // cfg.n_heads
    prefix = f"blocks.{layer_idx}.attn"

    qkv = x @ w[f"{prefix}.qkv.weight"].T  # (B, T, 3D) -- one batched GEMM for the whole batch
    qkv = qkv.reshape(B, T, 3, H, Dh)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # each (B, T, H, Dh)
    q, k, v = (cp.transpose(t, (0, 2, 1, 3)) for t in (q, k, v))  # (B, H, T, Dh)

    scores = cp.matmul(q, cp.transpose(k, (0, 1, 3, 2))) / np.sqrt(Dh)  # (B, H, T, T), batched GEMM
    causal_mask = cp.triu(cp.ones((T, T), dtype=cp.bool_), k=1)
    scores = cp.where(causal_mask, cp.float32(-1e30), scores)
    scores = scores - cp.max(scores, axis=-1, keepdims=True)
    attn = cp.exp(scores)
    attn = attn / cp.sum(attn, axis=-1, keepdims=True)

    out = cp.matmul(attn, v)  # (B, H, T, Dh), batched GEMM
    out = cp.transpose(out, (0, 2, 1, 3)).reshape(B, T, D)
    return out @ w[f"{prefix}.proj.weight"].T


def block_batched(x: cp.ndarray, w: dict, cfg: GQEConfig, layer_idx: int) -> cp.ndarray:
    prefix = f"blocks.{layer_idx}"
    ln1 = layer_norm_batched(x, w[f"{prefix}.ln1.weight"], w[f"{prefix}.ln1.bias"])
    x = x + attention_batched(ln1, w, cfg, layer_idx)

    ln2 = layer_norm_batched(x, w[f"{prefix}.ln2.weight"], w[f"{prefix}.ln2.bias"])
    h = ln2 @ w[f"{prefix}.mlp.0.weight"].T + w[f"{prefix}.mlp.0.bias"]
    h = gelu_batched(h)
    h = h @ w[f"{prefix}.mlp.2.weight"].T + w[f"{prefix}.mlp.2.bias"]
    return x + h


def policy_forward_batched(tokens_batch: cp.ndarray, w: dict, cfg: GQEConfig) -> cp.ndarray:
    """tokens_batch: (B, T) int64. This is the 'policy_forward' kernel,
    now processing the entire batch with no Python-level per-sample loop."""
    B, T = tokens_batch.shape
    tok_emb = w["tok_emb.weight"][tokens_batch]  # (B, T, D), one batched gather
    pos_emb = w["pos_emb"][:, :T, :]  # (1, T, D), broadcasts over batch
    x = tok_emb + pos_emb

    for layer_idx in range(cfg.n_layers):
        x = block_batched(x, w, cfg, layer_idx)

    x = layer_norm_batched(x, w["ln_f.weight"], w["ln_f.bias"])
    logits = x @ w["head.weight"].T
    return logits


_WEIGHT_CACHE: dict[int, dict] = {}


def upload_weights_cached(weights_np: dict, cache_key: int = 0) -> dict:
    """Unlike stage1, weights are uploaded once and cached by identity --
    a real training loop calls the forward pass many times per weight
    update, so re-uploading unchanged weights on every call would be a
    self-inflicted, entirely avoidable host-to-device transfer."""
    if cache_key not in _WEIGHT_CACHE:
        _WEIGHT_CACHE[cache_key] = {k: cp.asarray(v) for k, v in weights_np.items()}
    return _WEIGHT_CACHE[cache_key]


def main():
    ap = argparse.ArgumentParser(description="Stage 2 GPU-native batched GQE policy forward pass.")
    add_gqe_args(ap)
    args = ap.parse_args()

    cfg = GQEConfig(
        vocab_size=args.vocab_size, seq_len=args.seq_len, d_model=args.d_model,
        n_layers=args.n_layers, n_heads=args.n_heads,
    )
    model = deterministic_weights(cfg, seed=args.seed)
    weights_np = export_weights_numpy(model)
    weights_gpu = upload_weights_cached(weights_np)

    tokens_torch = sample_tokens(args.batch_size, cfg, seed=args.seed + 1)
    tokens_gpu = cp.asarray(tokens_torch.numpy())

    t0 = time.perf_counter()
    logits = policy_forward_batched(tokens_gpu, weights_gpu, cfg)
    cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()

    print(f"[stage2_batched] GQE policy_forward: batch={args.batch_size} seq_len={cfg.seq_len} "
          f"d_model={cfg.d_model} n_layers={cfg.n_layers}")
    print(f"[stage2_batched] logits shape = {logits.shape}")
    print(f"[stage2_batched] wall time = {(t1 - t0) * 1e3:.3f} ms")


if __name__ == "__main__":
    main()
