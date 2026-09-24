#!/usr/bin/env python3
"""Stage 1: naive GPU port of the GQE policy forward pass.

Same spirit as vqe_naive.py: this is what a straightforward "port the
PyTorch model to hand-rolled CuPy ops" attempt looks like, with no thought
given to batching efficiency or kernel fusion:

  1. Samples are processed ONE AT A TIME in a Python for-loop, each with its
     own forward pass through the network -- even though GQE always needs a
     *batch* of samples per training step. Batch size effectively becomes 1
     no matter what --batch-size is set to, which is precisely the kind of
     thing that looks "fine" in a correctness test at small scale and then
     tanks GPU utilization in practice.

  2. Every linear layer, softmax, and layernorm is its own small CuPy kernel
     call, each followed by `cp.cuda.Stream.null.synchronize()`. Attention
     and MLP blocks are not fused; small (32-dim) matmuls dominate.

  3. Weights are transferred from the exported NumPy dict on every call
     instead of being uploaded once and reused (a subtle host-device
     synchronous-copy tax that's easy to miss until you profile it).

Tracked kernel: 'policy_forward' -- the same logical operation (compute
next-token logits for a batch of gate-token sequences) re-implemented with
increasing batching/fusion in stage2-4.
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


def layer_norm_naive(x: cp.ndarray, weight: cp.ndarray, bias: cp.ndarray, eps: float = 1e-5) -> cp.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    out = (x - mean) / cp.sqrt(var + eps) * weight + bias
    cp.cuda.Stream.null.synchronize()
    return out


def linear_naive(x: cp.ndarray, weight: cp.ndarray, bias: cp.ndarray | None = None) -> cp.ndarray:
    out = x @ weight.T
    if bias is not None:
        out = out + bias
    cp.cuda.Stream.null.synchronize()  # naive: sync after every linear layer
    return out


def softmax_naive(x: cp.ndarray, axis: int = -1) -> cp.ndarray:
    x_max = cp.max(x, axis=axis, keepdims=True)
    e = cp.exp(x - x_max)
    out = e / cp.sum(e, axis=axis, keepdims=True)
    cp.cuda.Stream.null.synchronize()
    return out


def gelu_naive(x: cp.ndarray) -> cp.ndarray:
    out = 0.5 * x * (1.0 + cp.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    cp.cuda.Stream.null.synchronize()
    return out


def attention_one_sample(x: cp.ndarray, w: dict, cfg: GQEConfig, layer_idx: int) -> cp.ndarray:
    """x: (T, D) for a SINGLE sample -- naive stage1 never batches this."""
    T, D = x.shape
    H, Dh = cfg.n_heads, D // cfg.n_heads
    prefix = f"blocks.{layer_idx}.attn"

    qkv_w = w[f"{prefix}.qkv.weight"]
    qkv = linear_naive(x, qkv_w)  # (T, 3D)
    qkv = qkv.reshape(T, 3, H, Dh)
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each (T, H, Dh)
    q, k, v = (cp.transpose(t, (1, 0, 2)) for t in (q, k, v))  # (H, T, Dh)

    scores = cp.matmul(q, cp.transpose(k, (0, 2, 1))) / np.sqrt(Dh)  # (H, T, T)
    causal_mask = cp.triu(cp.ones((T, T), dtype=cp.bool_), k=1)
    scores = cp.where(causal_mask, cp.float32(-1e30), scores)
    attn = softmax_naive(scores, axis=-1)
    out = cp.matmul(attn, v)  # (H, T, Dh)
    out = cp.transpose(out, (1, 0, 2)).reshape(T, D)

    proj_w = w[f"{prefix}.proj.weight"]
    return linear_naive(out, proj_w)


def block_one_sample(x: cp.ndarray, w: dict, cfg: GQEConfig, layer_idx: int) -> cp.ndarray:
    prefix = f"blocks.{layer_idx}"
    ln1 = layer_norm_naive(x, w[f"{prefix}.ln1.weight"], w[f"{prefix}.ln1.bias"])
    x = x + attention_one_sample(ln1, w, cfg, layer_idx)

    ln2 = layer_norm_naive(x, w[f"{prefix}.ln2.weight"], w[f"{prefix}.ln2.bias"])
    h = linear_naive(ln2, w[f"{prefix}.mlp.0.weight"], w[f"{prefix}.mlp.0.bias"])
    h = gelu_naive(h)
    h = linear_naive(h, w[f"{prefix}.mlp.2.weight"], w[f"{prefix}.mlp.2.bias"])
    return x + h


def forward_one_sample(tokens: cp.ndarray, w: dict, cfg: GQEConfig) -> cp.ndarray:
    """tokens: (T,) int64 for ONE sample. Returns (T, vocab_size) logits."""
    T = tokens.shape[0]
    tok_emb = w["tok_emb.weight"][tokens]  # (T, D), naive gather
    pos_emb = w["pos_emb"][0, :T, :]
    x = tok_emb + pos_emb
    cp.cuda.Stream.null.synchronize()

    for layer_idx in range(cfg.n_layers):
        x = block_one_sample(x, w, cfg, layer_idx)

    x = layer_norm_naive(x, w["ln_f.weight"], w["ln_f.bias"])
    logits = linear_naive(x, w["head.weight"])
    return logits


def policy_forward_naive(tokens_batch: cp.ndarray, w: dict, cfg: GQEConfig) -> cp.ndarray:
    """Naive: Python for-loop over the batch, one sample at a time.
    This is the 'policy_forward kernel' tracked across all four stages."""
    B = tokens_batch.shape[0]
    outputs = []
    for b in range(B):
        logits = forward_one_sample(tokens_batch[b], w, cfg)
        outputs.append(logits)
    return cp.stack(outputs, axis=0)


def upload_weights(weights_np: dict) -> dict:
    """Naive: re-upload weights to GPU on every call (no caching)."""
    return {k: cp.asarray(v) for k, v in weights_np.items()}


def main():
    ap = argparse.ArgumentParser(description="Stage 1 naive GPU GQE policy forward pass.")
    add_gqe_args(ap)
    args = ap.parse_args()

    cfg = GQEConfig(
        vocab_size=args.vocab_size, seq_len=args.seq_len, d_model=args.d_model,
        n_layers=args.n_layers, n_heads=args.n_heads,
    )
    model = deterministic_weights(cfg, seed=args.seed)
    weights_np = export_weights_numpy(model)

    tokens_torch = sample_tokens(args.batch_size, cfg, seed=args.seed + 1)
    tokens_np = tokens_torch.numpy()
    tokens_gpu = cp.asarray(tokens_np)

    t0 = time.perf_counter()
    weights_gpu = upload_weights(weights_np)  # naive: re-upload every call
    logits = policy_forward_naive(tokens_gpu, weights_gpu, cfg)
    cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()

    print(f"[stage1_naive] GQE policy_forward: batch={args.batch_size} seq_len={cfg.seq_len} "
          f"d_model={cfg.d_model} n_layers={cfg.n_layers}")
    print(f"[stage1_naive] logits shape = {logits.shape}")
    print(f"[stage1_naive] wall time = {(t1 - t0) * 1e3:.3f} ms")


if __name__ == "__main__":
    main()
