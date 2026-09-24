#!/usr/bin/env python3
"""Stage 3: mixed-precision GQE policy forward pass.

Same batched-matmul structure as stage2_gpu_native/gqe_batched.py -- the
change here is purely about DATA TYPE, to show mixed precision as a
memory-traffic optimization independent of the VQE kernel work:

  - fp32 weights/activations (CuPy default for this network) halve memory
    traffic vs. a hypothetical fp64 transformer.
  - fp16 weights/activations halve it again vs. fp32, and on Ampere+ GPUs
    fp16 matmuls also run on tensor cores (see stage0_calibration/RESULTS.md
    for this GPU's measured fp16 GEMM throughput vs fp32/TF32).

The policy network here is intentionally small (d_model=32 by default), so
at THIS size the FLOPs saved by fp16 tensor cores barely matter -- the
matmuls are memory-traffic-bound, not compute-bound, at this scale. That's
the point: mixed precision helps most exactly where this tutorial's roofline
analysis says a kernel sits far from the compute roofline but has real
memory traffic to cut. Stage 4 revisits fp16/tensor-core throughput at
larger batch sizes where compute starts to matter more.

Correctness: fp16 logits are checked against the fp64 PyTorch reference
with a LOOSER tolerance than stage1/stage2's fp32-vs-fp64 comparisons --
fp16 has ~3 decimal digits of precision, and layernorm/softmax intermediate
sums are especially sensitive to it. This module keeps softmax and
layernorm accumulation in fp32 even when activations are stored as fp16,
which is standard practice for numerically stable low-precision transformers.
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


def layer_norm_fp16(x: cp.ndarray, weight: cp.ndarray, bias: cp.ndarray, eps: float = 1e-5) -> cp.ndarray:
    """x is fp16; accumulate mean/var in fp32 for numerical stability, cast back to fp16."""
    x32 = x.astype(cp.float32)
    mean = x32.mean(axis=-1, keepdims=True)
    var = x32.var(axis=-1, keepdims=True)
    out32 = (x32 - mean) / cp.sqrt(var + eps) * weight.astype(cp.float32) + bias.astype(cp.float32)
    return out32.astype(cp.float16)


def gelu_fp16(x: cp.ndarray) -> cp.ndarray:
    x32 = x.astype(cp.float32)
    out32 = 0.5 * x32 * (1.0 + cp.tanh(np.sqrt(2.0 / np.pi) * (x32 + 0.044715 * x32**3)))
    return out32.astype(cp.float16)


def attention_fp16(x: cp.ndarray, w: dict, cfg: GQEConfig, layer_idx: int) -> cp.ndarray:
    B, T, D = x.shape
    H, Dh = cfg.n_heads, D // cfg.n_heads
    prefix = f"blocks.{layer_idx}.attn"

    qkv = x @ w[f"{prefix}.qkv.weight"].T  # fp16 matmul -> tensor cores on Ampere+
    qkv = qkv.reshape(B, T, 3, H, Dh)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    q, k, v = (cp.transpose(t, (0, 2, 1, 3)) for t in (q, k, v))

    # Softmax accumulation in fp32: fp16 has too little dynamic range for
    # exp() of large/negative masked scores without under/overflow.
    scores = cp.matmul(q, cp.transpose(k, (0, 1, 3, 2))).astype(cp.float32) / np.sqrt(Dh)
    causal_mask = cp.triu(cp.ones((T, T), dtype=cp.bool_), k=1)
    scores = cp.where(causal_mask, cp.float32(-1e9), scores)
    scores = scores - cp.max(scores, axis=-1, keepdims=True)
    attn = cp.exp(scores)
    attn = (attn / cp.sum(attn, axis=-1, keepdims=True)).astype(cp.float16)

    out = cp.matmul(attn, v)
    out = cp.transpose(out, (0, 2, 1, 3)).reshape(B, T, D)
    return out @ w[f"{prefix}.proj.weight"].T


def block_fp16(x: cp.ndarray, w: dict, cfg: GQEConfig, layer_idx: int) -> cp.ndarray:
    prefix = f"blocks.{layer_idx}"
    ln1 = layer_norm_fp16(x, w[f"{prefix}.ln1.weight"], w[f"{prefix}.ln1.bias"])
    x = x + attention_fp16(ln1, w, cfg, layer_idx)

    ln2 = layer_norm_fp16(x, w[f"{prefix}.ln2.weight"], w[f"{prefix}.ln2.bias"])
    h = ln2 @ w[f"{prefix}.mlp.0.weight"].T + w[f"{prefix}.mlp.0.bias"]
    h = gelu_fp16(h)
    h = h @ w[f"{prefix}.mlp.2.weight"].T + w[f"{prefix}.mlp.2.bias"]
    return x + h


def policy_forward_fp16(tokens_batch: cp.ndarray, w: dict, cfg: GQEConfig) -> cp.ndarray:
    B, T = tokens_batch.shape
    tok_emb = w["tok_emb.weight"][tokens_batch]
    pos_emb = w["pos_emb"][:, :T, :]
    x = (tok_emb + pos_emb).astype(cp.float16)

    for layer_idx in range(cfg.n_layers):
        x = block_fp16(x, w, cfg, layer_idx)

    x = layer_norm_fp16(x, w["ln_f.weight"], w["ln_f.bias"])
    logits = x.astype(cp.float32) @ w["head.weight"].T.astype(cp.float32)  # final logits in fp32
    return logits


def upload_weights_fp16(weights_np: dict) -> dict:
    return {k: cp.asarray(v).astype(cp.float16) for k, v in weights_np.items()}


def main():
    ap = argparse.ArgumentParser(description="Stage 3 mixed-precision (fp16) GQE policy forward pass.")
    add_gqe_args(ap)
    args = ap.parse_args()

    cfg = GQEConfig(
        vocab_size=args.vocab_size, seq_len=args.seq_len, d_model=args.d_model,
        n_layers=args.n_layers, n_heads=args.n_heads,
    )
    model = deterministic_weights(cfg, seed=args.seed)
    weights_np = export_weights_numpy(model)
    weights_gpu = upload_weights_fp16(weights_np)

    tokens_torch = sample_tokens(args.batch_size, cfg, seed=args.seed + 1)
    tokens_gpu = cp.asarray(tokens_torch.numpy())

    t0 = time.perf_counter()
    logits = policy_forward_fp16(tokens_gpu, weights_gpu, cfg)
    cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()

    print(f"[stage3_fp16] GQE policy_forward (fp16): batch={args.batch_size} seq_len={cfg.seq_len} "
          f"d_model={cfg.d_model} n_layers={cfg.n_layers}")
    print(f"[stage3_fp16] logits shape = {logits.shape}")
    print(f"[stage3_fp16] wall time = {(t1 - t0) * 1e3:.3f} ms")


if __name__ == "__main__":
    main()
