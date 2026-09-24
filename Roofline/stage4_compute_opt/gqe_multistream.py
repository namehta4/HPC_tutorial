#!/usr/bin/env python3
"""Stage 4: tensor-core GEMM + multi-stream GQE policy forward pass.

Run ./build.sh first.

Two independent optimizations, combined:

  1. TENSOR-CORE LINEAR LAYERS. Every linear layer in the policy network
     (`qkv`, `proj`, both MLP layers, `head`) is replaced with a call to
     `batched_gemm_tc_fp16` (kernels/policy_forward_gemm.cu), a hand-written
     WMMA (Warp Matrix Multiply-Accumulate) kernel that runs on the GPU's
     tensor cores instead of ordinary CUDA cores. Unlike VQE's 2x2 gates,
     these ARE genuine matrix multiplies of a shape tensor cores are built
     for (padded to multiples of 16 where needed -- see
     policy_forward_gemm.py's padding helper). At this network's default
     size (d_model=32, batch~32) these GEMMs are small enough that the win
     is modest; the point is to show the mechanism and let you re-run at
     larger --d-model/--batch-size to see the tensor-core advantage grow
     (compare against stage0_calibration/RESULTS.md's measured fp16 vs
     fp32 GEMM peak FLOP/s on this GPU).

  2. MULTI-STREAM SCALING. `run_multistream_sampling_loop` demonstrates the
     GQE outer loop's actual concurrency pattern: sampling several
     INDEPENDENT batches of candidate gate sequences (e.g. for different
     temperature/exploration settings, or simply to pipeline generation
     with downstream VQE energy evaluation) on separate CUDA streams, so
     they execute concurrently on the GPU instead of one-after-another.

Attention's QK^T and attn@V matmuls remain plain CuPy batched matmuls (not
rewritten as custom WMMA calls) -- this tutorial's teaching point is "here
is how you'd hand-write a tensor-core GEMM and where it fits," not "here is
a from-scratch FlashAttention," which is a substantially larger undertaking
this tutorial's scope doesn't need.
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

KERNELS_DIR = Path(__file__).resolve().parent / "kernels"
_MODULE_CACHE: dict[str, cp.RawModule] = {}


def _get_module(cubin_name: str) -> cp.RawModule:
    if cubin_name not in _MODULE_CACHE:
        path = KERNELS_DIR / cubin_name
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run ./build.sh first")
        _MODULE_CACHE[cubin_name] = cp.RawModule(path=str(path))
    return _MODULE_CACHE[cubin_name]


def _pad_to_multiple(n: int, m: int = 16) -> int:
    return ((n + m - 1) // m) * m


def batched_gemm_tc(A: cp.ndarray, B: cp.ndarray) -> cp.ndarray:
    """C[b] = A[b] @ B for a batch of A against one shared weight B, via the
    hand-written WMMA kernel. A: (batch, M, K) fp16, B: (K, N) fp16.
    Returns (batch, M, N) fp32. Pads M/K/N up to multiples of 16 (WMMA's
    fp16 tile requirement) and slices the result back down -- zero-padding
    doesn't change the non-padded output entries."""
    batch, M, K = A.shape
    K2, N = B.shape
    assert K == K2

    Mp, Kp, Np = _pad_to_multiple(M), _pad_to_multiple(K), _pad_to_multiple(N)

    if (Mp, Kp, Np) != (M, K, N):
        A_padded = cp.zeros((batch, Mp, Kp), dtype=cp.float16)
        A_padded[:, :M, :K] = A
        B_padded = cp.zeros((Kp, Np), dtype=cp.float16)
        B_padded[:K, :N] = B
    else:
        A_padded, B_padded = A, B

    C_padded = cp.zeros((batch, Mp, Np), dtype=cp.float32)

    mod = _get_module("policy_forward_gemm.cubin")
    kernel = mod.get_function("batched_gemm_tc_fp16")
    grid = (Mp // 16, Np // 16, batch)
    block = (32,)
    kernel(grid, block, (A_padded, B_padded, C_padded, batch, Mp, Np, Kp))

    if (Mp, Kp, Np) != (M, K, N):
        return C_padded[:, :M, :N]
    return C_padded


def layer_norm(x: cp.ndarray, weight: cp.ndarray, bias: cp.ndarray, eps: float = 1e-5) -> cp.ndarray:
    x32 = x.astype(cp.float32)
    mean = x32.mean(axis=-1, keepdims=True)
    var = x32.var(axis=-1, keepdims=True)
    return (x32 - mean) / cp.sqrt(var + eps) * weight.astype(cp.float32) + bias.astype(cp.float32)


def gelu(x: cp.ndarray) -> cp.ndarray:
    return 0.5 * x * (1.0 + cp.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def linear_tc(x: cp.ndarray, weight: cp.ndarray) -> cp.ndarray:
    """x: (B, T, D_in) fp32-or-fp16, weight: (D_out, D_in) -> (B, T, D_out) fp32,
    computed via the tensor-core batched GEMM (x reshaped to (B*T, D_in))."""
    B, T, D_in = x.shape
    D_out = weight.shape[0]
    x_flat = x.reshape(B * T, 1, D_in).astype(cp.float16)
    w_t = weight.T.astype(cp.float16)  # (D_in, D_out)
    out = batched_gemm_tc(x_flat, w_t)  # (B*T, 1, D_out)
    return out.reshape(B, T, D_out)


def attention_tc(x: cp.ndarray, w: dict, cfg: GQEConfig, layer_idx: int) -> cp.ndarray:
    B, T, D = x.shape
    H, Dh = cfg.n_heads, D // cfg.n_heads
    prefix = f"blocks.{layer_idx}.attn"

    qkv = linear_tc(x, w[f"{prefix}.qkv.weight"])  # tensor-core GEMM
    qkv = qkv.reshape(B, T, 3, H, Dh)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    q, k, v = (cp.transpose(t, (0, 2, 1, 3)) for t in (q, k, v))

    scores = cp.matmul(q, cp.transpose(k, (0, 1, 3, 2))) / np.sqrt(Dh)  # plain CuPy batched matmul
    causal_mask = cp.triu(cp.ones((T, T), dtype=cp.bool_), k=1)
    scores = cp.where(causal_mask, cp.float32(-1e9), scores)
    scores = scores - cp.max(scores, axis=-1, keepdims=True)
    attn = cp.exp(scores)
    attn = attn / cp.sum(attn, axis=-1, keepdims=True)

    out = cp.matmul(attn, v)
    out = cp.transpose(out, (0, 2, 1, 3)).reshape(B, T, D)
    return linear_tc(out, w[f"{prefix}.proj.weight"])  # tensor-core GEMM


def block_tc(x: cp.ndarray, w: dict, cfg: GQEConfig, layer_idx: int) -> cp.ndarray:
    prefix = f"blocks.{layer_idx}"
    ln1 = layer_norm(x, w[f"{prefix}.ln1.weight"], w[f"{prefix}.ln1.bias"])
    x = x + attention_tc(ln1, w, cfg, layer_idx)

    ln2 = layer_norm(x, w[f"{prefix}.ln2.weight"], w[f"{prefix}.ln2.bias"])
    h = linear_tc(ln2, w[f"{prefix}.mlp.0.weight"]) + w[f"{prefix}.mlp.0.bias"]
    h = gelu(h)
    h = linear_tc(h, w[f"{prefix}.mlp.2.weight"]) + w[f"{prefix}.mlp.2.bias"]
    return x + h


def policy_forward_tc(tokens_batch: cp.ndarray, w: dict, cfg: GQEConfig) -> cp.ndarray:
    """This is the 'policy_forward' kernel for stage4: batched, tensor-core-
    GEMM-backed linear layers throughout."""
    B, T = tokens_batch.shape
    tok_emb = w["tok_emb.weight"][tokens_batch]
    pos_emb = w["pos_emb"][:, :T, :]
    x = (tok_emb + pos_emb).astype(cp.float32)

    for layer_idx in range(cfg.n_layers):
        x = block_tc(x, w, cfg, layer_idx)

    x = layer_norm(x, w["ln_f.weight"], w["ln_f.bias"])
    return linear_tc(x, w["head.weight"])


def run_multistream_sampling_loop(
    n_streams: int, batch_size: int, w: dict, cfg: GQEConfig, seed: int
) -> list[cp.ndarray]:
    """Run n_streams independent policy_forward batches concurrently on
    separate CUDA streams -- models GQE's need to generate/evaluate several
    independent candidate batches per training iteration (e.g. different
    exploration temperatures, or overlapping generation with the previous
    iteration's VQE energy evaluation)."""
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(n_streams)]
    results = [None] * n_streams

    for i, stream in enumerate(streams):
        tokens_torch = sample_tokens(batch_size, cfg, seed=seed + i)
        tokens_gpu = cp.asarray(tokens_torch.numpy())
        with stream:
            results[i] = policy_forward_tc(tokens_gpu, w, cfg)

    for stream in streams:
        stream.synchronize()

    return results


def upload_weights(weights_np: dict) -> dict:
    return {k: cp.asarray(v) for k, v in weights_np.items()}


def main():
    ap = argparse.ArgumentParser(description="Stage 4 tensor-core + multi-stream GQE policy forward pass.")
    add_gqe_args(ap)
    ap.add_argument("--n-streams", type=int, default=1,
                     help="If >1, run this many independent sampling batches concurrently on separate streams.")
    args = ap.parse_args()

    cfg = GQEConfig(
        vocab_size=args.vocab_size, seq_len=args.seq_len, d_model=args.d_model,
        n_layers=args.n_layers, n_heads=args.n_heads,
    )
    model = deterministic_weights(cfg, seed=args.seed)
    weights_np = export_weights_numpy(model)
    weights_gpu = upload_weights(weights_np)

    if args.n_streams <= 1:
        tokens_torch = sample_tokens(args.batch_size, cfg, seed=args.seed + 1)
        tokens_gpu = cp.asarray(tokens_torch.numpy())

        t0 = time.perf_counter()
        logits = policy_forward_tc(tokens_gpu, weights_gpu, cfg)
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()

        print(f"[stage4_tc] GQE policy_forward (tensor-core): batch={args.batch_size} "
              f"seq_len={cfg.seq_len} d_model={cfg.d_model} n_layers={cfg.n_layers}")
        print(f"[stage4_tc] logits shape = {logits.shape}")
        print(f"[stage4_tc] wall time = {(t1 - t0) * 1e3:.3f} ms")
    else:
        t0 = time.perf_counter()
        results = run_multistream_sampling_loop(args.n_streams, args.batch_size, weights_gpu, cfg, args.seed)
        t1 = time.perf_counter()

        print(f"[stage4_tc] {args.n_streams} concurrent streams, batch={args.batch_size} each")
        print(f"[stage4_tc] output shapes = {[r.shape for r in results]}")
        print(f"[stage4_tc] wall time = {(t1 - t0) * 1e3:.3f} ms")


if __name__ == "__main__":
    main()
