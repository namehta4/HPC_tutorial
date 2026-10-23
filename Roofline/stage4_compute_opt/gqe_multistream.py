#!/usr/bin/env python3
"""Stage 4: fused tensor-core + multi-stream GQE policy forward pass.

Run ./build.sh first.

What stage3's profile says: with the linear layers on tensor cores, those
GEMMs are only ~7% of the forward pass's GPU time. The other ~90% is ~175
tiny memory-bound CuPy kernels -- bias adds, GELU, the pieces of layernorm
and softmax, transposes, fp16<->fp32 casts -- each reading and writing the
activations from HBM. So stage4 fuses that glue into the kernels that
already hold the data (kernels/policy_forward_gemm.cu), cutting a forward
pass from ~180 kernel launches to ~16:

  1. FUSED TENSOR-CORE LINEAR LAYERS. Every linear layer (`qkv`, `proj`,
     both MLP layers, `head`) is one call to `gemm_tc_fp16_fused`, a
     hand-written WMMA (Warp Matrix Multiply-Accumulate) kernel that runs
     on the GPU's tensor cores and applies the layer's epilogue -- bias,
     GELU, or the residual add -- before writing its output. Each layer is
     ONE GEMM over all batch*seq_len rows; multiplying each row on its own
     (as a 16-row-padded tile) would waste 15/16 of the tensor-core work.

  2. FUSED LAYERNORM AND ATTENTION. Each layernorm is one kernel (the first
     one fused with the embedding lookup), and causal self-attention is one
     kernel per (batch, head) doing scores, mask, softmax and attn @ v in
     shared memory. Attention stays on CUDA cores: at seq_len 8 each head is
     an 8x8x32 problem, far too small for tensor cores to pay off.

  3. MULTI-STREAM SCALING. `run_multistream_sampling_loop` demonstrates the
     GQE outer loop's actual concurrency pattern: sampling several
     INDEPENDENT batches of candidate gate sequences (e.g. for different
     temperature/exploration settings, or simply to pipeline generation
     with downstream VQE energy evaluation) on separate CUDA streams, so
     they execute concurrently on the GPU instead of one-after-another.

Precision: GEMM inputs are fp16; accumulation, epilogues, layernorm and
softmax are fp32, and the residual stream is kept in fp32.
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

# Epilogue modes of gemm_tc_fp16_fused (must match kernels/policy_forward_gemm.cu).
EPI_STORE_F16, EPI_BIAS_GELU_F16, EPI_RESIDUAL_ADD_F32, EPI_STORE_F32 = 0, 1, 2, 3
LN_EPS = 1e-5
LINEAR_WEIGHTS = ("qkv.weight", "proj.weight", "mlp.0.weight", "mlp.2.weight", "head.weight")


def _get_module(cubin_name: str) -> cp.RawModule:
    if cubin_name not in _MODULE_CACHE:
        path = KERNELS_DIR / cubin_name
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run ./build.sh first")
        _MODULE_CACHE[cubin_name] = cp.RawModule(path=str(path))
    return _MODULE_CACHE[cubin_name]


def _kernel(name: str):
    return _get_module("policy_forward_gemm.cubin").get_function(name)


def _pad_to_multiple(n: int, m: int = 16) -> int:
    return ((n + m - 1) // m) * m


def linear_tc(x16: cp.ndarray, weight16: cp.ndarray, out: cp.ndarray, mode: int, bias: cp.ndarray | None = None):
    """out = epilogue(x16 @ weight16.T) on tensor cores, as ONE GEMM over all
    rows. x16: (M, K) fp16, weight16: (N, K) fp16 (PyTorch Linear layout),
    out: (M, N), fp16 or fp32 depending on `mode` (see EPI_*)."""
    M, K = x16.shape
    N = weight16.shape[0]
    grid = ((M + 31) // 32, (N + 31) // 32)
    _kernel("gemm_tc_fp16_fused")(
        grid, (128,),
        (x16, weight16, bias if bias is not None else out, out,
         np.int32(M), np.int32(N), np.int32(K), np.int32(mode), np.int32(bias is not None)),
    )


def layer_norm(x32: cp.ndarray, weight: cp.ndarray, bias: cp.ndarray, out16: cp.ndarray):
    """out16[row] = layernorm(x32[row]); one warp per row."""
    M, D = x32.shape
    _kernel("layernorm_f32_to_f16")(((M + 3) // 4,), (128,),
                                    (x32, weight, bias, out16, np.int32(M), np.int32(D), np.float32(LN_EPS)))


def attention(qkv16: cp.ndarray, out16: cp.ndarray, B: int, cfg: GQEConfig):
    """Fused causal self-attention, one CUDA block per (batch, head)."""
    T, D, H = cfg.seq_len, cfg.d_model, cfg.n_heads
    Dh = D // H
    smem = (3 * T * Dh + T * T) * 4
    _kernel("causal_attention")((B * H,), (128,),
                                (qkv16, out16, np.int32(T), np.int32(D), np.int32(H), np.float32(1.0 / np.sqrt(Dh))),
                                shared_mem=smem)


def policy_forward_tc(tokens_batch: cp.ndarray, w: dict, cfg: GQEConfig) -> cp.ndarray:
    """This is the 'policy_forward' kernel for stage4: ~16 fused kernel
    launches per forward pass, all linear layers on tensor cores."""
    B, T = tokens_batch.shape
    D, F, V = cfg.d_model, cfg.d_ff, cfg.vocab_size
    if D % 16 or F % 16 or V % 16:
        raise ValueError(f"d_model ({D}), d_ff ({F}) and vocab_size ({V}) must be multiples of 16 "
                         "for the WMMA fp16 tile")
    M = B * T
    Mp = _pad_to_multiple(M)  # GEMM row count must be a multiple of 16; extra rows are padding

    x = cp.empty((Mp, D), dtype=cp.float32)  # residual stream
    h16 = cp.empty((Mp, D), dtype=cp.float16)  # layernorm output = next GEMM's input
    qkv16 = cp.empty((Mp, 3 * D), dtype=cp.float16)
    # Attention only writes real rows; zero the padding rows so they stay finite.
    att16 = (cp.zeros if Mp != M else cp.empty)((Mp, D), dtype=cp.float16)
    ff16 = cp.empty((Mp, F), dtype=cp.float16)
    logits = cp.empty((Mp, V), dtype=cp.float32)

    tokens = cp.ascontiguousarray(tokens_batch, dtype=cp.int64).reshape(-1)
    _kernel("embed_layernorm")(
        ((Mp + 3) // 4,), (128,),
        (tokens, w["tok_emb.weight"], w["pos_emb"], w["blocks.0.ln1.weight"], w["blocks.0.ln1.bias"],
         x, h16, np.int32(M), np.int32(Mp), np.int32(T), np.int32(D), np.float32(LN_EPS)),
    )

    for i in range(cfg.n_layers):
        p = f"blocks.{i}"
        if i > 0:
            layer_norm(x, w[f"{p}.ln1.weight"], w[f"{p}.ln1.bias"], h16)
        linear_tc(h16, w[f"{p}.attn.qkv.weight"], qkv16, EPI_STORE_F16)
        attention(qkv16, att16, B, cfg)
        linear_tc(att16, w[f"{p}.attn.proj.weight"], x, EPI_RESIDUAL_ADD_F32)  # x += attn(...)

        layer_norm(x, w[f"{p}.ln2.weight"], w[f"{p}.ln2.bias"], h16)
        linear_tc(h16, w[f"{p}.mlp.0.weight"], ff16, EPI_BIAS_GELU_F16, w[f"{p}.mlp.0.bias"])
        linear_tc(ff16, w[f"{p}.mlp.2.weight"], x, EPI_RESIDUAL_ADD_F32, w[f"{p}.mlp.2.bias"])  # x += mlp(...)

    layer_norm(x, w["ln_f.weight"], w["ln_f.bias"], h16)
    linear_tc(h16, w["head.weight"], logits, EPI_STORE_F32)
    return logits[:M].reshape(B, T, V)


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
    """Linear-layer weights as fp16 (tensor-core GEMM inputs, PyTorch (out, in)
    layout); embeddings, biases and layernorm parameters as fp32."""
    return {k: cp.asarray(v, dtype=cp.float16 if k.endswith(LINEAR_WEIGHTS) else cp.float32)
            for k, v in weights_np.items()}


def main():
    ap = argparse.ArgumentParser(description="Stage 4 fused tensor-core + multi-stream GQE policy forward pass.")
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

        print(f"[stage4_tc] GQE policy_forward (fused tensor-core): batch={args.batch_size} "
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
