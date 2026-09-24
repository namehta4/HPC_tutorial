#!/usr/bin/env python3
"""Measure achievable GEMM FLOP/s across precisions, via cuBLAS (through CuPy).

Why this exists: the roofline model's "compute roof" is a ceiling of
achievable FLOP/s. Like memory bandwidth, spec-sheet peak FLOP/s numbers
(e.g. A100's 19.5 TFLOP/s FP64, 156 TFLOP/s TF32-tensor-core, 312 TFLOP/s
FP16-tensor-core) are theoretical maxima that real kernels approach but
rarely reach. This probe runs large square GEMMs -- the workload cuBLAS is
most heavily tuned for -- to find the practical ceiling for *this* GPU,
which is the number the tutorial's roofline plots actually use.

We sweep FP64, FP32, TF32 (FP32 storage, tensor-core matmul), and FP16 to
show the roofline's compute ceiling is not one number -- it depends on
precision, which is exactly why stage3/stage4 introduce mixed precision.

GEMM FLOP count: for C = A @ B with A (m x k), B (k x n): 2*m*n*k FLOPs.

Usage:
    python3 gemm_peak_probe.py [--size 8192] [--iters 20]
"""
from __future__ import annotations

import argparse
import json

import cupy as cp
import numpy as np


def run_gemm(m: int, n: int, k: int, dtype: np.dtype, iters: int, warmup: int,
             math_mode: int | None = None) -> float:
    handle = cp.cuda.device.get_cublas_handle()
    prev_mode = cp.cuda.cublas.getMathMode(handle)
    if math_mode is not None:
        cp.cuda.cublas.setMathMode(handle, math_mode)

    a = cp.random.random((m, k)).astype(dtype)
    b = cp.random.random((k, n)).astype(dtype)

    for _ in range(warmup):
        c = a @ b
    cp.cuda.Stream.null.synchronize()

    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    for _ in range(iters):
        c = a @ b
    end.record()
    end.synchronize()

    elapsed_ms = cp.cuda.get_elapsed_time(start, end)
    elapsed_s = (elapsed_ms / 1000.0) / iters

    if math_mode is not None:
        cp.cuda.cublas.setMathMode(handle, prev_mode)

    flops = 2 * m * n * k
    tflops = flops / elapsed_s / 1e12
    return tflops


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=8192, help="Square GEMM dimension (m=n=k).")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", type=str, default=None, help="Optional path to write JSON result.")
    args = ap.parse_args()

    dev = cp.cuda.Device(0)
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    print(f"Device: {props['name'].decode()}")
    print(f"GEMM size: {args.size}x{args.size}x{args.size}, {args.iters} iters\n")

    results = {}

    # FP64 -- no tensor cores on A100 for FP64 matmul (that's Hopper+); this
    # is the "pure CUDA-core" FLOP/s ceiling.
    tflops = run_gemm(args.size, args.size, args.size, np.float64, args.iters, args.warmup)
    print(f"FP64 (CUDA cores):        {tflops:6.2f} TFLOP/s")
    results["fp64"] = tflops

    # FP32 -- despite cuBLAS docs suggesting Ampere+ defaults to TF32
    # tensor-core matmul for FP32 inputs, the cuBLAS handle's math mode
    # measured here starts as CUBLAS_DEFAULT_MATH (no tensor cores), not
    # CUBLAS_TENSOR_OP_MATH -- so this must opt in explicitly via
    # cublasSetMathMode(CUBLAS_TF32_TENSOR_OP_MATH) or the FP32 number
    # silently measures plain CUDA-core FP32 (nearly identical to FP64).
    tflops = run_gemm(args.size, args.size, args.size, np.float32, args.iters, args.warmup,
                       math_mode=cp.cuda.cublas.CUBLAS_TENSOR_OP_MATH)
    print(f"FP32 (TF32 tensor cores): {tflops:6.2f} TFLOP/s")
    results["fp32_tf32"] = tflops

    # FP16 -- full tensor-core throughput.
    tflops = run_gemm(args.size, args.size, args.size, np.float16, args.iters, args.warmup)
    print(f"FP16 (tensor cores):      {tflops:6.2f} TFLOP/s")
    results["fp16"] = tflops

    results["device"] = props["name"].decode()
    results["gemm_size"] = args.size
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
