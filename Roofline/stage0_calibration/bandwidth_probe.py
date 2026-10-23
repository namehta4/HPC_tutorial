#!/usr/bin/env python3
"""Measure achievable GPU memory bandwidth with a STREAM-like Triad kernel.

Why this exists: the roofline model's "memory roof" is a ceiling of
achievable GB/s. Vendor spec sheets quote *theoretical peak* bandwidth
(e.g. an A100-PCIe's HBM2e peak is ~1935 GB/s), but real kernels never quite
reach that -- ECC overhead, imperfect access patterns, and other traffic
between the memory controller and the SMs all eat into it. A roofline plot
drawn against the spec-sheet peak overstates how far below the roof your
kernel actually is. This probe measures the number *this specific GPU*
actually delivers, so stage1-4 kernels are judged against reality.

Triad:  a[i] = b[i] + scalar * c[i]
Bytes moved per element: 2 loads + 1 store = 3 * sizeof(dtype)
Achieved bandwidth (GB/s) = total_bytes / elapsed_seconds / 1e9

The Triad must be ONE fused kernel for that byte count to be true. The
obvious CuPy one-liner `a[:] = b + scalar * c` is three kernels
(scalar*c -> tmp, b+tmp -> tmp2, copy tmp2 -> a) that move 7 arrays' worth
of data, not 3 -- counting 3 under-reports bandwidth by ~2.3x (an A100
"measured" ~590 GB/s that way vs ~1,370 GB/s fused). Hence the
ElementwiseKernel below.

Usage:
    python3 bandwidth_probe.py [--size-mb 1024] [--iters 50] [--dtype fp64]
"""
from __future__ import annotations

import argparse
import json
import time

import cupy as cp
import numpy as np


# One fused kernel: exactly 2 loads + 1 store per element (see module docstring).
_triad = cp.ElementwiseKernel("T b, T c, T scalar", "T a", "a = b + scalar * c", "stream_triad")


def run_triad(n_elements: int, dtype: np.dtype, iters: int, warmup: int) -> float:
    a = cp.empty(n_elements, dtype=dtype)
    b = cp.random.random(n_elements).astype(dtype)
    c = cp.random.random(n_elements).astype(dtype)
    scalar = dtype(3.14159)

    for _ in range(warmup):
        _triad(b, c, scalar, a)
    cp.cuda.Stream.null.synchronize()

    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    for _ in range(iters):
        _triad(b, c, scalar, a)
    end.record()
    end.synchronize()

    # A fast wrong kernel is no calibration; spot-check the result.
    idx = cp.arange(0, n_elements, max(1, n_elements // 4096))
    expected = b[idx].astype(np.float64) + float(scalar) * c[idx].astype(np.float64)
    if not cp.allclose(a[idx].astype(np.float64), expected, rtol=1e-2 if dtype == np.float16 else 1e-6):
        raise RuntimeError("stream_triad produced wrong results")

    elapsed_ms = cp.cuda.get_elapsed_time(start, end)
    elapsed_s = (elapsed_ms / 1000.0) / iters

    bytes_per_iter = 3 * n_elements * np.dtype(dtype).itemsize
    gb_per_s = bytes_per_iter / elapsed_s / 1e9
    return gb_per_s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size-mb", type=int, default=1024, help="Size of each of the 3 arrays, in MB.")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--dtype", choices=["fp64", "fp32", "fp16"], default="fp64")
    ap.add_argument("--out", type=str, default=None, help="Optional path to write JSON result.")
    args = ap.parse_args()

    dtype_map = {"fp64": np.float64, "fp32": np.float32, "fp16": np.float16}
    dtype = dtype_map[args.dtype]
    n_elements = (args.size_mb * 1024 * 1024) // np.dtype(dtype).itemsize

    dev = cp.cuda.Device(0)
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    print(f"Device: {props['name'].decode()}")
    print(f"Running Triad: a = b + s*c, {n_elements:,} elements ({args.dtype}), {args.iters} iters")

    gb_per_s = run_triad(n_elements, dtype, args.iters, args.warmup)
    print(f"Achieved bandwidth: {gb_per_s:.1f} GB/s")

    result = {
        "kernel": "stream_triad",
        "dtype": args.dtype,
        "n_elements": int(n_elements),
        "achieved_gbps": gb_per_s,
        "device": props["name"].decode(),
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
