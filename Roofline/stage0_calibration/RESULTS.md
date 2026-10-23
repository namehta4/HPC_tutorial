# Stage 0 Calibration Results

This file records the *measured* (not spec-sheet) bandwidth and FLOP/s of
the GPU you're actually running on. The roofline plots draw ncu's
theoretical ceilings, and add the measured HBM bandwidth from here as a
dashed line, so you can see how much of the theoretical peak is really
achievable.

## How to fill this in

On a **dedicated, idle GPU**, since numbers from a shared GPU are
meaningless as ceilings:

```bash
salloc -N 1 -C gpu -G 1 -q interactive -t 00:20:00 -A <your_project>   # Perlmutter
cd stage0_calibration
./run_calibration.sh        # writes bandwidth_fp64.json and gemm_peak.json
```

Then replace the numbers below with yours.

## Reference run: A100-SXM4-40GB, Perlmutter (2026-10-22)

Dedicated interactive node `nid001152`, GPU at 0% utilization.

| Metric | Measured | Spec / theoretical |
|---|---|---|
| HBM bandwidth (fused FP64 STREAM Triad) | **1,361.8 GB/s** | 1,555 GB/s spec; ~1,535 GB/s ncu theoretical at its locked clock |
| FP64 GEMM (cuBLAS, FP64 tensor cores / DMMA) | 18.36 TFLOP/s | 19.5 TFLOP/s |
| FP32 GEMM (cuBLAS, TF32 tensor cores) | 151.04 TFLOP/s | 156 TFLOP/s |
| FP16 GEMM (cuBLAS, tensor cores) | 272.14 TFLOP/s | 312 TFLOP/s |

Ridge points (peak FLOP/s ÷ measured HBM bandwidth):

| Precision | Ridge point |
|---|---|
| FP64 GEMM (tensor cores) | 18.36e12 / 1361.8e9 ≈ 13.5 FLOP/byte |
| TF32 GEMM (tensor cores) | 151.04e12 / 1361.8e9 ≈ 110.9 FLOP/byte |
| FP16 GEMM (tensor cores) | 272.14e12 / 1361.8e9 ≈ 199.8 FLOP/byte |
| FP64 CUDA-core FMA (what VQE's elementwise kernels use; ncu theoretical, locked clock) | ~7.5e12 / 1535e9 ≈ 4.9 FLOP/byte |

The **ridge point** is the arithmetic intensity (FLOP per byte moved) at
which a kernel crosses from memory-bound to compute-bound on this GPU.
Kernels to its left are memory-bound; kernels to its right are
compute-bound.

## Sanity checks

- **Bandwidth must come from a single fused kernel.** `a[:] = b + s*c` in
  plain CuPy is three kernels that move 7 arrays' worth of data, not 3.
  Counting 3 under-reports bandwidth by ~2.3x (this GPU "measured" ~590 GB/s
  that way). `bandwidth_probe.py` uses one `ElementwiseKernel`.
- **FP64 GEMM uses tensor cores on A100.** 18 TFLOP/s is about 2x the
  A100's 9.7 TFLOP/s CUDA-core FP64 peak, so it is not the ceiling for
  elementwise FP64 kernels like VQE's; the roofline plots take that
  CUDA-core ceiling from ncu.
- **TF32 should be ~8x FP64 CUDA-core speed.** If your FP32/TF32 number is
  close to your FP64 number, cuBLAS isn't in tensor-op math mode.
  `gemm_peak_probe.py` sets `CUBLAS_TENSOR_OP_MATH` explicitly, because
  CuPy's cuBLAS handle doesn't default to it.
