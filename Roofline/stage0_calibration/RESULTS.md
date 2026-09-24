# Stage 0 Calibration Results

This file records the *measured* (not spec-sheet) achievable bandwidth and
FLOP/s for the GPU you're actually running on. Every roofline plot in this
tutorial (`plot_roofline.py`) draws its ceilings from these numbers, not
from vendor marketing figures — that's the whole point of stage 0.

## How to fill this in

```bash
cd stage0_calibration
./run_calibration.sh
```

Then copy the printed numbers into the tables below.

**Run this on a dedicated, idle GPU.** On a shared/contended GPU the
numbers you get are meaningless as roofline ceilings — see the warning at
the top of `run_calibration.sh`. On Perlmutter:

```bash
salloc -N 1 -C gpu -G 1 -q interactive -t 00:20:00 -A <your_project>
```

## Example run (login-node GPU, contended by another job at 100% util)

This is what we measured while writing this tutorial, on a **shared** GPU
— included here only to show the *format*, and as a cautionary example of
why calibration matters. Re-run `run_calibration.sh` on your own dedicated
allocation and replace every number below.

| Metric | Measured (this run, contended GPU) | Notes |
|---|---|---|
| GPU | NVIDIA A100-PCIE-40GB (compute cap 8.0) | shared with another job at 100% util |
| Memory bandwidth (Triad, FP64) | ~192 GB/s | spec-sheet peak is ~1935 GB/s — **10x lower**, purely from contention |
| FP64 GEMM (cuBLAS, CUDA cores) | ~6.7 TFLOP/s | spec-sheet peak ~9.7 TFLOP/s |
| FP32 GEMM (cuBLAS, TF32 tensor cores) | ~9.6 TFLOP/s | spec-sheet peak ~156 TFLOP/s (also contention-limited) |
| FP16 GEMM (cuBLAS, tensor cores) | ~120 TFLOP/s | spec-sheet peak ~312 TFLOP/s |

## 2026-09-23 calibration run (Perlmutter, dedicated debug-QOS allocation) — CURRENT

| Metric | Measured value | Command used |
|---|---|---|
| GPU | NVIDIA A100-SXM4-40GB (compute cap 8.0) | `nvidia-smi --query-gpu=name --format=csv` |
| GPU utilization at time of run | 0% | `nvidia-smi --query-gpu=utilization.gpu --format=csv` |
| Memory bandwidth, achieved (GB/s), FP64 Triad | 589.5 GB/s | `python3 bandwidth_probe.py --size-mb 1024 --iters 50 --dtype fp64` |
| FP64 GEMM peak (TFLOP/s) | 18.64 TFLOP/s | `python3 gemm_peak_probe.py --size 8192 --iters 20` |
| FP32/TF32 GEMM peak (TFLOP/s) | **155.58 TFLOP/s** | `python3 gemm_peak_probe.py --size 8192 --iters 20` |
| FP16 GEMM peak (TFLOP/s) | 278.01 TFLOP/s | `python3 gemm_peak_probe.py --size 8192 --iters 20` |
| Ridge point, FP64 (FLOP/byte) = FP64 peak / BW peak | 18.64e12 / 589.5e9 ≈ 31.6 FLOP/byte | computed |
| Ridge point, FP32/TF32 (FLOP/byte) = FP32 peak / BW peak | 155.58e12 / 589.5e9 ≈ 263.9 FLOP/byte | computed |
| Ridge point, FP16 (FLOP/byte) = FP16 peak / BW peak | 278.01e12 / 589.5e9 ≈ 471.6 FLOP/byte | computed |

Node: `nid003281`, allocated via `salloc -N 1 -C gpu -G 1 -q debug -A nstaff -t 00:30:00`.
Raw JSON: `bandwidth_fp64.json`, `gemm_peak.json`.

**TF32 anomaly (found 2026-09-23, fixed same day):** an earlier calibration
run on this same GPU measured FP32/TF32 at only ~19.18 TFLOP/s — barely
above FP64 and nowhere near the A100's ~156 TFLOP/s spec. Root cause:
`cp.cuda.device.get_cublas_handle()`'s math mode defaulted to
`CUBLAS_DEFAULT_MATH` (plain FP32, no tensor cores), not
`CUBLAS_TENSOR_OP_MATH`, contrary to `gemm_peak_probe.py`'s old comment
claiming cuBLAS enables TF32 automatically for FP32 inputs on Ampere+ in
this CuPy/cuBLAS version (CuPy 14.2.0). Fixed by having
`gemm_peak_probe.py` explicitly call
`cp.cuda.cublas.setMathMode(handle, cp.cuda.cublas.CUBLAS_TENSOR_OP_MATH)`
around the FP32 GEMM (restored afterward). Confirmed fix: FP32/TF32 jumped
from ~19 TFLOP/s to 155.58 TFLOP/s, matching spec. **If you see an
FP32/TF32 number close to your FP64 number, your cuBLAS handle likely
isn't in tensor-op math mode either — check `gemm_peak_probe.py` is at
least this version before trusting the number.**

The **ridge point** is the arithmetic intensity (FLOP per byte moved) at
which a kernel transitions from memory-bound to compute-bound on this GPU.
Kernels to the left of the ridge point (lower AI) are memory-bound; kernels
to the right are compute-bound. You'll see in the stage NOTES.md files that
our gate-application and expectation-value kernels sit well to the *left*
of the ridge point in stage1 — that's the central diagnosis this tutorial
teaches you to make.
