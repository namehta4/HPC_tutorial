#!/usr/bin/env bash
# Compile stage3's hand-written CUDA C++ kernels to a .cubin loadable by
# cupy.RawModule. Run this once before vqe_memopt.py / test_correctness.py.
#
# -arch=sm_80 targets this tutorial's calibrated GPU (NVIDIA A100, compute
# capability 8.0 -- see stage0_calibration/RESULTS.md). If you're running on
# different hardware, change this to match (sm_90 for H100, sm_86 for
# RTX 30-series, sm_89 for RTX 40-series / L4, etc.) -- check with
# `nvidia-smi --query-gpu=compute_cap --format=csv`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ARCH="${CUDA_ARCH:-sm_80}"

echo "Compiling kernels for -arch=$ARCH ..."
nvcc -arch="$ARCH" -O3 --use_fast_math -cubin kernels/gate_apply.cu -o kernels/gate_apply.cubin
nvcc -arch="$ARCH" -O3 --use_fast_math -cubin kernels/expval_reduce.cu -o kernels/expval_reduce.cubin
nvcc -arch="$ARCH" -O3 --use_fast_math -cubin kernels/gate_apply_fp32.cu -o kernels/gate_apply_fp32.cubin
nvcc -arch="$ARCH" -O3 --use_fast_math -cubin kernels/expval_reduce_fp32.cu -o kernels/expval_reduce_fp32.cubin

echo "Built:"
ls -la kernels/*.cubin
