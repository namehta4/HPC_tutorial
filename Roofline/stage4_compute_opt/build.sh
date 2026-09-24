#!/usr/bin/env bash
# Compile stage4's hand-written CUDA C++ kernels (batched VQE gate/expval
# kernels + WMMA tensor-core GEMM) to .cubin files loadable by cupy.RawModule.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ARCH="${CUDA_ARCH:-sm_80}"

echo "Compiling kernels for -arch=$ARCH ..."
nvcc -arch="$ARCH" -O3 --use_fast_math -cubin kernels/gate_apply_tc.cu -o kernels/gate_apply_tc.cubin
nvcc -arch="$ARCH" -O3 -cubin kernels/policy_forward_gemm.cu -o kernels/policy_forward_gemm.cubin

echo "Built:"
ls -la kernels/*.cubin
