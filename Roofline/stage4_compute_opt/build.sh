#!/usr/bin/env bash
# Compile stage4's hand-written CUDA C++ kernels (batched VQE gate/expval
# kernels + fused GQE kernels: WMMA tensor-core GEMM with epilogues, layernorm,
# attention) to .cubin files loadable by cupy.RawModule.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ARCH="${CUDA_ARCH:-sm_80}"

# -lineinfo records which .cu line each SASS instruction came from, for
# Nsight Compute's Source page; it doesn't change the generated code.
echo "Compiling kernels for -arch=$ARCH ..."
nvcc -arch="$ARCH" -O3 -lineinfo --use_fast_math -cubin kernels/gate_apply_tc.cu -o kernels/gate_apply_tc.cubin
nvcc -arch="$ARCH" -O3 -lineinfo -cubin kernels/policy_forward_gemm.cu -o kernels/policy_forward_gemm.cubin

echo "Built:"
ls -la kernels/*.cubin
