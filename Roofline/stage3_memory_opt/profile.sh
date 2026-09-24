#!/usr/bin/env bash
# Profile stage3_memory_opt with Nsight Systems (timeline) and Nsight Compute (roofline).
# Run ./build.sh first. Compare against stage2's profiles/ to see the effect
# of the whole-circuit-in-shared-memory kernel design.
#
# TROUBLESHOOTING: if ncu fails with "Profiling failed because a driver
# resource was unavailable" even after the dcgmi pause below, the GPU is
# likely contended by another job -- check `nvidia-smi` and see
# ENV_SETUP.md for how to get a dedicated allocation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p profiles

if [ ! -f kernels/gate_apply.cubin ]; then
    echo "kernels/*.cubin not found -- running ./build.sh first"
    ./build.sh
fi

MOLECULE="${MOLECULE:-h2}"
DEPTH="${DEPTH:-3}"
BATCH="${BATCH:-32}"

if command -v dcgmi &> /dev/null; then
    dcgmi profile --pause || true
    trap 'dcgmi profile --resume || true' EXIT
fi

echo "=== Nsight Systems timeline: VQE memory-optimized, fp64 (molecule=$MOLECULE depth=$DEPTH) ==="
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=profiles/stage3_vqe_fp64_timeline \
    --force-overwrite=true \
    python3 vqe_memopt.py --molecule "$MOLECULE" --depth "$DEPTH" --precision fp64

echo
echo "=== Nsight Compute roofline: VQE memory-optimized fp64 kernels ==="
ncu \
    --set roofline \
    --export profiles/stage3_vqe_fp64_roofline \
    --force-overwrite \
    python3 vqe_memopt.py --molecule "$MOLECULE" --depth "$DEPTH" --precision fp64

echo
echo "=== Nsight Systems timeline: VQE memory-optimized, fp32 (molecule=$MOLECULE depth=$DEPTH) ==="
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=profiles/stage3_vqe_fp32_timeline \
    --force-overwrite=true \
    python3 vqe_memopt.py --molecule "$MOLECULE" --depth "$DEPTH" --precision fp32

echo
echo "=== Nsight Compute roofline: VQE memory-optimized fp32 kernels ==="
ncu \
    --set roofline \
    --export profiles/stage3_vqe_fp32_roofline \
    --force-overwrite \
    python3 vqe_memopt.py --molecule "$MOLECULE" --depth "$DEPTH" --precision fp32

echo
echo "=== Nsight Systems timeline: GQE fp16 (batch=$BATCH) ==="
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=profiles/stage3_gqe_fp16_timeline \
    --force-overwrite=true \
    python3 gqe_batched_fp16.py --batch-size "$BATCH"

echo
echo "=== Nsight Compute roofline: GQE fp16 policy_forward kernel ==="
ncu \
    --set roofline \
    --export profiles/stage3_gqe_fp16_roofline \
    --force-overwrite \
    python3 gqe_batched_fp16.py --batch-size "$BATCH"

echo
echo "Done. Reports in profiles/:"
ls -la profiles/
echo
echo "CSV export for plot_roofline.py:"
echo "  ncu --import profiles/stage3_vqe_fp64_roofline.ncu-rep --csv --page raw > profiles/stage3_vqe_fp64_roofline.csv"
echo "  ncu --import profiles/stage3_vqe_fp32_roofline.ncu-rep --csv --page raw > profiles/stage3_vqe_fp32_roofline.csv"
echo "  ncu --import profiles/stage3_gqe_fp16_roofline.ncu-rep --csv --page raw > profiles/stage3_gqe_fp16_roofline.csv"
