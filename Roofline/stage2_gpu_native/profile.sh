#!/usr/bin/env bash
# Profile stage2_gpu_native with Nsight Systems (timeline) and Nsight Compute (roofline).
# Compare against stage1's profiles/ to see the effect of fusion/batching/no-sync.
#
# TROUBLESHOOTING: if ncu fails with "Profiling failed because a driver
# resource was unavailable" even after the dcgmi pause below, the GPU is
# likely contended by another job -- check `nvidia-smi` and see
# ENV_SETUP.md for how to get a dedicated allocation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p profiles

MOLECULE="${MOLECULE:-h2}"
DEPTH="${DEPTH:-3}"
BATCH="${BATCH:-32}"

if command -v dcgmi &> /dev/null; then
    dcgmi profile --pause || true
    trap 'dcgmi profile --resume || true' EXIT
fi

echo "=== Nsight Systems timeline: VQE fused hand-rolled (molecule=$MOLECULE depth=$DEPTH) ==="
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=profiles/stage2_vqe_fused_timeline \
    --force-overwrite=true \
    python3 vqe_fused.py --molecule "$MOLECULE" --depth "$DEPTH"

echo
echo "=== Nsight Compute roofline: VQE fused gate-application + expval kernels ==="
ncu \
    --set roofline \
    --export profiles/stage2_vqe_fused_roofline \
    --force-overwrite \
    python3 vqe_fused.py --molecule "$MOLECULE" --depth "$DEPTH"

echo
echo "=== Nsight Systems timeline: VQE cuStateVec comparison (molecule=$MOLECULE depth=$DEPTH) ==="
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=profiles/stage2_vqe_custatevec_timeline \
    --force-overwrite=true \
    python3 vqe_custatevec.py --molecule "$MOLECULE" --depth "$DEPTH"

echo
echo "=== Nsight Compute roofline: VQE cuStateVec kernels ==="
ncu \
    --set roofline \
    --export profiles/stage2_vqe_custatevec_roofline \
    --force-overwrite \
    python3 vqe_custatevec.py --molecule "$MOLECULE" --depth "$DEPTH"

echo
echo "=== Nsight Systems timeline: GQE batched (batch=$BATCH) ==="
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=profiles/stage2_gqe_timeline \
    --force-overwrite=true \
    python3 gqe_batched.py --batch-size "$BATCH"

echo
echo "=== Nsight Compute roofline: GQE batched policy_forward kernel ==="
ncu \
    --set roofline \
    --export profiles/stage2_gqe_roofline \
    --force-overwrite \
    python3 gqe_batched.py --batch-size "$BATCH"

echo
echo "Done. Reports in profiles/:"
ls -la profiles/
echo
echo "CSV export for plot_roofline.py:"
echo "  ncu --import profiles/stage2_vqe_fused_roofline.ncu-rep --csv --page raw > profiles/stage2_vqe_fused_roofline.csv"
echo "  ncu --import profiles/stage2_gqe_roofline.ncu-rep --csv --page raw > profiles/stage2_gqe_roofline.csv"
