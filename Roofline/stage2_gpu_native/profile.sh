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
# Larger, roofline-friendly sizes (README §6): QUBITS=12 D_MODEL=128 BATCH=256 ./profile.sh
# QUBITS overrides MOLECULE with a random 12-qubit Hamiltonian.
QUBITS="${QUBITS:-}"
D_MODEL="${D_MODEL:-32}"
if [ -n "$QUBITS" ]; then
    VQE_SIZE=(--molecule random --qubits "$QUBITS")
else
    VQE_SIZE=(--molecule "$MOLECULE")
fi
BATCH="${BATCH:-32}"

if command -v dcgmi &> /dev/null; then
    dcgmi profile --pause || true
    trap 'dcgmi profile --resume || true' EXIT
fi

echo "=== Nsight Systems timeline: VQE fused hand-rolled (${VQE_SIZE[*]} depth=$DEPTH) ==="
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=profiles/stage2_vqe_fused_timeline \
    --force-overwrite=true \
    python3 vqe_fused.py "${VQE_SIZE[@]}" --depth "$DEPTH"

echo
echo "=== Nsight Compute roofline: VQE fused gate-application + expval kernels ==="
ncu \
    --set roofline \
    --export profiles/stage2_vqe_fused_roofline \
    --force-overwrite \
    python3 vqe_fused.py "${VQE_SIZE[@]}" --depth "$DEPTH"

echo
echo "=== Nsight Systems timeline: VQE cuStateVec comparison (${VQE_SIZE[*]} depth=$DEPTH) ==="
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=profiles/stage2_vqe_custatevec_timeline \
    --force-overwrite=true \
    python3 vqe_custatevec.py "${VQE_SIZE[@]}" --depth "$DEPTH"

echo
echo "=== Nsight Compute roofline: VQE cuStateVec kernels ==="
ncu \
    --set roofline \
    --export profiles/stage2_vqe_custatevec_roofline \
    --force-overwrite \
    python3 vqe_custatevec.py "${VQE_SIZE[@]}" --depth "$DEPTH"

echo
echo "=== Nsight Systems timeline: GQE batched (batch=$BATCH) ==="
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=profiles/stage2_gqe_timeline \
    --force-overwrite=true \
    python3 gqe_batched.py --batch-size "$BATCH" --d-model "$D_MODEL"

echo
echo "=== Nsight Compute roofline: GQE batched policy_forward kernel ==="
ncu \
    --set roofline \
    --export profiles/stage2_gqe_roofline \
    --force-overwrite \
    python3 gqe_batched.py --batch-size "$BATCH" --d-model "$D_MODEL"

echo
echo "=== Hierarchical roofline plots (L1/L2/HBM, same metrics as the ncu GUI) ==="
# Exports each report to profiles/<name>.csv, writes per-report plots to
# profiles/roofline_plots/, and refreshes the cross-stage
# ../roofline_hierarchical_{vqe,gqe}.png and the HBM-only
# ../roofline_story_{vqe,gqe}_<n>.png slides from every stage profiled so far.
python3 ../plot_hierarchical_roofline.py \
    profiles/stage2_vqe_fused_roofline.ncu-rep \
    profiles/stage2_vqe_custatevec_roofline.ncu-rep \
    profiles/stage2_gqe_roofline.ncu-rep \
    --compare vqe --compare gqe \
    || echo "WARNING: roofline plotting failed (the .ncu-rep reports are fine); retry with: python3 ../plot_hierarchical_roofline.py profiles/*.ncu-rep"

echo
echo "Done. Reports and CSVs in profiles/, roofline plots in profiles/roofline_plots/:"
ls -la profiles/ profiles/roofline_plots/
