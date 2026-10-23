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

echo "=== Nsight Systems timeline: VQE memory-optimized, fp64 (${VQE_SIZE[*]} depth=$DEPTH) ==="
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=profiles/stage3_vqe_fp64_timeline \
    --force-overwrite=true \
    python3 vqe_memopt.py "${VQE_SIZE[@]}" --depth "$DEPTH" --precision fp64

echo
echo "=== Nsight Compute roofline: VQE memory-optimized fp64 kernels ==="
ncu \
    --set roofline \
    --export profiles/stage3_vqe_fp64_roofline \
    --force-overwrite \
    python3 vqe_memopt.py "${VQE_SIZE[@]}" --depth "$DEPTH" --precision fp64

echo
echo "=== Nsight Systems timeline: VQE memory-optimized, fp32 (${VQE_SIZE[*]} depth=$DEPTH) ==="
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=profiles/stage3_vqe_fp32_timeline \
    --force-overwrite=true \
    python3 vqe_memopt.py "${VQE_SIZE[@]}" --depth "$DEPTH" --precision fp32

echo
echo "=== Nsight Compute roofline: VQE memory-optimized fp32 kernels ==="
ncu \
    --set roofline \
    --export profiles/stage3_vqe_fp32_roofline \
    --force-overwrite \
    python3 vqe_memopt.py "${VQE_SIZE[@]}" --depth "$DEPTH" --precision fp32

echo
echo "=== Nsight Systems timeline: GQE fp16 (batch=$BATCH) ==="
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=profiles/stage3_gqe_fp16_timeline \
    --force-overwrite=true \
    python3 gqe_batched_fp16.py --batch-size "$BATCH" --d-model "$D_MODEL"

echo
echo "=== Nsight Compute roofline: GQE fp16 policy_forward kernel ==="
ncu \
    --set roofline \
    --export profiles/stage3_gqe_fp16_roofline \
    --force-overwrite \
    python3 gqe_batched_fp16.py --batch-size "$BATCH" --d-model "$D_MODEL"

echo
echo "=== Hierarchical roofline plots (L1/L2/HBM, same metrics as the ncu GUI) ==="
# Exports each report to profiles/<name>.csv, writes per-report plots to
# profiles/roofline_plots/, and refreshes the cross-stage
# ../roofline_hierarchical_{vqe,gqe}.png and the HBM-only
# ../roofline_story_{vqe,gqe}_<n>.png slides from every stage profiled so far.
python3 ../plot_hierarchical_roofline.py \
    profiles/stage3_vqe_fp64_roofline.ncu-rep \
    profiles/stage3_vqe_fp32_roofline.ncu-rep \
    profiles/stage3_gqe_fp16_roofline.ncu-rep \
    --compare vqe --compare gqe \
    || echo "WARNING: roofline plotting failed (the .ncu-rep reports are fine); retry with: python3 ../plot_hierarchical_roofline.py profiles/*.ncu-rep"

echo
echo "Done. Reports and CSVs in profiles/, roofline plots in profiles/roofline_plots/:"
ls -la profiles/ profiles/roofline_plots/
