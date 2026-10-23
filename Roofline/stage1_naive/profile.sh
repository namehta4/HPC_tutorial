#!/usr/bin/env bash
# Profile stage1_naive with Nsight Systems (timeline) and Nsight Compute (roofline).
#
# Run this from a compute node with a GPU (see README.md / stage0_calibration
# for how to allocate one on Perlmutter). Reports land in ./profiles/.
#
# What to look at afterward:
#   - nsys: open profiles/stage1_vqe_timeline.nsys-rep in the Nsight Systems
#     GUI (or `nsys stats` it on the CLI). Expect a long row of short,
#     evenly-spaced kernel launches with visible GAPS between them (each gap
#     is the cp.cuda.Stream.null.synchronize() call) -- low "GPU busy %".
#   - ncu: open profiles/stage1_vqe_roofline.ncu-rep in the Nsight Compute
#     GUI's roofline section. Expect the gate-application and Pauli-matvec
#     kernels to land far below BOTH the memory and compute rooflines.
# See NOTES.md for the full explanation.
#
# TROUBLESHOOTING: if ncu fails partway through with "Profiling failed
# because a driver resource was unavailable" even after the dcgmi pause
# below, the GPU is likely under heavy contention from another job (check
# `nvidia-smi` -- if "Volatile GPU-Util" is near 100% from a process that
# isn't yours, that's it). ncu's multi-pass profiling is sensitive to this
# in a way plain execution isn't. Get a dedicated allocation (see
# ENV_SETUP.md) and re-run.
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
BATCH="${BATCH:-8}"

# Nsight Compute needs exclusive access to the GPU's hardware performance
# counters. On many HPC systems (this one included) NVIDIA's DCGM cluster
# monitoring daemon holds those counters by default, which makes ncu fail
# with "Profiling failed because a driver resource was unavailable." Pausing
# DCGM for the duration of this script (and resuming it afterward, even on
# failure, via the trap) fixes this. If dcgmi isn't installed or you don't
# have permission to run it, this is a harmless no-op -- see README.md
# "Troubleshooting" for what to do if ncu still fails after this.
if command -v dcgmi &> /dev/null; then
    dcgmi profile --pause || true
    trap 'dcgmi profile --resume || true' EXIT
fi

echo "=== Nsight Systems timeline: VQE naive (${VQE_SIZE[*]} depth=$DEPTH) ==="
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=profiles/stage1_vqe_timeline \
    --force-overwrite=true \
    python3 vqe_naive.py "${VQE_SIZE[@]}" --depth "$DEPTH"

echo
echo "=== Nsight Compute roofline: VQE naive gate-application + expval kernels ==="
# --set roofline captures the metrics needed for the roofline chart;
# --launch-count limits how many kernel launches we capture detail for,
# since stage1 launches MANY tiny kernels (one per gate) and a full-detail
# capture of every single one would be extremely slow.
ncu \
    --set roofline \
    --launch-count 60 \
    --export profiles/stage1_vqe_roofline \
    --force-overwrite \
    python3 vqe_naive.py "${VQE_SIZE[@]}" --depth "$DEPTH"

echo
echo "=== Nsight Systems timeline: GQE naive (batch=$BATCH) ==="
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=profiles/stage1_gqe_timeline \
    --force-overwrite=true \
    python3 gqe_naive.py --batch-size "$BATCH" --d-model "$D_MODEL"

echo
echo "=== Nsight Compute roofline: GQE naive policy_forward kernel ==="
ncu \
    --set roofline \
    --launch-count 60 \
    --export profiles/stage1_gqe_roofline \
    --force-overwrite \
    python3 gqe_naive.py --batch-size "$BATCH" --d-model "$D_MODEL"

echo
echo "=== Hierarchical roofline plots (L1/L2/HBM, same metrics as the ncu GUI) ==="
# Exports each report to profiles/<name>.csv, writes per-report plots to
# profiles/roofline_plots/, and refreshes the cross-stage
# ../roofline_hierarchical_{vqe,gqe}.png and the HBM-only
# ../roofline_story_{vqe,gqe}_<n>.png slides from every stage profiled so far.
python3 ../plot_hierarchical_roofline.py \
    profiles/stage1_vqe_roofline.ncu-rep \
    profiles/stage1_gqe_roofline.ncu-rep \
    --compare vqe --compare gqe \
    || echo "WARNING: roofline plotting failed (the .ncu-rep reports are fine); retry with: python3 ../plot_hierarchical_roofline.py profiles/*.ncu-rep"

echo
echo "Done. Reports and CSVs in profiles/, roofline plots in profiles/roofline_plots/:"
ls -la profiles/ profiles/roofline_plots/
