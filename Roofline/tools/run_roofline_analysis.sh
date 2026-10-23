#!/usr/bin/env bash
# Full roofline analysis, end to end, on whatever GPU this runs on:
# stage0 calibration, then every stage's profile.sh (nsys + ncu + hierarchical
# roofline plots) at roofline-friendly sizes, each stage's correctness tests,
# and finally a numerical check that the PNGs match the Nsight Compute GUI.
#
# Run it on a DEDICATED, idle GPU (see CLAUDE.md "Re-running the full
# roofline analysis"), with the Python environment from ENV_SETUP.md already
# active and ncu/nsys/nvcc on PATH:
#
#   tools/run_roofline_analysis.sh 2>&1 | tee roofline_analysis.log
#
# Sizes and arch can be overridden from the environment:
#   QUBITS=12 D_MODEL=128 BATCH=256 GQE_BATCH=256 CUDA_ARCH=sm_80 tools/run_roofline_analysis.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export QUBITS="${QUBITS:-12}" D_MODEL="${D_MODEL:-128}" BATCH="${BATCH:-256}" GQE_BATCH="${GQE_BATCH:-256}"
# Default the CUDA arch to this GPU's compute capability (8.0 -> sm_80).
export CUDA_ARCH="${CUDA_ARCH:-sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)}"

hostname
nvidia-smi --query-gpu=name,compute_cap,utilization.gpu,memory.used --format=csv
echo "QUBITS=$QUBITS D_MODEL=$D_MODEL BATCH=$BATCH GQE_BATCH=$GQE_BATCH CUDA_ARCH=$CUDA_ARCH"

echo "##### stage0 calibration"
(cd "$REPO/stage0_calibration" && ./run_calibration.sh)

for s in stage1_naive stage2_gpu_native stage3_memory_opt stage4_compute_opt; do
    cd "$REPO/$s"
    [ -x build.sh ] && ./build.sh
    echo "##### $s profile.sh"
    SECONDS=0
    ./profile.sh
    echo "##### $s profile.sh exit $? after ${SECONDS}s"
    echo "##### $s correctness tests"
    timeout 420 python3 -m pytest -q test_correctness.py 2>&1 | tail -8
done

echo "##### PNG vs Nsight Compute GUI check"
cd "$REPO"
python3 tools/verify_vs_gui.py stage*_*/profiles/stage*_roofline.csv | grep -E "^===|ceilings max|kernels compared|OVERALL"
echo "##### ALL DONE"
