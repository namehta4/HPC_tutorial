#!/usr/bin/env bash
# Profile stage4_compute_opt with Nsight Systems (timeline) and Nsight Compute (roofline).
# Run ./build.sh first. Compare against stage3's profiles/ to see the effect
# of grid-batched occupancy tuning (VQE) and tensor-core GEMM (GQE).
#
# TROUBLESHOOTING: if ncu fails with "Profiling failed because a driver
# resource was unavailable" even after the dcgmi pause below, the GPU is
# likely contended by another job -- check `nvidia-smi` and see
# ENV_SETUP.md for how to get a dedicated allocation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p profiles

if [ ! -f kernels/gate_apply_tc.cubin ]; then
    echo "kernels/*.cubin not found -- running ./build.sh first"
    ./build.sh
fi

MOLECULE="${MOLECULE:-h2}"
DEPTH="${DEPTH:-3}"
VQE_BATCH="${VQE_BATCH:-32}"
GQE_BATCH="${GQE_BATCH:-64}"
N_STREAMS="${N_STREAMS:-4}"

if command -v dcgmi &> /dev/null; then
    dcgmi profile --pause || true
    trap 'dcgmi profile --resume || true' EXIT
fi

echo "=== Nsight Systems timeline: VQE batched/occupancy-tuned (molecule=$MOLECULE depth=$DEPTH batch=$VQE_BATCH) ==="
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=profiles/stage4_vqe_timeline \
    --force-overwrite=true \
    python3 vqe_compute_opt.py --molecule "$MOLECULE" --depth "$DEPTH" --batch-size "$VQE_BATCH"

echo
echo "=== Nsight Compute roofline: VQE batched gate-application + expval kernels ==="
ncu \
    --set roofline \
    --export profiles/stage4_vqe_roofline \
    --force-overwrite \
    python3 vqe_compute_opt.py --molecule "$MOLECULE" --depth "$DEPTH" --batch-size "$VQE_BATCH"

echo
echo "=== Nsight Systems timeline: GQE tensor-core (batch=$GQE_BATCH) ==="
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=profiles/stage4_gqe_tc_timeline \
    --force-overwrite=true \
    python3 gqe_multistream.py --batch-size "$GQE_BATCH"

echo
echo "=== Nsight Compute roofline: GQE tensor-core policy_forward kernel ==="
ncu \
    --set roofline \
    --export profiles/stage4_gqe_tc_roofline \
    --force-overwrite \
    python3 gqe_multistream.py --batch-size "$GQE_BATCH"

echo
echo "=== Nsight Systems timeline: GQE multi-stream ($N_STREAMS streams x batch=$GQE_BATCH) ==="
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=profiles/stage4_gqe_multistream_timeline \
    --force-overwrite=true \
    python3 gqe_multistream.py --batch-size "$GQE_BATCH" --n-streams "$N_STREAMS"

echo
echo "Done. Reports in profiles/:"
ls -la profiles/
echo
echo "CSV export for plot_roofline.py:"
echo "  ncu --import profiles/stage4_vqe_roofline.ncu-rep --csv --page raw > profiles/stage4_vqe_roofline.csv"
echo "  ncu --import profiles/stage4_gqe_tc_roofline.ncu-rep --csv --page raw > profiles/stage4_gqe_tc_roofline.csv"
