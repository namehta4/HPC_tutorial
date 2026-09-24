#!/usr/bin/env bash
# Run both calibration probes and save results to RESULTS.md.
#
# IMPORTANT: run this on a dedicated (allocated) compute node, not a login
# node or a node shared with other jobs -- on Perlmutter that means:
#
#   salloc -N 1 -C gpu -G 1 -q interactive -t 00:20:00 -A <your_project>
#
# A shared/contended GPU will report bandwidth and FLOP/s numbers far below
# the hardware's real capability (we saw this directly: on a login-node GPU
# already at 100% utilization from another job, Triad bandwidth measured
# ~190 GB/s against an A100's ~1300-1500 GB/s achievable ceiling). Running
# calibration on a contended GPU silently miscalibrates every roofline plot
# downstream, so it is worth the extra step of grabbing a dedicated node.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== NVIDIA-SMI snapshot (check GPU is idle before trusting these numbers) ==="
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv

echo
echo "=== Bandwidth probe (STREAM Triad) ==="
python3 bandwidth_probe.py --size-mb 1024 --iters 50 --dtype fp64 --out bandwidth_fp64.json

echo
echo "=== GEMM peak-FLOPs probe ==="
python3 gemm_peak_probe.py --size 8192 --iters 20 --out gemm_peak.json

echo
echo "Done. Raw JSON in bandwidth_fp64.json / gemm_peak.json."
echo "Fill these numbers into RESULTS.md and into the ceiling values used by plot_roofline.py."
