#!/usr/bin/env bash
# One command from a fresh clone to roofline plots on this machine's GPU:
#
#   ./run_all.sh              # setup + preflight checks + full analysis (~45-75 min on an A100)
#   ./run_all.sh --check      # setup + preflight checks only (seconds; safe on a login node)
#   ./run_all.sh --force      # skip the idle-GPU / dedicated-allocation checks (numbers may be garbage)
#
# What it does:
#   1. Python: uses a python3 >= 3.10 that already has every dependency (on
#      Perlmutter, after `module load pytorch/2.13.0`). Otherwise it creates
#      ./.venv and pip-installs what's missing, with CuPy/cuQuantum matched to
#      `nvcc --version` (cu12 or cu13) and CPU-only PyTorch.
#   2. Preflight: nvcc/ncu/nsys on PATH, a working GPU from CuPy, not a login
#      node, an idle GPU, enough disk, enough Slurm time left, and a tiny ncu
#      run to prove this user can read the GPU's performance counters. Each
#      failure prints the fix and stops before wasting an hour.
#   3. Runs tools/run_roofline_analysis.sh (stage0 calibration, every stage's
#      build + profile + tests, cross-stage plots, PNG-vs-GUI check), logging
#      to roofline_analysis_<timestamp>.log, then prints a pass/fail summary
#      and where the plots are.
#
# Two things it can't do for you: get the GPU (on Perlmutter,
#   salloc -N 1 -C gpu -G 1 -q interactive -t 01:30:00 -A <project>
# ) and grant performance-counter access on a locked-down workstation (an
# admin must set NVreg_RestrictProfilingToAdminUsers=0; see ENV_SETUP.md §4).
#
# Environment overrides: QUBITS D_MODEL BATCH GQE_BATCH CUDA_ARCH (passed to
# the analysis), PYTHON (interpreter to use), VENV (default ./.venv),
# PERLMUTTER_PYTORCH_MODULE (default pytorch/2.13.0).
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$REPO/.venv}"
PERLMUTTER_PYTORCH_MODULE="${PERLMUTTER_PYTORCH_MODULE:-pytorch/2.13.0}"
MIN_FREE_GB=2          # ~700 MB of reports + CSVs + plots, with headroom
MIN_SLURM_MINUTES=80   # the full run takes 45-75 min on an A100

MODE=run
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --check) MODE=check ;;
        --force) FORCE=1 ;;
        -h|--help) sed -n '2,/^set -uo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $arg (see --help)"; exit 2 ;;
    esac
done

step() { echo; echo "=== $* ==="; }
ok()   { echo "  [ok]   $*"; }
warn() { echo "  [warn] $*"; }
# fail <what went wrong> <how to fix it>
fail() { echo "  [FAIL] $1"; echo; echo "  Fix: $2"; echo; exit 1; }
# fail unless --force, for checks that make numbers meaningless but don't break the run
fail_unless_forced() {
    if [ "$FORCE" = 1 ]; then warn "$1 (continuing because of --force)"; else fail "$1" "$2"; fi
}

# --- 1. Python environment ------------------------------------------------------

step "Python environment"

command -v nvcc &>/dev/null || fail "nvcc not found on PATH" \
    "install the CUDA toolkit (or load its module, e.g. 'module load cudatoolkit') and re-run."
CUDA_MAJOR="$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')"
case "$CUDA_MAJOR" in
    12|13) ok "CUDA toolkit $CUDA_MAJOR.x (nvcc $(command -v nvcc))" ;;
    *) fail "CUDA toolkit '$CUDA_MAJOR' from nvcc --version; this tutorial supports CUDA 12.x and 13.x" \
            "put a CUDA 12 or 13 nvcc first on PATH." ;;
esac

# import name -> pip requirement. torch is installed separately, from the CPU wheel index:
# it's only the GQE correctness oracle, so the multi-GB CUDA build isn't needed.
declare -A PIP_NAME=(
    [numpy]="numpy>=2.0" [scipy]="scipy>=1.9" [matplotlib]="matplotlib>=3.9" [pytest]="pytest>=8.0"
    [cupy]="cupy-cuda${CUDA_MAJOR}x" [cuquantum]="cuquantum-python-cu${CUDA_MAJOR}" [torch]="torch"
)
missing_modules() {  # prints the import names $1 can't import
    "$1" - "${!PIP_NAME[@]}" <<'EOF'
import importlib, sys
for m in sys.argv[1:]:
    try:
        importlib.import_module(m)
    except Exception:
        print(m)
EOF
}
new_enough() { "$1" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; }
find_python() {
    local py
    for py in ${PYTHON:-} python3 python3.13 python3.12 python3.11 python3.10 python; do
        if command -v "$py" &>/dev/null && new_enough "$py"; then command -v "$py"; return; fi
    done
}
activate_venv() { set +u; source "$VENV/bin/activate"; set -u; }

if [ -x "$VENV/bin/python3" ]; then
    activate_venv
    ok "using existing venv $VENV"
else
    PY="$(find_python)"
    if [ "${NERSC_HOST:-}" = perlmutter ] && { [ -z "$PY" ] || [ -n "$(missing_modules "$PY")" ]; }; then
        # Perlmutter's system python3 is 3.6; the pytorch module brings a modern
        # Python with numpy/scipy/matplotlib/torch already installed.
        set +u
        type module &>/dev/null || source "${MODULESHOME:-/opt/cray/pe/lmod/lmod}/init/bash"
        module load "$PERLMUTTER_PYTORCH_MODULE" &>/dev/null
        set -u
        ok "loaded module $PERLMUTTER_PYTORCH_MODULE"
        PY="$(find_python)"
    fi
    [ -n "$PY" ] || fail "no python3 >= 3.10 found" \
        "install Python 3.10+ (or point PYTHON=/path/to/python3 at one) and re-run."
    MISSING="$(missing_modules "$PY")"
    if [ -z "$MISSING" ]; then
        ok "$PY already has every dependency; not creating a venv"
        PY_DESC="$PY"
        # profile.sh and the stage scripts call a bare `python3`: make it this one.
        SHIM="$(mktemp -d)"; trap 'rm -rf "$SHIM"' EXIT
        ln -s "$PY" "$SHIM/python3"
        export PATH="$SHIM:$PATH"
    else
        echo "  missing: $(echo $MISSING)"
        echo "  creating $VENV (with access to $PY's packages) and installing them..."
        "$PY" -m venv --system-site-packages "$VENV" || fail "could not create $VENV" \
            "install your distro's python3-venv package, or set VENV to a writable path."
        activate_venv
    fi
fi

MISSING="$(missing_modules python3)"
if [ -n "$MISSING" ]; then
    REQS=()
    for m in $MISSING; do [ "$m" = torch ] || REQS+=("${PIP_NAME[$m]}"); done
    [ ${#REQS[@]} -gt 0 ] && { python3 -m pip install "${REQS[@]}" || fail "pip install ${REQS[*]} failed" \
        "check the error above (network access? disk quota?); 'rm -rf $VENV' and re-run to start clean."; }
    if [[ " $MISSING " == *" torch "* ]]; then
        python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu || fail \
            "pip install torch failed" "check the error above; 'rm -rf $VENV' and re-run to start clean."
    fi
    MISSING="$(missing_modules python3)"
    [ -z "$MISSING" ] || fail "still can't import: $(echo $MISSING)" \
        "run python3 -c 'import <name>' inside $VENV to see the real error."
fi
ok "python3 = ${PY_DESC:-$(command -v python3)} ($(python3 -c 'import sys; print(sys.version.split()[0])')), all dependencies importable"

# --- 2. Preflight checks --------------------------------------------------------

step "Preflight checks"

for tool in nvidia-smi ncu nsys; do
    command -v "$tool" &>/dev/null || fail "$tool not found on PATH" \
        "install NVIDIA Nsight Compute/Systems (they ship with the CUDA toolkit or HPC SDK) and put them on PATH."
done
ok "nvidia-smi, ncu, nsys on PATH"

python3 -c 'import cupy; assert int(cupy.arange(4).sum()) == 6' &>/dev/null || fail \
    "CuPy can't run a kernel on the GPU" \
    "run: python3 -c 'import cupy; print(cupy.arange(4).sum())' to see why (no GPU visible? cupy-cuda${CUDA_MAJOR}x vs driver mismatch?)."
GPU_INDEX="${CUDA_VISIBLE_DEVICES:-0}"; GPU_INDEX="${GPU_INDEX%%,*}"
[[ "$GPU_INDEX" =~ ^[0-9]+$ ]] || GPU_INDEX=0
IFS=',' read -r GPU_NAME COMPUTE_CAP GPU_UTIL GPU_MEM < <(nvidia-smi -i "$GPU_INDEX" \
    --query-gpu=name,compute_cap,utilization.gpu,memory.used --format=csv,noheader,nounits | sed 's/, */,/g')
export CUDA_ARCH="${CUDA_ARCH:-sm_${COMPUTE_CAP//./}}"
ok "GPU $GPU_INDEX: $GPU_NAME (compute capability $COMPUTE_CAP -> CUDA_ARCH=$CUDA_ARCH), CuPy works"

# Login nodes share their GPU and hold its counters, so profiling there is impossible or meaningless.
if [ "${NERSC_HOST:-}" = perlmutter ] && [[ "$(hostname)" == login* ]]; then
    fail_unless_forced "this is a Perlmutter login node ($(hostname))" \
        "get a compute node first: salloc -N 1 -C gpu -G 1 -q interactive -t 01:30:00 -A <project>, then re-run."
elif command -v sbatch &>/dev/null && [ -z "${SLURM_JOB_ID:-}" ]; then
    fail_unless_forced "Slurm is installed but this shell isn't inside a job (likely a login node)" \
        "allocate a dedicated GPU node with salloc and re-run inside it."
fi

N_PROCS="$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader | grep -c .)"
if [ "${GPU_UTIL:-0}" -gt 10 ] || [ "$N_PROCS" -gt 0 ]; then
    fail_unless_forced "GPU $GPU_INDEX is busy (${GPU_UTIL}% utilization, ${N_PROCS} process(es), ${GPU_MEM} MiB used)" \
        "timing and counter numbers on a shared GPU are off by up to 10x. Stop whatever is using it (see nvidia-smi) or use a dedicated node."
else
    ok "GPU $GPU_INDEX is idle (${GPU_UTIL}% utilization, no processes)"
fi

FREE_GB=$(( $(df -Pk "$REPO" | awk 'NR==2 {print $4}') / 1024 / 1024 ))
[ "$FREE_GB" -ge "$MIN_FREE_GB" ] || fail "only ${FREE_GB} GB free under $REPO" \
    "free at least ${MIN_FREE_GB} GB (old stage*/profiles/ directories can go), or clone onto a scratch filesystem."
if [ "$FREE_GB" -ge 1024 ]; then ok "$(( FREE_GB / 1024 )) TB free on the repo's filesystem"; else ok "${FREE_GB} GB free on the repo's filesystem"; fi

if [ -n "${SLURM_JOB_ID:-}" ] && command -v squeue &>/dev/null; then
    LEFT="$(squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null)"  # [D-]HH:MM:SS, MM:SS, or UNLIMITED
    MINUTES="$(echo "$LEFT" | awk -F'[-:]' '/UNLIMITED/ {print 99999; exit}
        NF==4 {print $1*1440+$2*60+$3} NF==3 {print $1*60+$2} NF==2 {print $1}')"
    if [ -n "$MINUTES" ] && [ "$MINUTES" -lt "$MIN_SLURM_MINUTES" ]; then
        warn "only $LEFT left in Slurm job $SLURM_JOB_ID; the full run takes 45-75 min on an A100, so later stages may be cut off"
    elif [ -n "$MINUTES" ]; then
        ok "$LEFT left in Slurm job $SLURM_JOB_ID"
    fi
fi

# Prove ncu can read the performance counters with a one-kernel program,
# pausing DCGM exactly the way every profile.sh does.
PROBE="$(mktemp -d)"
cat > "$PROBE/probe.cu" <<'EOF'
__global__ void bump(float *x) { x[threadIdx.x] += 1.0f; }
int main() { float *x; cudaMalloc(&x, 128 * sizeof(float)); bump<<<1, 128>>>(x); return cudaDeviceSynchronize(); }
EOF
nvcc -arch="$CUDA_ARCH" -o "$PROBE/probe" "$PROBE/probe.cu" &> "$PROBE/nvcc.log" || {
    cat "$PROBE/nvcc.log"; fail "nvcc couldn't build a trivial kernel for $CUDA_ARCH" \
        "set CUDA_ARCH to your GPU's sm_XX (nvidia-smi --query-gpu=compute_cap --format=csv) or use a newer CUDA toolkit."; }
if command -v dcgmi &>/dev/null; then dcgmi profile --pause &>/dev/null || true; fi
ncu --metrics sm__cycles_elapsed.avg "$PROBE/probe" &> "$PROBE/ncu.log"
if command -v dcgmi &>/dev/null; then dcgmi profile --resume &>/dev/null || true; fi
if grep -q "sm__cycles_elapsed.avg" "$PROBE/ncu.log" && ! grep -q "==ERROR==" "$PROBE/ncu.log"; then
    ok "ncu can read GPU performance counters"
    rm -rf "$PROBE"
elif grep -q "ERR_NVGPUCTRPERM" "$PROBE/ncu.log"; then
    fail "ncu: this user isn't allowed to read GPU performance counters (ERR_NVGPUCTRPERM)" \
        "an admin must set the nvidia driver option NVreg_RestrictProfilingToAdminUsers=0 (ENV_SETUP.md §4), or run as root."
elif grep -q "driver resource was unavailable" "$PROBE/ncu.log"; then
    fail "ncu: the GPU's counters are held by something else (DCGM, or another profiler/job)" \
        "make sure 'dcgmi profile --pause' works for you, and that no one else is using this GPU (ENV_SETUP.md §4)."
else
    sed 's/^/    /' "$PROBE/ncu.log" | tail -15
    fail "ncu couldn't profile a trivial kernel (output above, full log in $PROBE/ncu.log)" \
        "see README §11 Troubleshooting."
fi

if [ "$MODE" = check ]; then
    echo; echo "All checks passed. Run ./run_all.sh (without --check) to start the full analysis."
    exit 0
fi

# --- 3. Full analysis -----------------------------------------------------------

LOG="$REPO/roofline_analysis_$(date +%Y%m%d_%H%M%S).log"
step "Running the full roofline analysis (45-75 min on an A100); log: $LOG"
"$REPO/tools/run_roofline_analysis.sh" 2>&1 | tee "$LOG"

step "Summary"
STATUS=0
for s in stage1_naive stage2_gpu_native stage3_memory_opt stage4_compute_opt; do
    PROFILE="$(grep -E "^##### $s profile.sh exit" "$LOG" | awk '{print $5}')"
    # pytest's last line looks like "12 passed in 3.4s" or "1 failed, 11 passed in ...".
    TESTS="$(awk -v s="##### $s correctness tests" '$0 == s {on=1; next} /^#####/ {on=0}
        on && /(passed|failed|error)/ {line=$0} END {print line}' "$LOG" | sed 's/=//g; s/^ *//; s/ *$//')"
    if [ "$PROFILE" = 0 ] && [[ "$TESTS" == *passed* ]] && [[ "$TESTS" != *failed* ]] && [[ "$TESTS" != *error* ]]; then
        ok "$s: profile.sh exit 0; tests: $TESTS"
    else
        echo "  [FAIL] $s: profile.sh exit ${PROFILE:-?}; tests: ${TESTS:-no pytest summary found}"
        STATUS=1
    fi
done
OVERALL="$(grep "OVERALL" "$LOG" | tail -1)"
if [ -n "$OVERALL" ]; then ok "PNG vs Nsight Compute GUI: ${OVERALL#*: }"; else warn "no PNG-vs-GUI result in the log"; STATUS=1; fi

echo
echo "Plots:"
for f in "$REPO"/roofline_hierarchical_{vqe,gqe}.png "$REPO"/stage0_calibration/roofline_calibration.png; do
    [ -f "$f" ] && echo "  ${f#$REPO/}"
done
echo "  roofline_story_{vqe,gqe}_<n>.png         stage-by-stage slides (repo root)"
echo "  stage*/profiles/roofline_plots/          per-report plots and per-kernel tables"
echo "  stage*/profiles/*.ncu-rep, *.nsys-rep    raw reports for the Nsight Compute / Systems GUIs"
echo "Compare with the A100 reference in reference_results/ (README §10). Full log: ${LOG#$REPO/}"
[ "$STATUS" = 0 ] && echo "All stages passed." || echo "Some steps FAILED; search the log for the stage marked FAIL above."
exit "$STATUS"
