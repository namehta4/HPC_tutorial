# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A hands-on GPU performance tutorial that teaches roofline analysis (memory
bandwidth vs. compute FLOP/s ceilings) by re-implementing the *same* two
quantum-computing workloads four times, each stage fixing the bottleneck the
previous stage's profile revealed:

- **VQE** (Variational Quantum Eigensolver): a statevector simulator that
  applies a hardware-efficient ansatz circuit to `|0...0>` and evaluates
  `sum_i coeff_i * <psi|P_i|psi>` over a molecular Hamiltonian's Pauli terms.
- **GQE** (Generative Quantum Eigensolver): a small causal transformer policy
  network's forward pass (the part of GQE that's actually GPU-performance
  relevant — sampling candidate gate sequences).

Stage progression (each stage's `NOTES.md` has the full diagnosis):

| Stage | Bottleneck fixed | Technique |
|---|---|---|
| `stage0_calibration` | n/a | measures *this GPU's* real achievable BW/FLOPs (not vendor spec sheets) |
| `stage1_naive` | — | legitimate naive CuPy port: one kernel + host sync per gate/op |
| `stage2_gpu_native` | kernel-launch/sync overhead | fused gates, batched Hamiltonian terms, no syncs until the end, CUDA streams |
| `stage3_memory_opt` | global memory traffic | hand-written CUDA kernels, whole-circuit-in-shared-memory, FP32/FP16 mixed precision |
| `stage4_compute_opt` | occupancy / compute throughput | grid-batched statevectors (one block per circuit) for VQE; WMMA tensor-core GEMM + multi-stream for GQE |

The overall arc plotted by `plot_roofline.py`: overhead-bound → memory-bound
→ compute-bound, with a concrete code change motivating each transition.
Every stage must reproduce the same physics/weights as the CPU reference —
speed without correctness doesn't count as progress in this tutorial.

The top-level `README.md` is the participant-facing entry point: environment
setup for Perlmutter and generic Linux+GPU machines, the quick sanity check,
the shared-GPU-contention warning, per-stage run commands, a computational-
motifs explainer (§7), a generalization section on applying the tensor-core/
mixed-precision lessons to other codebases (§8), a 45-minute live-session
cheat sheet (§9), an expected-results table (§10), and troubleshooting
(§11). `ENV_SETUP.md` has the same setup content in more depth/rationale;
keep the two in sync if you edit either. If you renumber README's `##`
sections, grep it for `§[0-9]` first — cross-references are by section
number and easy to leave stale.

## Environment setup

Full details in `ENV_SETUP.md`. On NERSC Perlmutter:

```bash
module load pytorch/2.13.0
pip install --user cupy-cuda13x cuquantum-python-cu13 qutip pytest   # match cuXX to `nvcc --version`
```

Elsewhere: `python3 -m venv venv && pip install -r requirements.txt` (pick
the `cupy-cuda12x`/`cuquantum-python-cu12` line for CUDA 12.x).

**Get a dedicated GPU before trusting any timing/profiling number** — a
shared/login-node GPU under contention gives numbers off by 10x:

```bash
salloc -N 1 -C gpu -G 1 -q interactive -t 00:30:00 -A <your_NERSC_project>
```

## Commands

Run everything from the repo root or from inside a stage directory; each
stage's scripts insert the repo root onto `sys.path` themselves.

```bash
# Correctness tests, per stage (stage2-4 require ./build.sh first — see below)
cd stage1_naive && python3 -m pytest test_correctness.py -v
cd stage2_gpu_native && python3 -m pytest test_correctness.py -v
cd stage3_memory_opt && ./build.sh && python3 -m pytest test_correctness.py -v
cd stage4_compute_opt && ./build.sh && python3 -m pytest test_correctness.py -v

# Run a single test
python3 -m pytest test_correctness.py -v -k "test_vqe_energy_matches_cpu_reference and h2"

# Compile stage3/4's hand-written CUDA kernels (.cu -> .cubin, loaded via cupy.RawModule)
CUDA_ARCH=sm_80 ./build.sh   # sm_80 = A100; check yours with nvidia-smi --query-gpu=compute_cap --format=csv

# Run a stage's workload directly
python3 vqe_naive.py --molecule h2 --depth 3            # or lih, or random --qubits N (4-12)
python3 gqe_naive.py --batch-size 8

# Stage 0: measure THIS GPU's real bandwidth/FLOPs ceilings (run on a dedicated allocation)
cd stage0_calibration && ./run_calibration.sh   # writes bandwidth_fp64.json, gemm_peak.json
# then hand-copy the numbers into stage0_calibration/RESULTS.md

# Profile a stage (Nsight Systems timeline + Nsight Compute roofline; handles the
# DCGM-vs-ncu permission conflict automatically via dcgmi pause/resume)
cd stage1_naive && ./profile.sh   # writes profiles/*.nsys-rep, profiles/*.ncu-rep

# Export an ncu report to CSV for plot_roofline.py
ncu --import profiles/stage1_vqe_roofline.ncu-rep --csv --page raw > profiles/stage1_vqe_roofline.csv

# Overlay all four stages on one roofline chart (peak values are TFLOP/s, from stage0's RESULTS.md)
python3 plot_roofline.py --peak-bw 1500 --peak-flops-fp64 9.7 \
    --peak-flops-fp32 156 --peak-flops-fp16 312 \
    --csv stage1_naive/profiles/stage1_vqe_roofline.csv \
    --csv stage2_gpu_native/profiles/stage2_vqe_fused_roofline.csv \
    --csv stage3_memory_opt/profiles/stage3_vqe_fp64_roofline.csv \
    --csv stage4_compute_opt/profiles/stage4_vqe_roofline.csv \
    --out roofline_overlay.png
```

If `ncu` fails with `Profiling failed because a driver resource was
unavailable`, DCGM (or another counter-holding tool) is likely still
running — see ENV_SETUP.md §4, or check `nvidia-smi` for GPU contention.

## Architecture

**`common/`** is the single source of truth every stage builds on and must
stay bit-compatible with:
- `ansatz.py` — defines the one HEA circuit (RY, RZ per qubit per layer, then
  a CNOT ladder) used identically by all four stages. `build_circuit()`
  layout (all RY, then all RZ per layer) is what stage2+ exploit to fuse
  RZ(phi)@RY(theta) into one matrix per qubit before ever touching the
  statevector.
- `hamiltonians.py` / `_hamiltonian_data.py` — H2 (4 qubits, 15 terms) and
  LiH (8 qubits, 105 terms) as hardcoded, pre-verified Pauli-term lists
  (generated once by `tools/generate_hamiltonians.py` via PySCF/OpenFermion,
  which is NOT a runtime dependency). `random_pauli_hamiltonian()` covers
  the tutorial's 4-12 qubit sweep for sizes the two real molecules don't hit.
- `reference_cpu.py` — the **correctness oracle** for VQE: dense NumPy
  statevector sim, deliberately naive (O(4^n) memory per Pauli term via
  Kronecker products), fine up to ~16 qubits. Every GPU stage's energy must
  match this within `ENERGY_TOL` (usually `1e-6`).
- `gqe_policy_ref.py` — the CPU/PyTorch correctness oracle for GQE: a small
  causal transformer (`GQEPolicy`). `export_weights_numpy()` flattens its
  `state_dict()` so hand-rolled CuPy/CUDA stages can load identical weights
  without any PyTorch/autograd dependency at runtime.
- `config.py` — the shared CLI (`add_vqe_args`, `add_gqe_args`,
  `resolve_qubits`, `get_terms`) so `python vqe_naive.py --qubits 8` and
  `python vqe_compute_opt.py --qubits 8` run the identical logical problem.

**Bit/qubit convention** (load-bearing, appears everywhere): qubit 0 is the
**most significant bit** of the statevector index (big-endian), used
consistently in `common/`, all four stages' hand-rolled kernels, and the
Pauli mask math. The one deliberate exception is `stage2_gpu_native/
vqe_custatevec.py`, which calls into NVIDIA's cuStateVec library — that
library is little-endian internally, so every qubit index passed to a
`cusv.*` call goes through `to_custatevec_bit(n_qubits, q) = n_qubits-1-q`
first.

**Batched Pauli-expectation algebra** (used in stage2's
`batched_pauli_expectation`, and re-implemented as CUDA in stage3/4's
`expval_reduce*.cu`): applying Pauli string P to computational basis ket
`|k>` is an XOR bit-flip (X/Y positions) composed with a sign/phase that
depends only on the *pre-flip* bit values at the Z/Y positions —
`P|k> = (-1)^popcount(k & z_mask) * i^(#Y) * |k XOR flip_mask>`. This makes
every term's expectation value an O(2^n) vectorizable elementwise op instead
of stage1's O(4^n) dense matvec, and lets all `n_terms` terms be computed in
one batched pass. If you touch this code, verify against `common/
reference_cpu.py`'s dense matvec, not by inspection — the phase bookkeeping
(z_mask includes both Z *and* Y positions; the i^(#Y) factor is a per-term
constant, not a function of `k`) is easy to get subtly wrong.

**Tracked kernel names**, consistent across every stage's `NOTES.md`,
`profile.sh`, and `plot_roofline.py`, so cross-stage roofline comparison is
apples-to-apples:
- `gate_apply` — circuit application (one kernel/gate in stage1 → one
  whole-circuit-in-shared-memory kernel in stage3 → one grid-batched kernel
  in stage4).
- `expval_reduce` — Hamiltonian expectation-value reduction (one dense
  matvec/term in stage1 → one batched CuPy op in stage2 → one shared-memory
  kernel reused across all terms in stage3/4).
- `policy_forward` — GQE transformer forward pass (per-sample Python loop in
  stage1 → batched GEMMs in stage2 → fp16 in stage3 → WMMA tensor-core GEMM
  in stage4).

**Stage 3/4 CUDA kernels** (`kernels/*.cu`) are compiled by each stage's
`build.sh` into `.cubin` files (`nvcc -arch=sm_XX -cubin`, default
`sm_80`/A100 — override with `CUDA_ARCH`) and loaded at runtime via
`cupy.RawModule` + `get_function(...)`; there is no separate Python
extension build step. If you edit a `.cu` file, re-run `./build.sh` before
testing — nothing does this automatically.

**Precision handling**: `--precision {fp64,fp32,tf32,fp16}` is accepted
everywhere via `common/config.py` but not every stage implements every
value — stage3's statevector kernels only ship fp64/fp32 variants (fp16
statevector storage is numerically unusable for this tutorial's energy
tolerances) and fall back with a printed note; `tf32` for the statevector
kernels also falls back to fp32 since there's no tensor-core matmul in that
code path to apply TF32 to. TF32/fp16 tensor-core paths only exist in
stage4's GEMM-shaped GQE kernels. Each precision level has its own,
progressively looser correctness tolerance in that stage's
`test_correctness.py` (documented per-stage in `NOTES.md`'s Correctness
section) — don't tighten these without re-deriving them, they reflect real
measured error, not arbitrary margins.

**Stage4 does NOT use tensor cores for VQE gates** — this is intentional,
not a missing optimization. A single-qubit gate is a 2x2 unitary; WMMA tiles
are >=16x16x16, so forcing a 2x2 gate through one wastes >99% of the tile.
`kernels/gate_apply_tc.cu`'s header explains this; stage4's VQE speedup
instead comes from occupancy (one CUDA block per statevector in a batch,
`grid.x = batch_size`, so B independent evaluations span up to B SMs instead
of running single-SM kernels back-to-back). Tensor cores (WMMA, in
`kernels/policy_forward_gemm.cu`) are reserved for GQE's genuinely
GEMM-shaped linear layers.

**Roofline plotting** (`plot_roofline.py`) parses `ncu --csv --page raw`
exports looking for a short list of known column-name aliases (ncu's schema
has shifted across versions); pass `--manual-points <json>` instead when a
report's columns don't match — this is also currently the only way to plot
FP16/tensor-core kernels, since the script's FLOP-column heuristics only
cover FP64/FP32 FMA counters. `--peak-flops-fp64/-fp32/-fp16` take **TFLOP/s**
(from `stage0_calibration/RESULTS.md`), not GFLOP/s, despite what the
`--help` text for those flags currently says — trust the module docstring's
example and the runtime error message over `--help` here.
