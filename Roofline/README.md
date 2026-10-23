# GPU Roofline Tutorial: Quantum Chemistry Edition

A hands-on tutorial that teaches **roofline performance analysis** on NVIDIA
GPUs — the standard way to reason about whether a kernel is limited by
memory bandwidth, compute throughput, or neither — by optimizing the same
two quantum-computing workloads four times over:

- **VQE** (Variational Quantum Eigensolver) — a statevector simulator that
  evaluates a molecule's energy from a parameterized quantum circuit.
- **GQE** (Generative Quantum Eigensolver) — a small transformer's forward
  pass, used to sample candidate quantum circuits.

You don't need any GPU performance engineering background to follow this.
You do need to already be comfortable with Python and, ideally, have some
quantum chemistry / quantum computing context for *why* VQE and GQE look the
way they do — the tutorial doesn't re-teach that part.

**The story, in one sentence per stage:**

| Stage | Directory | Bottleneck it fixes | The fix |
|---|---|---|---|
| 0 | `stage0_calibration/` | n/a | measure *this GPU's* real bandwidth/FLOP-s ceilings |
| 1 | `stage1_naive/` | — (baseline) | a legitimate, unoptimized CuPy port |
| 2 | `stage2_gpu_native/` | kernel-launch & sync overhead | fuse gates, batch work, remove syncs, use streams |
| 3 | `stage3_memory_opt/` | global memory traffic | hand-written CUDA kernels, shared memory, mixed precision |
| 4 | `stage4_compute_opt/` | occupancy / compute throughput | batch across SMs, tensor cores for the GEMM-shaped GQE |

Each stage's own `NOTES.md` explains, in depth, what's wrong with that
stage's code, what you should see in the profiler, and what change gets you
to the next stage. This README is about **getting the tutorial running at
all** — environment setup, installation, and the commands to execute each
stage in order.

If you're new to HPC/SLURM/CUDA, read this whole page top to bottom before
running anything; it's written so you don't need outside context.

---

## Quick start

### Step 1 — Set up (once per machine; the module/env step every session)

**On NERSC Perlmutter** (details: §2):

```bash
git clone https://github.com/namehta4/HPC_tutorial.git && cd HPC_tutorial/Roofline

# Python 3.12 + NumPy/SciPy/PyTorch. Needed in every new shell, including
# on the compute node after salloc -- modules don't carry over.
module load pytorch/2.13.0

# One-time: the remaining dependencies, into your user site-packages.
# Use cupy-cuda12x / cuquantum-python-cu12 instead if `nvcc --version` says CUDA 12.x.
pip install --user cupy-cuda13x cuquantum-python-cu13 qutip pytest

# Get a dedicated GPU node (timings on shared login-node GPUs are meaningless, §5),
# then load the module again there.
salloc -N 1 -C gpu -G 1 -q interactive -t 01:30:00 -A <your_NERSC_project>
module load pytorch/2.13.0
```

**On any other Linux machine with an NVIDIA GPU** (details: §3): first
install the CUDA Toolkit and Nsight Systems/Compute (check with
`nvcc --version`, `nsys --version`, `ncu --version`), then:

```bash
git clone https://github.com/namehta4/HPC_tutorial.git && cd HPC_tutorial/Roofline

# Either a venv ...
python3 -m venv venv && source venv/bin/activate
# ... or a conda env:
#   conda create -n roofline python=3.12 -y && conda activate roofline

# requirements.txt targets CUDA 13.x; for CUDA 12.x, change its two cu13 lines
# to cupy-cuda12x / cuquantum-python-cu12 first.
pip install -r requirements.txt
```

Re-activate the venv/conda env in every new shell. Profiling needs access
to the GPU's performance counters. If `ncu` reports `ERR_NVGPUCTRPERM`, see
§11.

**Then, on either system, check that everything works:**

```bash
nvidia-smi --query-gpu=name,compute_cap,utilization.gpu --format=csv   # GPU idle? note compute_cap
python3 -c "import cupy as cp; print(cp.arange(5).sum())"               # CuPy sees the GPU -> 10
```

`compute_cap` gives the `CUDA_ARCH` for building stage3/4's CUDA kernels:
8.0 → `sm_80` (A100, the default), 8.6 → `sm_86`, 8.9 → `sm_89`,
9.0 → `sm_90` (H100).

### Step 2 — Run the per-stage loop

On your **dedicated, idle GPU**, the whole tutorial is the same loop
repeated per stage:

```bash
# Stage 0, once: measure this GPU's real bandwidth / FLOP-s ceilings
cd stage0_calibration && ./run_calibration.sh && cd ..

# Stages 1-4: stage1_naive, stage2_gpu_native, stage3_memory_opt, stage4_compute_opt
cd stage1_naive
./build.sh                                                  # stage3/4 only; CUDA_ARCH=sm_XX ./build.sh if not an A100
python3 -m pytest test_correctness.py -v                    # correctness first
python3 vqe_naive.py --molecule random --qubits 12 --depth 3   # run it directly (script names per stage: §6)
QUBITS=12 D_MODEL=128 BATCH=256 GQE_BATCH=256 ./profile.sh  # nsys timeline + ncu reports + roofline plots
cd ..
```

### Step 3 — Look at the results

After each stage's `profile.sh`, look at:

- `profiles/*_timeline.nsys-rep` (Nsight Systems): launch gaps and host
  syncs, i.e. the overhead story.
- `profiles/roofline_plots/<report>_all.png`: each kernel at L1 (o), L2 (s)
  and HBM (v). The gap between a kernel's three points shows which memory
  level serves it. `<report>_overview.png` is the whole workload as one
  point per level. The table printed at the end of `profile.sh` (also
  `<report>_kernels.csv`) has the numbers.
- `profiles/*.ncu-rep` in the Nsight Compute GUI (optional): its
  hierarchical roofline charts show the same numbers as the PNGs.
- That stage's `NOTES.md`: the diagnosis, and the fix the next stage applies.

The repo-root `roofline_hierarchical_vqe.png` / `roofline_hierarchical_gqe.png`
are refreshed after every stage with one point triplet per completed stage,
so the overhead-bound → memory-bound → compute-bound progression builds up
as you go. `roofline_story_{vqe,gqe}_<n>.png` show the same progression as
a slide sequence (HBM points only, with speedup arrows). The reference
versions from an A100 on Perlmutter are in `reference_results/` (files
tagged `_a100_pm`).

Two things to know:
- **Each `profile.sh` takes several minutes**, because ncu replays every
  kernel many times. In the 45-minute live format (§9), run the scripts
  directly and look at pre-generated plots; run `profile.sh` afterwards.
- **Use the `QUBITS=12 ...` sizes.** At the default H2 size, every stage's
  points land in the same corner of the plot (see the note at the top of §6).

Instructors can regenerate everything (calibration, all four stages, tests,
plots, and a check that the PNGs match the ncu GUI) in one go with
`tools/run_roofline_analysis.sh`.

---

## 1. What you need before you start

- **An NVIDIA GPU** you can get exclusive/dedicated access to (shared,
  contended GPUs give you garbage timing numbers — more on this in §5).
- **The CUDA Toolkit** (`nvcc`) and **Nsight Systems** (`nsys`) / **Nsight
  Compute** (`ncu`), the two profilers this tutorial's `profile.sh` scripts
  drive. These normally ship together with the CUDA Toolkit.
- **Python 3.10+** (3.12 is what this was built/tested with).

Check what you already have:

```bash
nvidia-smi                        # GPU present? Which model? How busy is it?
nvcc --version                    # CUDA compiler available?
nsys --version && ncu --version   # Profilers available?
```

If any of these say "command not found," see §2 (Perlmutter) or §3
(anywhere else) below before continuing.

---

## 2. Setup on NERSC Perlmutter (exact, copy-paste steps)

This tutorial was built and tested here: NVIDIA A100-PCIE-40GB, compute
capability 8.0, CUDA 13.2. If you're on Perlmutter, use these exact
commands.

```bash
# Perlmutter's bare login-node python3 has no working `pip --user`.
# This module gives you a real Python 3.12 + NumPy/SciPy/PyTorch stack.
module load pytorch/2.13.0

# Install this tutorial's remaining dependencies (cupy, cuquantum, qutip, pytest)
# into your user site-packages. cupy-cuda13x / cuquantum-python-cu13 match
# Perlmutter's CUDA 13.x toolkit -- if `nvcc --version` shows CUDA 12.x on
# your allocation instead, use cupy-cuda12x / cuquantum-python-cu12.
pip install --user cupy-cuda13x cuquantum-python-cu13 qutip pytest

# Sanity check: is the GPU visible to CuPy?
python3 -c "import cupy as cp; print(cp.cuda.Device(0).compute_capability); print(cp.arange(5).sum())"
```

**Get a dedicated GPU** for anything you plan to time or profile — a login
node's GPU is often shared with other users' jobs, and every number you
measure on a contended GPU will be meaningless (see §5). Request one with:

```bash
salloc -N 1 -C gpu -G 1 -q interactive -t 00:30:00 -A <your_NERSC_project>
```

(`sacctmgr show associations user=$USER format=account` lists which
`<your_NERSC_project>` values you can use.) Once your allocation starts,
you're on a fresh compute node — **re-run `module load pytorch/2.13.0`
there too**; modules loaded on the login node don't carry over.

Skip to §4 (Quick Sanity Check) once this is done.

---

## 3. Setup anywhere else (any Linux machine with an NVIDIA GPU)

1. **Install the CUDA Toolkit** from
   [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads),
   matching the CUDA version your installed driver supports (`nvidia-smi`'s
   header shows the max CUDA version your driver supports).
2. **Install Nsight Systems and Nsight Compute** — usually bundled with the
   CUDA Toolkit installer, or downloadable separately from
   [developer.nvidia.com/nsight-systems](https://developer.nvidia.com/nsight-systems)
   and [developer.nvidia.com/nsight-compute](https://developer.nvidia.com/nsight-compute).
   Confirm with `nsys --version` and `ncu --version`.
3. **Create a Python environment and install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
   `requirements.txt` defaults to CUDA 13.x packages (`cupy-cuda13x`,
   `cuquantum-python-cu13`) — edit those two lines to `cupy-cuda12x` /
   `cuquantum-python-cu12` if `nvcc --version` shows CUDA 12.x instead.
4. **Find your GPU's compute capability** — you'll need this to compile
   stage3/stage4's hand-written CUDA kernels:
   ```bash
   nvidia-smi --query-gpu=compute_cap --format=csv
   ```
   Common values: `8.0` (A100), `8.6` (RTX 30-series / A40), `8.9` (RTX
   40-series / L4 / L40), `9.0` (H100). You'll pass this as
   `CUDA_ARCH=sm_XX` when building stage3/4 (see §6).

Full details and rationale for every step above live in `ENV_SETUP.md`, if
you want the long version.

---

## 4. Quick sanity check

Before doing anything else, confirm the whole pipeline works end to end
(this only needs stage1, which has no CUDA kernels to compile):

```bash
cd stage1_naive
python3 -m pytest test_correctness.py -v
```

**You should see 10 tests pass.** If they do, your environment is correctly
set up and you're ready to start the tutorial. If they don't:

- `ModuleNotFoundError: cupy` → your `pip install` didn't target the right
  CUDA version, or didn't finish. Re-check §2/§3.
- `cupy.cuda.runtime.CUDARuntimeError` / no GPU visible → you're likely on a
  login node with no GPU attached, or your `salloc`/allocation hasn't
  started yet. Run `nvidia-smi` to confirm a GPU is visible first.
- Anything else → see the Troubleshooting section (§11) below.

---

## 5. Before you trust ANY timing number: the shared-GPU warning

This tutorial's whole point is teaching you to compare measured performance
against *achievable* hardware ceilings. If another job is running on the
same physical GPU while you measure, your numbers reflect **contention**,
not your code's efficiency — and roofline conclusions drawn from contended
numbers are simply wrong.

We hit this ourselves while building this tutorial: on a login-node GPU
already at 100% utilization from an unrelated job, we measured **~192 GB/s**
of memory bandwidth against that same A100's real achievable ceiling of
roughly **1300–1500 GB/s** — 10x off, purely from sharing the GPU.

Before trusting any timing, profiling, or roofline number:

1. Run `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv` and
   confirm the GPU is near-idle before you start.
2. Use a dedicated allocation (`salloc` on Perlmutter, or any machine nobody
   else is actively using) — not a shared login node.
3. Run stage0's calibration (next section) on that dedicated allocation
   before trusting any later stage's roofline plot.

---

## 6. Running the tutorial, stage by stage

Work through these in order — each stage's `NOTES.md` assumes you've
already read the previous one's.

**Problem size.** The direct-run commands below use `--molecule h2`
(4 qubits): fast, and right for a quick correctness check. For profiling,
use the larger sizes via `profile.sh`'s environment variables
(`QUBITS=12 D_MODEL=128 BATCH=256 GQE_BATCH=256`). At H2 scale every
stage's kernels move only a few KB, so all stages land in the same corner
of the roofline chart. At 12 qubits (a 64 KB statevector, the tutorial's
cap) the overhead → memory → compute progression is clearly visible. The
12-qubit Hamiltonian is a synthetic, molecule-like Pauli sum
(`common/hamiltonians.py: random_pauli_hamiltonian`), not a real molecule.

### Stage 0 — Calibrate this GPU's real ceilings

Do this first, on a dedicated/idle GPU (see §5):

```bash
cd stage0_calibration
./run_calibration.sh
```

This measures achieved memory bandwidth (a fused STREAM Triad kernel) and
achieved cuBLAS GEMM FLOP/s (FP64, TF32, FP16) on *your actual GPU*, and
writes `bandwidth_fp64.json` / `gemm_peak.json`. Copy the numbers into
`stage0_calibration/RESULTS.md`. The roofline plots draw ncu's theoretical
ceilings, plus this measured HBM bandwidth as a dashed line, so you can
see how much of the theoretical peak is actually achievable.

If your FP32/TF32 number comes back close to your FP64 number instead of
~8x higher, cuBLAS isn't using TF32 tensor cores; `gemm_peak_probe.py`
requests them explicitly, so check you're on the current version.

### Stage 1 — Naive baseline

```bash
cd stage1_naive
python3 -m pytest test_correctness.py -v                            # correctness first
python3 vqe_naive.py --molecule h2 --depth 3                        # run it directly
python3 gqe_naive.py --batch-size 8
QUBITS=12 D_MODEL=128 BATCH=256 ./profile.sh                        # nsys + ncu + roofline plots -> profiles/
```

Read `stage1_naive/NOTES.md` afterward. It explains what the Nsight
Systems timeline and the roofline plots should look like for this stage,
and why.

### Stage 2 — GPU-native restructuring

```bash
cd stage2_gpu_native
python3 -m pytest test_correctness.py -v
python3 vqe_fused.py --molecule h2 --depth 3
python3 vqe_custatevec.py --molecule h2 --depth 3   # library comparison point
python3 gqe_batched.py --batch-size 32
QUBITS=12 D_MODEL=128 BATCH=256 ./profile.sh
```

### Stage 3 — Memory-traffic reduction (compile CUDA kernels first)

Stage 3 introduces hand-written CUDA kernels, which need to be compiled to
`.cubin` files before anything else in this stage will run:

```bash
cd stage3_memory_opt
CUDA_ARCH=sm_80 ./build.sh    # match sm_80 to YOUR GPU -- see §3 step 4
python3 -m pytest test_correctness.py -v
python3 vqe_memopt.py --molecule h2 --precision fp32
QUBITS=12 D_MODEL=128 BATCH=256 ./profile.sh
```

If you edit any `kernels/*.cu` file, re-run `./build.sh`; nothing does
this for you automatically.

### Stage 4 — Compute-bound tuning

```bash
cd stage4_compute_opt
CUDA_ARCH=sm_80 ./build.sh    # same note as stage3
python3 -m pytest test_correctness.py -v
python3 vqe_compute_opt.py --molecule h2
python3 gqe_multistream.py --batch-size 32
QUBITS=12 D_MODEL=128 GQE_BATCH=256 ./profile.sh
```

### The roofline plots

Every stage's `profile.sh` ends by running `plot_hierarchical_roofline.py`
on the reports it just wrote:

- `profiles/roofline_plots/<report>_all.png`: each kernel (top 8 by GPU
  time) plotted at L1 (o), L2 (s) and HBM (v). All three share the same
  GFLOP/s; the horizontal gap between them shows which memory level serves
  the kernel's traffic.
- `profiles/roofline_plots/<report>_{hbm,l2,l1}.png`: one memory level per plot.
- `profiles/roofline_plots/<report>_overview.png`: the whole workload as
  one point per level.
- `profiles/roofline_plots/<report>_kernels.csv`: the numbers behind the plots.
- `roofline_hierarchical_vqe.png` / `roofline_hierarchical_gqe.png` at the
  repo root: one point triplet per stage profiled so far, i.e. the
  tutorial's stage-by-stage progression.
- `roofline_story_vqe_<n>.png` / `roofline_story_gqe_<n>.png` at the repo
  root: a slide sequence for presenting the progression. Slide 1 shows
  stage 1 and each later slide adds the next stage, with all ceilings
  drawn but only HBM points, identical axes, and arrows labeled with the
  speedup.

Reference copies of all of these from a dedicated A100 run are in
`reference_results/` (§10).

The points and ceilings use the same `--set roofline` metrics and formulas
as the Nsight Compute GUI's "Hierarchical Roofline" charts, so they match
what you'd see in the GUI. Ceilings are ncu's theoretical peaks at the
clocks ncu locks during profiling (~1.1 GHz on A100, below boost). There
are two deliberate differences from the GUI. First, a kernel's double,
single and tensor-core FLOPs are summed into one point, and repeated
launches of the same kernel are combined (total FLOPs / total time); the
GUI draws one chart per precision and one point per launch. Second, the
GUI's CUDA-core half-precision chart is left out: in this tutorial it only
counts compiler register moves (see the `plot_hierarchical_roofline.py`
docstring), not arithmetic.

To re-plot existing reports by hand (works without a GPU):

```bash
python3 plot_hierarchical_roofline.py stage1_naive/profiles/stage1_vqe_roofline.ncu-rep
python3 plot_hierarchical_roofline.py --compare vqe --compare gqe
```

---

## 7. Computational motifs: what's actually changing, stage to stage

The four stages can look like four unrelated bags of tricks (fuse gates!
use shared memory! try fp16! use tensor cores!) if you only skim each
stage's `NOTES.md` in isolation. They aren't. Both VQE and GQE each reduce
to exactly **two** recurring computational motifs, and every stage
transition is the *same* underlying move — first collapse launch overhead,
then move data on-chip, then re-target the actual compute unit — applied to
those two motifs. This section makes that structure explicit.

### VQE's two motifs

1. **Statevector transform** — apply a small unitary (a 2×2 rotation, or a
   controlled-swap-like CNOT) to a length-`2^n` complex vector. Structurally
   this is a **stencil over an index space partitioned by one or two bits**,
   applied once per gate.
2. **Batched reduction** — for every Hamiltonian term, compute `<psi|P|psi>`,
   then sum with coefficients. Structurally, for each of `n_terms`: a
   **phase-and-permute pass over the same `2^n` vector, followed by a
   dot-product reduction**.

| Stage | Statevector-transform motif | Batched-reduction motif |
|---|---|---|
| 1 naive | One `tensordot`/`moveaxis` kernel **per gate** (33 separate launches for H2 depth 3), each a full global-memory round trip over its own tiny slice of index space | Dense `2^n x 2^n` matrix built via repeated `kron`, then a dense matvec — the motif is **materialized as an explicit matrix multiply**, `O(4^n)` work for what's structurally an `O(2^n)` operation, once per term, with a host sync every time |
| 2 GPU-native | RY+RZ pre-multiplied on the host into one 2x2 matrix, so two gates on the same qubit collapse into one kernel — same stencil, **coarsened per-qubit before ever touching the statevector** | Dense matvec replaced with a **closed-form algebraic identity**, `P\|k> = phase(k)*\|k XOR mask>`, vectorized over both the `2^n` index space *and* all `n_terms` simultaneously via broadcasting — motif changes from "matrix multiply" to "batched XOR + phase + reduce," `O(2^n * n_terms)` instead of `O(4^n * n_terms)` |
| 3 memory-opt | Same fused-matrix stencil, but the **entire statevector loads into shared memory once**; every gate updates it in place (`__syncthreads()` between gates); one store at the end — motif is unchanged, its **residence** moves from "global memory, revisited every gate" to "on-chip, visited once" | Same XOR/phase/reduce algebra, but the shared-memory statevector tile is **reused across all `n_terms` reductions in one kernel launch** — same data-locality shift as the transform motif |
| 4 compute-opt | Identical shared-memory stencil kernel, but **gridded**: one CUDA block per statevector, `grid.x = batch_size` — the motif itself doesn't change, only its **parallel decomposition** (many independent per-circuit blocks spanning many SMs, instead of one block looping) | Same batched shared-memory reduction, same grid-per-statevector batching |

Stage 3→4 does not change *what* is computed at all for VQE — profiling
confirms stage3 and stage4 both launch exactly 2 kernels per energy
evaluation (`apply_ansatz_*` and `expval_pauli_batched_*`); stage4's change
is purely how many SMs those 2 kernels occupy concurrently.

### GQE's two motifs

1. **Linear layer (GEMM)** — `x @ W.T` for the QKV projection, the output
   projection, and both MLP layers.
2. **Attention** — `softmax(QK^T / sqrt(d)) @ V`, itself a pair of
   GEMM-shaped contractions plus an elementwise/reduction softmax.

Unlike VQE's stage1, which represented a bitwise operation as a wasteful
dense matrix multiply, GQE's GEMMs are "real" GEMMs from the very first
stage — so GQE's progression is a more direct case study in "how do you
execute a GEMM well on a GPU," which is exactly why it's this tutorial's
worked example for the tensor-core transition in §8 below.

| Stage | GEMM motif | Attention motif |
|---|---|---|
| 1 naive | Per-sample Python loop — each of the batch's samples gets its own small `(seq_len, d_model)` matmul, launched separately. Motif is **B independent small unbatched GEMMs**; 1088 total kernel instances measured for batch=8, the most extreme "many tiny launches" case in the tutorial | Same per-sample loop; QK^T and attn·V computed per sample |
| 2 GPU-native | The whole `(B, T, D)` tensor multiplied at once via `@` — cuBLAS's native **batched/strided-GEMM** primitive. Motif becomes **one batched GEMM instead of B separate ones** | `cp.matmul` on 4D tensors `(B, H, T, T)` — same batched-GEMM primitive applied to attention's two contractions |
| 3 memory-opt | Same batched GEMM, now at **fp16** — half the bytes moved per weight/activation load. Motif is unchanged; **precision is reduced** (fp32 kept for softmax/layernorm accumulation — standard practice for low-precision transformers) | Same, fp16 |
| 4 compute-opt | Same GEMM shape, executed by a **hand-written WMMA tensor-core kernel** (`kernels/policy_forward_gemm.cu`) — each warp computes one 16x16 output tile via `wmma::mma_sync`, fp16-multiply/fp32-accumulate. Motif moves from "cuBLAS batched GEMM on CUDA cores" to "the same GEMM on tensor cores." Multiple independent sampling batches also now run on **separate CUDA streams concurrently** | Same tensor-core routing — attention's contractions go through the same GEMM kernel |

### The one-sentence summary, for both workloads

- **Stage 1 -> 2**: doesn't change the *math*, changes the **granularity of
  kernel launches** — many tiny ops become a few large batched/fused ops.
  (VQE's stage2 also genuinely changes the *algorithm* for the reduction
  motif — dense matvec becomes XOR/phase algebra — a bigger jump than GQE
  gets here, since GQE's GEMMs were never algorithmically wasteful the way
  stage1's dense Pauli matrices were.)
- **Stage 2 -> 3**: doesn't change the *op count or algorithm*, changes
  **where the data lives during computation** — global memory, revisited
  repeatedly, becomes on-chip (shared memory for VQE, lower-precision
  bytes for GQE) so fewer bytes cross the memory bus per unit of useful work.
- **Stage 3 -> 4**: doesn't change the *per-instance computation* at all,
  changes the **hardware unit and parallel decomposition** used to execute
  it — more SMs occupied concurrently for VQE (grid of blocks), tensor
  cores instead of CUDA cores plus concurrent streams for GQE.

This is also why the tutorial holds together as one story instead of four
unrelated tricks: each transition targets exactly the bottleneck the
*previous* stage's profile revealed (launch overhead -> memory traffic ->
compute-unit utilization), while the underlying computational motif —
stencil+reduction for VQE, GEMM+attention for GQE — stays recognizably the
same object throughout.

---

## 8. Applying these lessons to other codes

The specific kernels here are quantum-chemistry-flavored, but the
*diagnostic process* and the *motif transitions* in §7 are general. This
section is the "so what do I do with this on Monday, on my own codebase"
translation.

### The diagnostic process, restated generically

1. **Profile before optimizing, always in that order.** Every stage of
   this tutorial exists because a *specific* profiler finding motivated it
   — nobody guessed. `nsys stats --report cuda_gpu_kern_sum` answers "am I
   overhead-bound?" (hundreds of microsecond-scale kernel launches with
   gaps between them, low GPU-busy%). `ncu --set roofline` then answers
   "given that I'm not overhead-bound, am I memory- or compute-bound?" by
   placing your kernel's (arithmetic intensity, achieved GFLOP/s) point
   against ceilings **you measured on your own hardware** (stage0's whole
   reason for existing) — not vendor spec sheets, which routinely overstate
   achievable performance by 1.5-3x.
2. **Fix bottlenecks in the order the profile ranks them, not the order
   that sounds most impressive.** Tensor cores are stage4's optimization,
   not stage1's, because they only matter once you're compute-bound — using
   them earlier (on a launch-overhead-bound or memory-bound kernel) would
   have made no measurable difference and hidden the real problem. Chasing
   the "exciting" optimization before the ones the profile actually flags
   is the single most common way performance work stalls out.
3. **Re-profile after each change.** A memory-bound kernel that gets
   fused/batched may become compute-bound; a compute-bound kernel that gets
   tensor cores may become memory-bound again if data layout wasn't also
   fixed. The bottleneck moves — that's not a bug in your optimization
   process, it's the expected outcome of a real roofline improvement.

### Applying the VQE motif transitions: any small-op-count, launch-bound kernel sequence

VQE's arc (unfused stencil -> fused stencil -> shared-memory-resident
stencil -> gridded/batched stencil) generalizes to **any workload built
from many small, independent operations over the same working set** —
finite-difference stencils, N-body / particle-in-cell updates, small
per-node graph-neural-network message passes, small-matrix linear algebra
(batches of tiny GEMMs/GEMVs that are individually too small to saturate a
GPU), or any simulation with a "many small time-steps/many small local
updates" structure:

- If `nsys` shows many kernels a few microseconds long with visible gaps:
  look for independent, back-to-back host-callable operations on the same
  buffer and **fuse them into one kernel** (stage1->2's move), and remove
  intermediate `.get()`/synchronize() calls that force the host to wait
  between them.
- If the fused kernel still round-trips global memory once per step: check
  whether the whole per-step working set fits in shared memory / registers.
  If it does, **load once, iterate on-chip, store once** (stage2->3's
  move) — this is the single highest-leverage change for anything with a
  "small working set, many sequential steps over it" shape.
- If a single instance of the (now-efficient) kernel only occupies one SM
  because there's only one instance in flight: check whether your workload
  has *independent* problem instances that could run concurrently (batches
  of parameter sets, ensemble members, grid cells, particles) and **batch
  them across a grid** (`grid.x = batch_size`, stage3->4's move) rather
  than looping over them sequentially on the host.

### Applying the GQE motif transitions: general playbook for reaching tensor cores

This is the more broadly-applicable half, since it's exactly the path any
PyTorch/JAX/CUDA model takes toward tensor-core utilization, whether the
model is a quantum-circuit-sampling transformer or a production LLM:

1. **Batch your GEMMs.** If your code loops over samples/sequences calling
   a matmul once per item (stage1's pattern — the single most common
   GPU-performance bug in research code translated naively from a CPU/NumPy
   prototype), collapse the loop into one call with a leading batch
   dimension. `torch.bmm`/`@` on batched tensors, `cp.matmul` on 4D
   tensors, or `cublasGemmStridedBatchedEx` at the C++ level all do this;
   the win is identical in kind to stage1->2 here — many small GEMMs become
   one batched GEMM cuBLAS can actually saturate the GPU with.
2. **Check whether tensor cores are already being used before hand-writing
   anything.** cuBLAS/cuDNN/PyTorch/JAX all dispatch to tensor cores
   automatically for GEMMs in fp16, bf16, or tf32 *when the operand shapes
   and alignment qualify* (dimensions that are multiples of 8 or 16 are the
   usual rule of thumb, since a WMMA/MMA tile is >=16x16x16 — this
   tutorial's `kernels/gate_apply_tc.cu` header spells out exactly why
   forcing a too-small operation through a tile wastes the tile). In
   PyTorch: `torch.backends.cuda.matmul.allow_tf32 = True` and running in
   `torch.autocast("cuda", dtype=torch.float16)` gets you most of stage3+4's
   benefit here for free, without hand-writing WMMA the way
   `policy_forward_gemm.cu` does for teaching purposes. Hand-written WMMA
   (this tutorial's stage4 approach) is worth it when you have an unusual
   fused-op shape a stock GEMM call won't express, not as a first step.
   This tutorial has its own example: stage3's GQE stores everything in
   fp16, yet ncu shows zero tensor-core FLOPs, because cuBLAS picks a
   CUDA-core kernel for those skinny shapes (§10). Measure; don't assume.
3. **Only reduce precision where the tolerance budget allows it, and
   measure the actual error, don't assume it.** Stage3's fp16 GQE path
   keeps softmax/layernorm accumulation in fp32 specifically because
   naively casting *everything* to fp16 loses too much precision in
   reductions — this is the standard "mixed precision," not "low
   precision," pattern used in virtually all production tensor-core AI
   training/inference code (it's what `torch.autocast` and NVIDIA's AMP do
   under the hood). Always validate against a higher-precision reference
   (this tutorial's `test_correctness.py` pattern) rather than trusting
   that a precision drop "should be fine."
4. **Recognize when a shape does *not* suit tensor cores, and don't force
   it.** VQE's stage4 explicitly does NOT use tensor cores for exactly this
   reason (2x2 gate into a >=16x16x16 tile wastes over 99% of the tile) and
   instead solves its compute-bound problem with occupancy. The general
   lesson: **tensor cores are a GEMM-shaped-operation optimization, not a
   universal "make it faster" switch.** Attention layers, linear layers,
   and convolutions (which lower to GEMMs via im2col) suit them; small
   fixed-size unitary applications, sparse/irregular access patterns, and
   anything dominated by non-GEMM elementwise/reduction work generally do
   not — profile first (per §8's first point) rather than assuming.
5. **Multi-stream concurrency is a separate, complementary lever from
   tensor cores, not a substitute for them.** `gqe_multistream.py`'s
   pattern — several independent sampling batches on separate CUDA streams
   — generalizes to any workload with multiple independent problem
   instances that individually don't saturate the GPU (small-batch
   inference serving multiple requests, ensemble evaluation, hyperparameter
   sweeps): concurrent streams let the GPU scheduler overlap them instead
   of running them back-to-back, exactly like VQE's grid-batching but at
   the stream level instead of the block level.

**In one sentence:** whether the workload is quantum chemistry or a
production transformer, the path to tensor cores is *not* "add
`wmma::mma_sync` calls" — it's "batch your GEMMs, confirm the shapes and
precision actually qualify for tensor-core dispatch, verify the achieved
precision against a reference, and only hand-write a kernel when a stock
library call can't express your fused shape." Reaching for tensor cores or
mixed precision on a kernel that's still launch-overhead-bound or has the
wrong operand shape wastes the effort and, per §7, is exactly the mistake
this tutorial's stage ordering is designed to prevent.

---

## 9. 45-minute live session cheat sheet

This section is for running the tutorial **live, hands-on, in a fixed time
box** (e.g. a workshop session) — one copy-pastable block per time segment,
assuming everyone already has a dedicated GPU allocation (§2's `salloc`) and
already ran §2/§3's environment setup **before** the session starts. Skip
this section if you're working through the tutorial at your own pace — use
§6 instead.

**Why this cheat sheet skips the full `ncu --set roofline` capture in each
stage's `profile.sh`:** a single `ncu --set roofline` run takes 60–150
seconds (multi-pass hardware counter collection), and each stage's
`profile.sh` runs 2–4 of them. Doing that live for all four stages would
consume the entire 45 minutes on progress bars. Instead, this path uses
`nsys stats --report cuda_gpu_kern_sum` — seconds, not minutes — to make the
"how many kernel launches, how big" story visible directly, which is the
same underlying diagnosis the roofline plot makes visually. Run each stage's
full `./profile.sh` afterward, on your own time, to get the roofline charts.

Copy `<your_NERSC_project>` from `sacctmgr show associations user=$USER
format=account` once, before you start the clock.

### 0-5 min — Get on a GPU, confirm setup

```bash
salloc -N 1 -C gpu -G 1 -q interactive -t 00:45:00 -A <your_NERSC_project>
module load pytorch/2.13.0        # must re-run on the compute node, not just the login node
nvidia-smi --query-gpu=name,utilization.gpu --format=csv    # utilization should read ~0 %
cd Roofline_tutorial               # or wherever you cloned this repo
```

### 5-8 min — Sanity check

```bash
cd stage1_naive
python3 -m pytest test_correctness.py -v
```
Expect `10 passed`. If this fails, stop here and fix it (see §11) before continuing.

### 8-16 min — Stage 1: naive baseline — *see* the overhead problem

```bash
python3 vqe_naive.py --molecule h2 --depth 3       # note the printed wall time
python3 gqe_naive.py --batch-size 8                # note this one too -- it's much slower

nsys profile --trace=cuda,nvtx,osrt -o /tmp/s1_vqe --force-overwrite=true \
    python3 vqe_naive.py --molecule h2 --depth 3
nsys stats --report cuda_gpu_kern_sum /tmp/s1_vqe.nsys-rep
```
Look at the **Instances** column and each kernel's **Avg (ns)** — expect on
the order of 100-200 kernel launches, most only a few microseconds each.
That's the overhead-bound diagnosis, directly visible, no roofline chart
needed yet. Skim `stage1_naive/NOTES.md`'s "What's inefficient" section (2 min).

### 16-24 min — Stage 2: fuse, batch, stop syncing

```bash
cd ../stage2_gpu_native
python3 vqe_fused.py --molecule h2 --depth 3            # compare wall time to stage1
python3 vqe_custatevec.py --molecule h2 --depth 3       # vendor-library comparison, same energy

nsys profile --trace=cuda,nvtx,osrt -o /tmp/s2_vqe --force-overwrite=true \
    python3 vqe_fused.py --molecule h2 --depth 3
nsys stats --report cuda_gpu_kern_sum /tmp/s2_vqe.nsys-rep
```
Compare the kernel-launch count against stage1's — fusion + batching should
visibly shrink it. `vqe_custatevec.py` should print the same energy as
`vqe_fused.py` — same physics, NVIDIA's library implementation.

### 24-32 min — Stage 3: hand-written kernels, memory traffic

```bash
cd ../stage3_memory_opt
nvidia-smi --query-gpu=compute_cap --format=csv     # confirm sm_XX for your GPU (A100 = 80)
CUDA_ARCH=sm_80 ./build.sh                          # compiles in a few seconds
python3 -m pytest test_correctness.py -v            # confirm the kernels are correct
python3 vqe_memopt.py --molecule h2 --precision fp64
python3 vqe_memopt.py --molecule h2 --precision fp32   # compare wall time + energy (tiny diff)

nsys profile --trace=cuda,nvtx,osrt -o /tmp/s3_vqe --force-overwrite=true \
    python3 vqe_memopt.py --molecule h2 --precision fp64
nsys stats --report cuda_gpu_kern_sum /tmp/s3_vqe.nsys-rep
```
This is the sharpest reveal in the tutorial: expect **exactly 2 kernel
instances** — `apply_ansatz_shared` and `expval_pauli_batched_shared` —
regardless of circuit depth or Hamiltonian size. Worth pausing on.

### 32-40 min — Stage 4: occupancy + tensor cores

```bash
cd ../stage4_compute_opt
CUDA_ARCH=sm_80 ./build.sh
python3 -m pytest test_correctness.py -v
python3 vqe_compute_opt.py --molecule h2          # batched across SMs (one block per statevector)
python3 gqe_multistream.py --batch-size 32        # WMMA tensor-core GEMM + concurrent streams
```
Talk through the one deliberate asymmetry: VQE does **not** use tensor
cores here (a 2x2 gate wastes >99% of a 16x16x16 tile — see
`kernels/gate_apply_tc.cu`'s header) and wins from occupancy instead; GQE's
linear layers are genuinely GEMM-shaped, so tensor cores make sense there.
That's a "know when *not* to reach for a tool" moment, not an oversight.

### 40-45 min — Wrap up

- Line up the wall times you saw printed across all four `vqe_*.py` runs —
  no profiling needed for this comparison, you already have the numbers.
- Step through `reference_results/story/roofline_story_vqe_1_a100_pm.png` … `_4_a100_pm.png`
  (then the GQE set): each slide adds one stage, with an arrow showing the
  speedup. The kernel-count story you just watched live is the same story
  those slides tell visually.
- Take-home: run each stage's full `QUBITS=12 D_MODEL=128 BATCH=256 GQE_BATCH=256 ./profile.sh`
  (nsys timeline + ncu roofline + plots) on your own allocation. This
  generates your own roofline charts from your own GPU's numbers in
  `profiles/roofline_plots/` and the repo-root `roofline_hierarchical_*.png`
  (see "Quick start" at the top).

---

## 10. Expected results

`reference_results/` holds the roofline plots and per-kernel tables from
this tutorial's reference run: an A100-SXM4-40GB on a dedicated Perlmutter
node, every stage's `profile.sh` run with `QUBITS=12 D_MODEL=128 BATCH=256
GQE_BATCH=256`. Absolute numbers are hardware-specific, so compare the
*shape* of your results with these, not the digits. The reference files
are tagged `_a100_pm`, so your own runs (which write untagged plots under
`stage*/profiles/` and the repo root) never overwrite them.

Ceilings under ncu (clocks locked at ~1.1 GHz): HBM ≈1,535 GB/s, L2
≈5,300 GB/s, L1 ≈14,900 GB/s, FP64 FMA ≈7.5 TFLOP/s, FP16 tensor
≈238 TFLOP/s. stage0's measured HBM bandwidth was 1,362 GB/s. AI is
arithmetic intensity in FLOP/byte; "HBM AI" uses DRAM bytes.

**VQE** (12 qubits, random 576-term Hamiltonian, depth 3)

| Stage | `gate_apply` | `expval_reduce` | Whole workload |
|---|---|---|---|
| 1 — naive | `zgemm1x1_kernel_core`, 1 launch/gate: 19.7 GFLOP/s, HBM AI 1.2 | not captured¹ | 7.7 GFLOP/s |
| 2 — fused (CuPy) | `zgemm1x1_kernel_core`, 33 launches: 19.6 GFLOP/s, HBM AI 1.2 | batched CuPy elementwise ops, e.g. `cupy_multiply`: 251 GFLOP/s, HBM AI 0.19 | 28.7 GFLOP/s |
| 2 — cuStateVec | `deviceMatAppl*`: ~15 GFLOP/s, HBM AI 0.9 | `expectationPauliKernel`: 29 GFLOP/s, HBM AI 4.9 | 20.5 GFLOP/s |
| 3 — shared mem, fp64 | `apply_ansatz_shared`, 1 launch: 16.0 GFLOP/s, HBM AI 28 | `expval_pauli_batched_shared`, 1 launch: 20.4 GFLOP/s, HBM AI 400 | 20.0 GFLOP/s |
| 3 — shared mem, fp32 | 24.6 GFLOP/s, HBM AI 51 | 25.5 GFLOP/s, HBM AI 693 | 25.3 GFLOP/s |
| 4 — batched grid | `apply_ansatz_batched`: 505 GFLOP/s, HBM AI 30 | `expval_pauli_batched_grid`: 657 GFLOP/s, HBM AI 490 | 641 GFLOP/s |

¹ Stage 1's `profile.sh` captures only the first 60 kernel launches
(`--launch-count 60`), which at 12 qubits are all gate applications.

How to read it:
- **Stages 1→2:** stage 1 launches one tiny kernel per gate with a host
  sync after each. Stage 2 fuses gates and batches all Hamiltonian terms
  into a few large elementwise ops, so whole-workload throughput rises
  about 4x. Both stay far below the memory roof: they're overhead- and
  launch-bound, which the Nsight Systems timeline shows best.
- **Stage 3:** the whole circuit and all Pauli terms run inside
  shared memory in just 2 kernel launches. HBM AI jumps from ~0.2 to
  hundreds, so DRAM traffic is essentially gone, but GFLOP/s stays flat.
  Each kernel runs as one thread block on one SM, so the GPU is mostly idle.
- **Stage 4:** one block per statevector in a batch of 32 spreads the same
  kernels across the GPU: about 30x stage 3's throughput at the same AI.
  That's the move from memory-traffic-bound to occupancy/compute-bound.
  Even so, it reaches under 10% of the FP64 FMA ceiling, because a
  64 KB statevector is small work for an A100.

**GQE** (batch 256, d_model 128, 2 layers)

| Stage | Dominant GEMM kernel(s) | Whole workload |
|---|---|---|
| 1 — naive | cuBLAS `gemmSN_TN_kernel<float>`, per-sample loop: 31–107 GFLOP/s | 6.8 GFLOP/s |
| 2 — batched | cuBLAS `gemmSN_NN_kernel<float>`, batched: 2.1–4.8 TFLOP/s | 706 GFLOP/s |
| 3 — "fp16" | cuBLAS `gemmSN_NN_kernel<float>`: 2.5–4.8 TFLOP/s, **no tensor-core FLOPs**² | 577 GFLOP/s |
| 4 — WMMA tensor core | hand-written `batched_gemm_tc_fp16`: 12.1 TFLOP/s (5% of the FP16 tensor ceiling), HBM AI 73 | 5,912 GFLOP/s |

² For these skinny GEMM shapes cuBLAS picks a CUDA-core fp32 kernel even
though the inputs are fp16, so stage 3's fp16 storage halves memory
traffic but doesn't engage tensor cores. Only stage 4's hand-written WMMA
kernel actually runs on tensor cores, which you can see from its nonzero
`sm__ops_path_tensor_*` counters in ncu.

---

## 11. Troubleshooting

**`ncu` fails with `Profiling failed because a driver resource was
unavailable`.** Another tool (commonly NVIDIA's DCGM cluster-monitoring
daemon) is holding the GPU's hardware performance counters. Every
`profile.sh` in this tutorial already runs `dcgmi profile --pause` /
`--resume` around its `ncu` calls automatically — if you're invoking `ncu`
by hand and hit this, do the same:
```bash
dcgmi profile --pause
ncu ...
dcgmi profile --resume
```
If `dcgmi` isn't installed, this is a non-issue for you; if it IS installed
and pausing doesn't help, check `ps aux | grep dcgm` and consult your
cluster's sysadmin.

**`ncu` fails with `ERR_NVGPUCTRPERM` (no permission to access GPU
performance counters).** Common on workstations: by default the NVIDIA
driver lets only admins read the counters. Either run `ncu` (i.e.
`profile.sh`) with `sudo`, or have an admin allow all users. On Linux,
add `options nvidia NVreg_RestrictProfilingToAdminUsers=0` to a file in
`/etc/modprobe.d/` and reboot. Details:
[developer.nvidia.com/ERR_NVGPUCTRPERM](https://developer.nvidia.com/ERR_NVGPUCTRPERM).

**`nvcc: command not found` when running `build.sh`.** The CUDA Toolkit
isn't loaded/installed. On Perlmutter, `module load pytorch/2.13.0`
provides a working `nvcc`; check with `nvcc --version` after loading.

**`build.sh` compiles but tests fail with garbage/NaN energies.** Almost
always a `CUDA_ARCH` mismatch — you compiled for the wrong compute
capability. Re-check with `nvidia-smi --query-gpu=compute_cap --format=csv`
and rebuild with the matching `CUDA_ARCH=sm_XX`.

**Timings look absurdly slow, or don't improve stage-to-stage.** You're
probably on a shared/contended GPU — see §5. Get a dedicated allocation and
re-measure before drawing any conclusions.

**`pip install` fails or installs the wrong CuPy variant.** `cupy-cudaXXx`
and `cuquantum-python-cuXX` must match your CUDA *major* version (12 or
13), not your driver version — check with `nvcc --version`, not
`nvidia-smi`'s header number.

**Something else.** Each stage's `NOTES.md` and this project's
`ENV_SETUP.md` have more detail than fits here; `CLAUDE.md` documents the
full internal architecture (bit-ordering conventions, shared `common/`
modules, etc.) if you're going to modify the code rather than just run it.

---

## 12. Repository layout

```
common/                  Shared, must-stay-bit-compatible code every stage builds on:
                          ansatz.py, hamiltonians.py, reference_cpu.py (the correctness
                          oracle), gqe_policy_ref.py, config.py (shared CLI args)
stage0_calibration/       Bandwidth + GEMM peak-FLOPs probes, RESULTS.md
stage1_naive/             Naive CuPy baseline
stage2_gpu_native/        Fused/batched/streamed CuPy + cuStateVec comparison
stage3_memory_opt/        Hand-written CUDA kernels, shared memory, mixed precision
stage4_compute_opt/       Batched/occupancy-tuned kernels, WMMA tensor-core GEMM
plot_hierarchical_roofline.py  L1/L2/HBM roofline plots (same numbers as the ncu GUI); run by every profile.sh
reference_results/        Roofline plots + per-kernel tables from the A100/Perlmutter reference run (§10; files tagged _a100_pm)
tools/                    run_roofline_analysis.sh (whole analysis in one go, for instructors),
                          verify_vs_gui.py (checks the PNGs against ncu's GUI formulas), and the
                          one-time H2/LiH Hamiltonian generator (not a runtime dependency)
ENV_SETUP.md              The long-form version of sections 1-3 above
CLAUDE.md                 Architecture reference for anyone (human or AI) modifying the code
```

Each stage directory also has its own `NOTES.md` (the actual tutorial
content — the diagnosis and the fix) and, for stage1/2, no build step; for
stage3/4, a `build.sh` and `kernels/*.cu`.
