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

**A note on problem size before you start:** the commands below default to
`--molecule h2` (4 qubits) for VQE and small batch/`d-model` for GQE — that's
the right size for a *quick correctness check* (fast to run, easy to
eyeball), but it is **too small to produce a visually convincing roofline
plot**. At H2/depth-3 scale, stage1/2/3/4's VQE kernels move only a few KB
of data each, so even stage4's occupancy-tuned kernels land in
approximately the same bottom-left corner of the chart as stage1's —
correct, but not compelling. To see the actual overhead → memory → compute
progression on the chart (§10 below), re-run stage1/2/3/4's VQE commands
with `--molecule random --qubits 12` once you've confirmed correctness at
the default size, and re-run GQE with a larger `--batch-size`/`--d-model`
(e.g. `--batch-size 256 --d-model 128`) for stage4's tensor-core kernel to
do meaningfully more work per launch.

### Stage 0 — Calibrate this GPU's real ceilings

Do this first, on a dedicated/idle GPU (see §5):

```bash
cd stage0_calibration
./run_calibration.sh
```

This measures achieved memory bandwidth (STREAM-style Triad) and achieved
GEMM FLOP/s (FP64, TF32, FP16) on *your actual GPU*, and writes
`bandwidth_fp64.json` / `gemm_peak.json`. Copy the printed numbers into the
table at the bottom of `stage0_calibration/RESULTS.md` — every later
roofline plot uses these as its ceilings, not vendor spec sheets.

**Sanity-check your FP32/TF32 number before trusting it.** On at least one
CuPy/cuBLAS build we measured, the cuBLAS handle CuPy uses does *not*
default to `CUBLAS_TENSOR_OP_MATH` for FP32 inputs despite cuBLAS's own
docs suggesting Ampere+ does this automatically — `gemm_peak_probe.py`
now sets this explicitly, but if you're on an older checkout or a
different CuPy version and your measured FP32/TF32 number comes back close
to your FP64 number (instead of ~8x higher), your GPU almost certainly
supports TF32 tensor cores and the math mode just isn't being requested —
see `stage0_calibration/RESULTS.md`'s "TF32 anomaly" note for the fix.

### Stage 1 — Naive baseline

```bash
cd stage1_naive
python3 -m pytest test_correctness.py -v      # correctness first
python3 vqe_naive.py --molecule h2 --depth 3   # run it directly
python3 gqe_naive.py --batch-size 8
./profile.sh                                    # nsys + ncu -> profiles/

# Larger size -- needed for a visible point on the roofline overlay (see §6 note above)
python3 vqe_naive.py --molecule random --qubits 12 --depth 3
python3 gqe_naive.py --batch-size 64 --d-model 128
```

Read `stage1_naive/NOTES.md` afterward — it explains exactly what the
Nsight Systems timeline and Nsight Compute roofline report should look like
for this stage, and why.

### Stage 2 — GPU-native restructuring

```bash
cd stage2_gpu_native
python3 -m pytest test_correctness.py -v
python3 vqe_fused.py --molecule h2 --depth 3
python3 vqe_custatevec.py --molecule h2 --depth 3   # library comparison point
python3 gqe_batched.py --batch-size 32
./profile.sh

# Larger size
python3 vqe_fused.py --molecule random --qubits 12 --depth 3
```

### Stage 3 — Memory-traffic reduction (compile CUDA kernels first)

Stage 3 introduces hand-written CUDA kernels, which need to be compiled to
`.cubin` files before anything else in this stage will run:

```bash
cd stage3_memory_opt
CUDA_ARCH=sm_80 ./build.sh    # match sm_80 to YOUR GPU -- see §3 step 4
python3 -m pytest test_correctness.py -v
python3 vqe_memopt.py --molecule h2 --precision fp32
./profile.sh
```

If you edit any `kernels/*.cu` file, re-run `./build.sh` — nothing does
this for you automatically.

### Stage 4 — Compute-bound tuning

```bash
cd stage4_compute_opt
CUDA_ARCH=sm_80 ./build.sh    # same note as stage3
python3 -m pytest test_correctness.py -v
python3 vqe_compute_opt.py --molecule h2
python3 gqe_multistream.py --batch-size 32
./profile.sh

# Larger size -- this is the pair of runs that actually shows separation
# from stage1/2 on the roofline overlay
python3 vqe_compute_opt.py --molecule random --qubits 12 --depth 3 --batch-size 32
python3 gqe_multistream.py --batch-size 256 --d-model 128
```

### Putting it all together: the overlay roofline plot

Once you've run each stage's `profile.sh` and have `.ncu-rep` files in each
stage's `profiles/` directory, export them to CSV and overlay them on one
chart. **Use the `--qubits 12` runs from §6 above for this, not the
default H2 ones** — see the size note at the top of §6 for why: at H2
scale every stage's VQE kernel lands in the same corner of the chart and
the plot doesn't show the progression it's meant to.

```bash
# Profile the larger-size VQE run directly with ncu (profile.sh's default
# invocation targets H2; pause DCGM first the same way profile.sh does,
# or reuse profile.sh's MOLECULE/DEPTH env-var overrides if your stage
# script's args allow it):
dcgmi profile --pause   # skip if dcgmi isn't installed -- see §11
ncu --set roofline --export stage1_naive/profiles/stage1_vqe_q12_roofline --force-overwrite \
    python3 stage1_naive/vqe_naive.py --molecule random --qubits 12 --depth 3
ncu --set roofline --export stage2_gpu_native/profiles/stage2_vqe_q12_roofline --force-overwrite \
    python3 stage2_gpu_native/vqe_fused.py --molecule random --qubits 12 --depth 3
ncu --set roofline --export stage4_compute_opt/profiles/stage4_vqe_q12_roofline --force-overwrite \
    python3 stage4_compute_opt/vqe_compute_opt.py --molecule random --qubits 12 --depth 3 --batch-size 32
dcgmi profile --resume

# Export each stage's ncu report to CSV
ncu --import stage1_naive/profiles/stage1_vqe_q12_roofline.ncu-rep --csv --page raw > stage1_naive/profiles/stage1_vqe_q12_roofline.csv
ncu --import stage2_gpu_native/profiles/stage2_vqe_q12_roofline.ncu-rep --csv --page raw > stage2_gpu_native/profiles/stage2_vqe_q12_roofline.csv
ncu --import stage3_memory_opt/profiles/stage3_vqe_fp64_roofline.ncu-rep --csv --page raw > stage3_memory_opt/profiles/stage3_vqe_fp64_roofline.csv
ncu --import stage3_memory_opt/profiles/stage3_vqe_fp32_roofline.ncu-rep --csv --page raw > stage3_memory_opt/profiles/stage3_vqe_fp32_roofline.csv
ncu --import stage4_compute_opt/profiles/stage4_vqe_q12_roofline.ncu-rep --csv --page raw > stage4_compute_opt/profiles/stage4_vqe_q12_roofline.csv

# Overlay all four stages on one chart (peak values are TFLOP/s -- these are
# this tutorial's own measured A100-SXM4-40GB numbers from
# stage0_calibration/RESULTS.md; substitute YOUR stage0 numbers instead,
# not these)
python3 plot_roofline.py --peak-bw 589.5 --peak-flops-fp64 18.64 \
    --peak-flops-fp32 155.58 --peak-flops-fp16 278.01 \
    --csv stage1_naive/profiles/stage1_vqe_q12_roofline.csv \
    --csv stage2_gpu_native/profiles/stage2_vqe_q12_roofline.csv \
    --csv stage3_memory_opt/profiles/stage3_vqe_fp64_roofline.csv \
    --csv stage3_memory_opt/profiles/stage3_vqe_fp32_roofline.csv \
    --csv stage4_compute_opt/profiles/stage4_vqe_q12_roofline.csv \
    --out roofline_overlay.png
```

Substitute the `--peak-*` values with what *you* measured in stage0, not
the numbers above (which are only valid for the specific A100 this
tutorial was profiled on). At `--qubits 12`, the resulting
`roofline_overlay.png` shows stage1/2 staying flat around 5-10 GFLOP/s
(still overhead-bound regardless of problem size, as expected) while
stage3 reaches roughly 2-160 GFLOP/s and stage4 reaches roughly
4,700-6,700 GFLOP/s at much higher arithmetic intensity — real,
qualitative movement up and to the right, unlike the default H2-size plot.

**GQE's tensor-core kernel currently can't be plotted this way.**
`plot_roofline.py`'s automatic CSV parsing only recognizes FP64/FP32 FMA
instruction counters (see the module docstring), so it silently skips
stage4's fp16 `batched_gemm_tc_fp16` WMMA kernel — you can confirm it ran
and used tensor cores via `ncu`'s
`sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_elapsed` metric on
that kernel, but getting a correct point for it onto this chart requires
`--manual-points` with a hand-computed (arithmetic intensity, GFLOP/s)
pair for the kernel's actual (padded) tile dimensions — not yet automated.

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
- If you (or the facilitator) pre-generated `roofline_overlay.png` (§6's
  "Putting it all together" section, run ahead of time since each
  `profile.sh` takes several minutes), show it now — the kernel-count story
  you just watched live is the same story that chart tells visually.
- Take-home: run each stage's full `./profile.sh` (nsys timeline + ncu
  roofline) on your own allocation, then §6's CSV-export + `plot_roofline.py`
  steps, to generate your own roofline chart from your own GPU's numbers.

---

## 10. Expected results

The table below is filled in with *actual measured* results from this
tutorial's own reference run (A100-SXM4-40GB, dedicated Perlmutter
allocation, VQE at `--molecule random --qubits 12 --depth 3`,
`stage0_calibration/RESULTS.md`'s 2026-09-23 numbers) — replace it with
*your own* measurements once you've run your profiler on your GPU, since
absolute GFLOP/s and % of roofline are hardware-specific. Arithmetic
intensity (AI) is FLOPs moved per byte of DRAM traffic; the ridge point
(FP64: ~31.6 FLOP/byte, FP32/TF32: ~264 FLOP/byte, on this reference GPU)
is the AI where a kernel crosses from memory-bound to compute-bound.

| Stage | Kernel(s) tracked | Arithmetic Intensity (FLOP/byte) | Achieved GFLOP/s | % of FP64 roofline (18,640 GFLOP/s) |
|---|---|---|---|---|
| 1 — naive | `gate_apply` (`zgemm1x1_kernel_core`) | ~0.28 | ~4.6 | ~0.02% |
| 2 — GPU-native | `gate_apply` (fused, `zgemm1x1_kernel_core`) | n/a¹ | ~4.6 | ~0.02% |
| 3 — memory opt (fp64) | `gate_apply` (`apply_ansatz_shared`), `expval_reduce` (`expval_pauli_batched_shared`) | ~2.5-2.8 | ~1.3-1.7 | ~0.01% |
| 3 — memory opt (fp32) | same, fp32 | ~2.4-3.1 | ~1.2-2.0 | ~0.01% |
| 4 — compute opt | `gate_apply` (`apply_ansatz_batched`), `expval_reduce` (`expval_pauli_batched_grid`) | ~9.5 | ~12.5-158.7 | ~0.07-0.85% |

¹ Stage1 and stage2's `gate_apply` kernel moves so little DRAM traffic per
launch (a handful of KB) that the AI ratio is dominated by measurement
noise rather than anything meaningful — don't read too much into small AI
differences between stage1 and stage2 specifically; the GFLOP/s column and
kernel-launch-count story (§7, each stage's `NOTES.md`) are the more
reliable signal for these two stages, not roofline position.

**What this table actually shows:** even at `--qubits 12`, every VQE
kernel in this tutorial stays well under 1% of the FP64 roofline — that's
expected and correct, not a bug. Statevector sizes here (up to 4096 complex
values) are simply too small for this class of kernel to approach a
modern GPU's compute ceiling; the meaningful comparison is the ~7-90x
GFLOP/s improvement from stage1/2 to stage4 (not the absolute % of
roofline), plus the *kernel-launch-count* collapse (33+ launches in stage1
down to 2 in stage3/4) that the roofline chart alone doesn't capture. See
`roofline_overlay.png` (or `roofline_overlay_q12.png` if you're comparing
against this tutorial's own reference run) for the visual version of this
table, and each stage's `NOTES.md` "What the Nsight Compute roofline plot
is expected to show" section for the qualitative walkthrough.

GQE's tensor-core kernel (`batched_gemm_tc_fp16`, stage4) is confirmed to
actually execute on tensor cores (nonzero
`sm__pipe_tensor_cycles_active.sum.pct_of_peak_sustained_elapsed`) at
`--batch-size 256 --d-model 128`, but isn't in the table above because
`plot_roofline.py` doesn't yet parse fp16/tensor-core ncu columns — see
the note in §6 above.

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
plot_roofline.py          Overlays all four stages' ncu CSV exports on one roofline chart
tools/                    One-time generator for the hardcoded H2/LiH Hamiltonian data
                          (NOT a runtime dependency -- pre-verified output is checked in)
ENV_SETUP.md              The long-form version of sections 1-3 above
CLAUDE.md                 Architecture reference for anyone (human or AI) modifying the code
```

Each stage directory also has its own `NOTES.md` (the actual tutorial
content — the diagnosis and the fix) and, for stage1/2, no build step; for
stage3/4, a `build.sh` and `kernels/*.cu`.
