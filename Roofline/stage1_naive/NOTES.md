# Stage 1: Naive Baseline — Notes

## What this stage is

A straightforward, legitimate first attempt at running the VQE statevector
simulation and the GQE policy forward pass on a GPU using CuPy. This is
what you'd write if you knew NumPy well and had never done GPU performance
engineering before. It is **not** a strawman — it's correct, it runs, and
it's the kind of code every one of us has shipped as a "v1."

## What's inefficient about this version

**VQE (`vqe_naive.py`):**

1. **One kernel launch per gate, with a host sync after every single one.**
   A depth-3 H2 circuit has `2*4*3 (rotations) + 3*3 (CNOTs) = 33` gates.
   That's 33 separate `cp.tensordot`/`cp.moveaxis` kernel sequences, each
   followed by `cp.cuda.Stream.null.synchronize()`. The GPU never gets to
   queue up work; the CPU launches one tiny kernel, waits, launches the
   next.
2. **No gate fusion.** RY then RZ on the same qubit is two full passes over
   the 2^n-element statevector (read + write global memory twice) when it
   could be one fused 2x2 matrix (a single combined rotation) applied once.
3. **Dense Hamiltonian-term evaluation, one term at a time.** For every
   Pauli string, we build a full `2^n x 2^n` matrix via repeated `cp.kron`
   (H2: `15` such matrices, each `16x16`; LiH: `105` matrices, each
   `256x256`), do a dense matvec, and then call `.get()` to pull one scalar
   back to host — 15 (or 105) independent host round-trips for what is
   logically a single batched reduction against a shared statevector.
4. All complex128, no attempt at coalescing, tiling, or reuse.

**GQE (`gqe_naive.py`):** the equivalent sin — samples are processed one at
a time in a Python loop no matter what `--batch-size` is, and every linear
layer / layernorm / softmax / GELU is its own kernel + sync, so a 2-layer,
32-dim transformer forward pass on 8 samples issues on the order of 100
tiny kernel launches.

## What the Nsight Systems timeline is expected to show

Run `./profile.sh` and open `profiles/stage1_vqe_timeline.nsys-rep` (or run
`nsys stats profiles/stage1_vqe_timeline.nsys-rep` on the CLI). Expect:

- A long, thin sequence of very short kernels (each processing at most a
  few KB of statevector — trivially small for a GPU).
- Visible **idle gaps** between kernels corresponding to each
  `synchronize()` call and the CPU-side Python overhead of building the
  next NumPy/CuPy gate matrix.
- Low "GPU busy %" in the timeline summary — most of the wall-clock time
  is *not* spent doing GPU work, it's spent on launch overhead and
  synchronization stalls.
- For the GQE trace, an even more extreme version of the same pattern,
  since the transformer forward pass has many more distinct op types
  (matmul, layernorm, softmax, GELU) each launched separately, per sample.

## What the Nsight Compute roofline plot is expected to show

Run the `ncu --set roofline` command in `profile.sh` and open
`profiles/stage1_vqe_roofline.ncu-rep`. Expect:

- The gate-application kernel and the Pauli-matvec kernel to land **far
  below both rooflines** (the memory-bandwidth roof and the compute/FLOP
  roof) on the roofline chart, achieving only a small fraction of either.
- This is *not* primarily a memory-bandwidth or compute-throughput
  problem at this qubit count — at 4-12 qubits (16 to 4096 statevector
  elements), the actual data volume per kernel is tiny (KBs), so the
  bottleneck is **kernel launch and synchronization overhead**, which the
  roofline model doesn't directly capture but which shows up as "the
  achieved GFLOP/s and achieved GB/s are both minuscule relative to what a
  kernel touching this little data should take essentially no measurable
  time to move/compute."
- Practically: expect achieved arithmetic intensity to look moderate (it's
  a ratio of the actual, tiny amount of memory traffic and compute done per
  kernel), but the *absolute* GB/s and GFLOP/s numbers will be minuscule
  because each kernel runs for such a short, launch-overhead-dominated
  time.

## What change leads into Stage 2

The diagnosis from this profile: we are latency-bound and launch-overhead
-bound, not memory- or compute-bound. Stage 2 fixes this by:

- **Fusing gate application** — combine sequences of single-qubit gates
  into one matrix per qubit-group before hitting the statevector, so one
  kernel does the work of several.
- **Batching Pauli-term expectation values** — instead of 15 (or 105)
  separate dense matvecs + host round-trips, use `custatevec`'s or a
  hand-rolled batched-Pauli-expectation kernel that reduces all terms in
  one pass over the statevector.
- **Removing host syncs** — only synchronize once, after the full circuit
  + full Hamiltonian evaluation, letting the GPU's own stream ordering
  handle dependencies.
- **Streams** — for the GQE case, run multiple samples' forward passes
  concurrently on separate CUDA streams instead of serializing them in a
  Python loop.

This should turn the Nsight Systems timeline from "many small gapped
kernels" into "a few large back-to-back kernels with no gaps," and should
be visible in Nsight Compute as fewer, larger kernel invocations — setting
up stage 3's question of whether those larger kernels are now
memory-bound (and if so, how far below the memory roofline they sit).
