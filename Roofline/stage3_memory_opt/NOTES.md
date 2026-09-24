# Stage 3: Memory-Traffic Reduction — Notes

## What changed from Stage 2

Stage 2 fused gates and batched Hamiltonian-term evaluation, but every gate
application and the batched expectation-value reduction still read/wrote
the full statevector from **global** memory — stage2's `moveaxis`/
`tensordot` gate application does `n_gates` separate global-memory
read-modify-write passes, and its `batched_pauli_expectation` does an
XOR-indexed `take_along_axis` gather from global memory.

Stage 3 replaces both with hand-written CUDA C++ kernels
(`kernels/gate_apply.cu`, `kernels/expval_reduce.cu`, compiled by
`./build.sh` into `.cubin` files loaded via `cupy.RawModule`):

1. **Whole-circuit-in-shared-memory gate application.** `apply_ansatz_shared`
   loads the entire `2^n`-element statevector into on-chip shared memory
   ONCE, applies every rotation and every CNOT of the whole circuit as
   in-place shared-memory updates (with `__syncthreads()` between gates for
   correctness), and writes the result back to global memory ONCE. Global
   memory traffic on the statevector drops from `O(n_gates)` full passes to
   exactly 2 (one load, one store) — independent of circuit depth.

2. **Statevector tile reuse across all Pauli terms.**
   `expval_pauli_batched_shared` loads the statevector into shared memory
   ONCE and reuses it across every one of `n_terms` Hamiltonian terms
   (LiH: 105 terms), each contributing a block-wide parallel reduction.
   Global memory traffic drops from `O(n_terms)` full passes to 1.

3. **Coalesced access.** Both kernels load/store the statevector with
   `for (i = tid; i < dim; i += nthreads)` — consecutive threads touch
   consecutive addresses, which is the textbook coalesced pattern, unlike
   stage2's `moveaxis`-then-strided-access pattern (whose stride depends on
   which qubit's bit position is being operated on).

4. **Mixed precision (FP32).** `kernels/gate_apply_fp32.cu` and
   `kernels/expval_reduce_fp32.cu` run the identical algorithm on
   `cuFloatComplex` (8 bytes/amplitude) instead of `cuDoubleComplex` (16
   bytes/amplitude) — half the global-memory traffic for the load/store
   bookends, at the cost of precision (see Correctness below).
   `gqe_batched_fp16.py` does the same for the GQE policy network, at fp16
   (2 bytes per value), with softmax/layernorm accumulation kept in fp32
   for numerical stability — a standard practice for low-precision
   transformers.

Net effect for a full VQE energy evaluation: statevector global-memory
traffic drops from stage2's `O(n_gates + n_terms)` full passes to exactly
**3** full passes total (load for gate application, store, load for
expectation-value reduction) — independent of circuit depth or Hamiltonian
size, and halved again under FP32.

## Correctness

`test_correctness.py` checks:
- FP64 kernels match the CPU reference to `< 1e-6` Ha (they should be
  essentially exact — same math, same precision, just different memory
  layout).
- FP32 kernels match to a **looser** `< 1e-5` Ha tolerance — this is a real
  measured tradeoff, not zero-cost. In practice we saw FP32 VQE energies
  land within `~1e-7 - 1e-6` Ha of the FP64 reference for H2/LiH depth 1-3,
  comfortably inside a chemist's usual "chemical accuracy" target
  (`1.6e-3` Ha) with room to spare.
- FP16 GQE logits match to `< 0.01` absolute — fp16 has roughly 3 decimal
  digits of precision, so this is the tightest tolerance that format can
  reasonably support.
- The FP64 kernel is verified across the **full 4-12 qubit range** the
  tutorial promises, confirming the whole-circuit-in-shared-memory design
  fits within the GPU's dynamic shared memory budget (up to 64KB for 12
  qubits at complex128, comfortably under the A100's ~163KB opt-in limit
  — see `kernels/gate_apply.cu`'s header comment).

## What the Nsight Systems timeline is expected to show

Compare `profiles/stage3_vqe_fp64_timeline.nsys-rep` against stage2's.
Expect:

- Exactly **2 kernel launches** for the whole VQE energy evaluation (one
  `apply_ansatz_shared` call, one `expval_pauli_batched_shared` call) —
  down from stage2's "one kernel per gate, plus one batched CuPy operation
  for expectation values."
- Each of those 2 kernels should run measurably LONGER per launch than any
  single stage2 kernel, because each one is now doing the equivalent of
  many gates'/terms' worth of work internally, entirely out of shared
  memory, without any intervening global-memory round trips.
- The FP32 timeline should look structurally identical, just with smaller
  data sizes moving in the global load/store phases of each kernel.

## What the Nsight Compute roofline plot is expected to show

Open `profiles/stage3_vqe_fp64_roofline.ncu-rep` and
`profiles/stage3_vqe_fp32_roofline.ncu-rep`. Expect:

- Both kernels' achieved GB/s should be noticeably higher than stage2's
  equivalent kernels — with global-memory access now coalesced and
  amortized (2-3 full passes total instead of many), the fraction of the
  memory roofline achieved should increase substantially.
- The FP32 point should sit at roughly HALF the arithmetic intensity in
  bytes-moved terms compared to FP64 for the same logical operation (half
  the bytes for the same number of "useful" floating-point comparisons),
  which should visibly shift its position on the roofline chart relative
  to the FP64 point — this is the clearest illustration in the whole
  tutorial of what "arithmetic intensity" means in practice.
- Kernels should now be measurably closer to the memory roofline than in
  any previous stage, though likely still short of actually touching it —
  achieving that last gap (register-level optimization, launch
  configuration tuning, occupancy) is stage 4's job, along with asking
  whether the GQE policy_forward kernel — which involves real GEMMs, unlike
  the statevector kernels — could be moved toward the COMPUTE roofline
  instead via tensor cores.

## What change leads into Stage 4

At this point the statevector kernels are memory-traffic-efficient but
still running on plain CUDA cores with a straightforward, unoptimized
launch configuration (fixed thread count, no attempt to reason about
occupancy or warp-level efficiency). Meanwhile the GQE policy_forward
kernel is fundamentally GEMM-shaped (attention and MLP layers are
matrix multiplies) and, at larger batch sizes, has enough arithmetic
intensity to potentially become compute-bound rather than memory-bound —
which means it's the right candidate for tensor-core acceleration. Stage 4
addresses both: occupancy/launch-configuration tuning for the statevector
kernels, and a tensor-core (WMMA) GEMM path plus multi-stream scaling for
the GQE sampling loop.
