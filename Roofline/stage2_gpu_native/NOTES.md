# Stage 2: GPU-Native Restructuring — Notes

## What changed from Stage 1

**VQE (`vqe_fused.py`):**

1. **Fused rotation gates.** RY(theta) then RZ(phi) on the same qubit are
   pre-multiplied on the host into one 2x2 matrix (`RZ @ RY`) and applied to
   the statevector in a single pass, instead of two.
2. **No synchronization until the final energy value is needed.** All gate
   applications and the batched expectation-value reduction are enqueued on
   CUDA's default stream and execute in the GPU's own order; the host only
   blocks once, on `.get()` at the very end.
3. **Batched, algebraic Pauli-term expectation values.** Instead of
   building a dense `2^n x 2^n` matrix per term, we exploit the fact that a
   Pauli string's action on a computational basis state is (a) an XOR bit
   flip and (b) a sign/phase that depends on the pre-flip bit values —
   both vectorizable over the full `2^n`-element index space AND over all
   `n_terms` Hamiltonian terms simultaneously, using CuPy broadcasting. See
   `batched_pauli_expectation()`'s docstring for the derivation. This is
   `O(2^n * n_terms)` instead of stage1's `O(4^n)` per term.
4. **CUDA streams** (`vqe_energy_multi_params_streamed`) let independent
   parameter-set evaluations (e.g., parameter-shift-rule gradient terms)
   run concurrently instead of serially.
5. **`vqe_custatevec.py`** shows the same circuit run through NVIDIA's
   cuStateVec library (`apply_matrix` + `compute_expectations_on_pauli_basis`)
   as a "what good looks like, from a library" comparison point — see that
   file's docstring for the bit-index convention mapping needed (cuStateVec
   is little-endian; this tutorial is big-endian).

**GQE (`gqe_batched.py`):** the whole batch is processed by a handful of
batched-GEMM calls (via `cp.matmul` / `@` on tensors with a leading batch
dimension) instead of stage1's per-sample Python loop. Weights are uploaded
once and cached, not re-uploaded on every call.

## What the Nsight Systems timeline is expected to show

Compare `profiles/stage2_vqe_fused_timeline.nsys-rep` against stage1's
timeline. Expect:

- Roughly half as many gate-application kernel launches (fusion merged
  RY+RZ pairs).
- **No visible gaps** between kernels — the synchronization stalls that
  dominated stage1's timeline are gone; kernels back up against each other.
- One Hamiltonian-evaluation region instead of 15 (or 105) separate
  dense-matvec-plus-host-sync regions.
- The `vqe_custatevec` timeline should look qualitatively similar in
  structure (few kernels, no gaps) — this is the point of the comparison.
- The GQE timeline should collapse from ~100 tiny per-sample kernels down
  to roughly (n_layers * a handful of ops) kernels regardless of batch size.

## What the Nsight Compute roofline plot is expected to show

Open `profiles/stage2_vqe_fused_roofline.ncu-rep`. Expect:

- Kernels now do meaningfully more work per launch (larger effective batch
  of statevector elements/terms processed together), so achieved GB/s and
  achieved GFLOP/s should be measurably higher than stage1's — the point on
  the roofline chart should move up and to the... well, probably still to
  the left of the ridge point, because the *problem itself* still doesn't
  move enough bytes to be compute-bound at 4-12 qubits. But it should now
  sit much closer to the **memory roofline**, since a fused, unsynchronized
  kernel finally has enough work queued to make its actual memory-traffic
  efficiency visible instead of being swamped by launch overhead.
- Expect the batched Pauli-expectation kernel in particular to show a
  large improvement, since it went from `O(4^n)` dense work per term to
  `O(2^n)` per term across all terms in one shot.
- This is the moment where "are we memory-bound or compute-bound" becomes
  a meaningful question to ask (in stage1 the answer was "neither — we're
  overhead-bound").

## What change leads into Stage 3

Now that kernels are fused and un-gapped, the profile should reveal a new
bottleneck: **memory access pattern**. The `moveaxis`/`tensordot`-based gate
application still does non-coalesced global memory access (each thread's
memory accesses are strided by the qubit's bit position, not sequential),
and the batched Pauli-expectation kernel does a `take_along_axis` gather
with an XOR-computed index — also non-coalesced. Stage 3 addresses this
directly: hand-written CUDA kernels with coalesced access patterns, shared
memory to reduce redundant global memory reads, and statevector tile reuse
across multiple Pauli terms evaluated in the same kernel launch. We'll also
introduce mixed precision (FP32/TF32) as a further memory-traffic
reduction, since complex128 storage moves twice the bytes of complex64 for
the same information content (at the cost of numerical precision, which
stage3's tests will explicitly verify stays within tolerance).
