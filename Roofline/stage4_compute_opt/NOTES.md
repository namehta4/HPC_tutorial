# Stage 4: Compute-Bound Tuning — Notes

## What changed from Stage 3

Stage 3 made each individual kernel memory-traffic-efficient (coalesced
access, shared-memory statevector reuse) but each VQE kernel launch still
used exactly **one CUDA thread block**, meaning a single evaluation
occupied at most 1 of the GPU's 108 Streaming Multiprocessors (A100). Stage
4 addresses two different bottlenecks for the two workloads:

**VQE (`vqe_compute_opt.py`, `kernels/gate_apply_tc.cu`): occupancy, not
tensor cores.** VQE's gate-application kernel operates on 2x2 single-qubit
unitaries. Tensor cores compute >=16x16x16 tile matrix multiplies — forcing
a 2x2 gate through one would waste over 99% of the tile on zero padding, so
this tutorial does NOT claim tensor cores help VQE's core kernel (see the
header comment in `kernels/gate_apply_tc.cu` for the full argument — we
think this honesty matters more than a contrived "tensor cores everywhere"
story). Instead, we fix the *real* compute-bound bottleneck: **occupancy**.
`apply_ansatz_batched` and `expval_pauli_batched_grid` launch **one CUDA
block per statevector** in a batch (`grid.x = batch_size`), so a batch of
independent VQE evaluations (e.g. the `2 * n_params` evaluations a
parameter-shift-rule gradient needs, or a batch of GQE-proposed candidate
parameter sets) occupies many SMs concurrently instead of running
single-SM kernels back-to-back.

**GQE (`gqe_multistream.py`, `kernels/policy_forward_gemm.cu`): genuine
tensor-core GEMM.** The policy network's linear layers ARE real matrix
multiplies — exactly tensor cores' natural shape. `batched_gemm_tc_fp16`
is a hand-written WMMA (Warp Matrix Multiply-Accumulate) kernel: each warp
computes one 16x16 output tile using the tensor core's native
fp16-multiply/fp32-accumulate primitive. Every linear layer in the policy
network (`qkv`, `proj`, both MLP layers, `head`) now routes through this
kernel. `run_multistream_sampling_loop` additionally runs several
independent sampling batches concurrently on separate CUDA streams,
modeling the realistic GQE pattern of overlapping generation with
downstream energy evaluation or exploring multiple sampling temperatures at once.

## Correctness

- VQE batched energies are checked against the CPU reference at
  `< 1e-6` Ha (should be essentially exact — same algorithm as stage3,
  just grid-parallelized), across the full 4-12 qubit range.
- GQE tensor-core logits are checked against the fp64 PyTorch reference at
  a looser `< 0.05` absolute tolerance (fp16 GEMM path, similar precision
  budget to stage3's fp16 forward pass).
- The multi-stream sampling loop is checked for correct output shapes and
  the absence of NaNs across all streams.

## What the Nsight Systems timeline is expected to show

Compare `profiles/stage4_vqe_timeline.nsys-rep` against stage3's. Expect:

- Still exactly 2 kernel launches for the whole batched-VQE evaluation
  (one `apply_ansatz_batched` call, one `expval_pauli_batched_grid` call)
  — but now each kernel's grid spans `batch_size` blocks instead of 1, so
  the SAME 2-launch structure now does `batch_size` times the useful work
  per launch.
- For the GQE tensor-core trace, expect several `batched_gemm_tc_fp16`
  kernel invocations (one per linear layer per transformer block) — Nsight
  Systems should show these as tensor-core-active kernels if you enable
  `--gpu-metrics-device` capture with tensor-core utilization counters.
- The multi-stream trace should show 2+ streams' kernels genuinely
  overlapping in time on the timeline, rather than one stream's kernels
  finishing before the next stream's begin.

## What the Nsight Compute roofline plot is expected to show

Open `profiles/stage4_vqe_roofline.ncu-rep` and
`profiles/stage4_gqe_tc_roofline.ncu-rep`. Expect:

- The batched VQE kernels' achieved GB/s should be substantially higher
  than stage3's single-block kernels at the same qubit count — same
  algorithm, but now running on many SMs concurrently instead of one,
  which is exactly what "occupancy" buys you: more of the GPU's total
  memory bandwidth and compute throughput actually gets used per unit
  wall-clock time.
- The GQE `batched_gemm_tc_fp16` kernel should show measurably higher
  achieved GFLOP/s than stage3's plain-CuPy fp16 matmuls, and should sit
  closer to the **compute roofline** (specifically, closer to the fp16
  tensor-core ceiling measured in `stage0_calibration/RESULTS.md`) rather
  than the memory roofline — this is the one kernel in the whole tutorial
  where "compute-bound" is a meaningful, achievable target, because GEMMs
  have genuinely high arithmetic intensity (`O(M*N*K)` FLOPs over
  `O(M*K + K*N + M*N)` bytes) unlike the statevector kernels' fundamentally
  memory-bound-at-this-scale Pauli operations.
- At this tutorial's default small sizes (`d_model=32`, modest batch), the
  tensor-core GEMM may still be far from its own ceiling simply because the
  matrices are small relative to a 16x16x16 tile — re-run with larger
  `--d-model` / `--batch-size` to see the achieved-GFLOP/s point move
  further up the compute roofline as problem size grows.

## Where this leaves the tutorial

This is the tutorial's final stage. The overall arc, visible in
`plot_roofline.py`'s combined chart: stage1's kernels sit far below both
rooflines (launch-overhead-bound); stage2 moves them measurably closer to
the memory roofline (fusion + batching removes overhead, revealing the
underlying memory-bound-ness); stage3 pushes further toward the memory
roofline (coalescing + shared-memory reuse actually reduces bytes moved);
stage4 either extracts more of the memory roofline via occupancy (VQE) or
crosses over toward the compute roofline entirely via tensor cores (GQE).
That progression — overhead-bound -> memory-bound -> compute-bound, with a
concrete GPU-code change motivating each transition — is the whole point
of this tutorial.
