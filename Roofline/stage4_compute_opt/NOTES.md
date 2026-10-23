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

**GQE (`gqe_multistream.py`, `kernels/policy_forward_gemm.cu`): fused
tensor-core forward pass.** Stage 3 already put the linear layers on tensor
cores, and its profile showed those GEMMs were only ~7% of the GPU time.
The other ~90% was ~175 tiny memory-bound kernels (bias adds, GELU,
layernorm and softmax pieces, fp16<->fp32 casts), each a separate trip to
HBM. Stage 4 fuses that glue into the kernels that already hold the data:

- `gemm_tc_fp16_fused`: a hand-written WMMA (Warp Matrix Multiply-
  Accumulate) tensor-core GEMM (fp16 multiply, fp32 accumulate) for every
  linear layer (`qkv`, `proj`, both MLP layers, `head`). It applies the
  layer's epilogue (bias, GELU, or the residual add) before writing its
  output. Each layer is ONE GEMM over all batch × seq_len rows;
  multiplying each row separately as a 16-row-padded tile would waste
  15/16 of the tensor-core work on zeros.
- `layernorm_f32_to_f16` (the first one fused with the embedding,
  `embed_layernorm`): one warp per row.
- `causal_attention`: scores, causal mask, softmax and attn @ v for one
  (batch, head) in shared memory. It stays on CUDA cores: at seq_len 8
  each head is an 8x8x32 problem, too small for tensor cores to pay off.

A forward pass is 16 kernel launches instead of ~180, with the residual
stream kept in fp32. `run_multistream_sampling_loop` additionally runs several
independent sampling batches concurrently on separate CUDA streams,
modeling the realistic GQE pattern of overlapping generation with
downstream energy evaluation or exploring multiple sampling temperatures at once.

## Correctness

- VQE batched energies are checked against the CPU reference at
  `< 1e-6` Ha (should be essentially exact — same algorithm as stage3,
  just grid-parallelized), across the full 4-12 qubit range.
- GQE logits are checked against the fp64 PyTorch reference at a looser
  `< 0.05` absolute tolerance (fp16 GEMM inputs), at d_model 32 and 128.
  The measured error is ~8e-4 at d_model 128 (stage3: ~1.4e-3), lower
  because the residual stream stays in fp32.
- The multi-stream sampling loop is checked for correct output shapes and
  the absence of NaNs across all streams.

## What the Nsight Systems timeline is expected to show

Compare `profiles/stage4_vqe_timeline.nsys-rep` against stage3's. Expect:

- Still exactly 2 kernel launches for the whole batched-VQE evaluation
  (one `apply_ansatz_batched` call, one `expval_pauli_batched_grid` call)
  — but now each kernel's grid spans `batch_size` blocks instead of 1, so
  the SAME 2-launch structure now does `batch_size` times the useful work
  per launch.
- For the GQE trace, expect exactly 16 kernels per forward pass: 9
  `gemm_tc_fp16_fused` (4 linear layers × 2 blocks + head), 2
  `causal_attention`, 4 `layernorm_f32_to_f16` and 1 `embed_layernorm`,
  versus ~190 short kernels in stage3's trace.
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
- GQE: the whole-workload point moves up and to the right of stage3's
  (reference run: 770 → 3,436 GFLOP/s, GPU time 1,020 → 209 µs for the same
  work), because the memory-bound glue kernels and their HBM round trips
  are gone. `gemm_tc_fp16_fused` is the highest point (~4.8 TFLOP/s): its
  tensor-core FLOPs equal the useful work (no padding waste). GEMMs are
  the one kernel type in this tutorial where "compute-bound" is a
  meaningful target, because they have genuinely high arithmetic intensity
  (`O(M*N*K)` FLOPs over `O(M*K + K*N + M*N)` bytes), unlike the
  statevector kernels' memory-bound-at-this-scale Pauli operations.
- At this tutorial's default small sizes (`d_model=32`, modest batch), the
  tensor-core GEMM may still be far from its own ceiling simply because the
  matrices are small relative to a 16x16x16 tile — re-run with larger
  `--d-model` / `--batch-size` to see the achieved-GFLOP/s point move
  further up the compute roofline as problem size grows.

## Where this leaves the tutorial

This is the tutorial's final stage. The overall arc, visible in the
repo-root `roofline_hierarchical_vqe.png` / `roofline_hierarchical_gqe.png`
(reference copies in `reference_results/`): stage1's kernels sit far below both
rooflines (launch-overhead-bound); stage2 moves them measurably closer to
the memory roofline (fusion + batching removes overhead, revealing the
underlying memory-bound-ness); stage3 pushes further toward the memory
roofline (coalescing + shared-memory reuse actually reduces bytes moved);
stage4 either extracts more of the memory roofline via occupancy (VQE) or
crosses over toward the compute roofline entirely via tensor cores (GQE).
That progression — overhead-bound -> memory-bound -> compute-bound, with a
concrete GPU-code change motivating each transition — is the whole point
of this tutorial.
