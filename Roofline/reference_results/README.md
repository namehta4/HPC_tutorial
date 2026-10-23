# Reference results

Roofline plots and per-kernel tables from this tutorial's reference run, so
you can compare your own results (or show them in a live session without
profiling first). README §10 walks through what they show.

**How they were made:** NVIDIA A100-SXM4-40GB on dedicated, idle
Perlmutter nodes (2026-10-22; stages 3 and 4 re-profiled 2026-10-23 after
the GQE tensor-core and fusion changes). Every stage's `profile.sh` ran with

```bash
QUBITS=12 D_MODEL=128 BATCH=256 GQE_BATCH=256 ./profile.sh
```

i.e. VQE on a random 12-qubit, 576-term Hamiltonian at depth 3, and GQE
at batch 256, d_model 128. All correctness tests passed for every stage. The
plotted values match the Nsight Compute GUI's roofline charts
(`tools/verify_vs_gui.py`).

Every file name ends in `_a100_pm` (A100 on Perlmutter), so it can't be
mistaken for, or replaced by, the untagged plots `profile.sh` writes when
you run the tutorial on your own system.

| File | What it shows |
|---|---|
| `roofline_hierarchical_vqe_a100_pm.png` | VQE: one aggregate L1 (o) / L2 (s) / HBM (v) point triplet per stage |
| `roofline_hierarchical_gqe_a100_pm.png` | Same for GQE |
| `story/roofline_story_{vqe,gqe}_<n>_a100_pm.png` | Slide sequence: stages 1..n at HBM only, with all ceilings, identical axes, and speedup arrows |
| `per_stage/<report>_all_a100_pm.png` | Each report's top kernels at L1/L2/HBM |
| `per_stage/<report>_kernels_a100_pm.csv` | The numbers behind each plot: per kernel, launches, time (s), FLOPs, bytes per level, arithmetic intensity, GFLOP/s |

Your absolute numbers will differ on other GPUs (and somewhat from run to
run). Compare the *shape*, meaning which way each stage moves and why, not
the digits.
