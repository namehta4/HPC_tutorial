# Reference results

Roofline plots and per-kernel tables from this tutorial's reference run, so
you can compare your own results (or show them in a live session without
profiling first). README §10 walks through what they show.

**How they were made:** NVIDIA A100-SXM4-40GB on a dedicated, idle
Perlmutter node (2026-10-22). Every stage's `profile.sh` ran with

```bash
QUBITS=12 D_MODEL=128 BATCH=256 GQE_BATCH=256 ./profile.sh
```

i.e. VQE on a random 12-qubit, 576-term Hamiltonian at depth 3, and GQE
at batch 256, d_model 128. All correctness tests passed for every stage. The
plotted values match the Nsight Compute GUI's roofline charts
(`tools/verify_vs_gui.py`).

| File | What it shows |
|---|---|
| `roofline_hierarchical_vqe.png` | VQE: one aggregate L1 (o) / L2 (s) / HBM (v) point triplet per stage |
| `roofline_hierarchical_gqe.png` | Same for GQE |
| `per_stage/<report>_all.png` | Each report's top kernels at L1/L2/HBM |
| `per_stage/<report>_kernels.csv` | The numbers behind each plot: per kernel, launches, time (s), FLOPs, bytes per level, arithmetic intensity, GFLOP/s |

Your absolute numbers will differ on other GPUs (and somewhat from run to
run). Compare the *shape*, meaning which way each stage moves and why, not
the digits.
