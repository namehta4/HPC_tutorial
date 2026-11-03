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
| `profiles/<report>_a100_pm.ncu-rep.zip` | The raw Nsight Compute reports behind every plot above (10: 6 VQE, 4 GQE) |
| `profiles/<report>_a100_pm.nsys-rep.zip` | The matching Nsight Systems timelines (11, including stage4's multi-stream GQE run) |

## Opening the raw reports

The reports are zipped because ncu reports compress ~10-150x (the largest
is 184 MB unzipped, 12 MB zipped), which keeps them under GitHub's file
size limit. You don't need a GPU to look at them: download a zip, extract
it, and open the file in the free desktop GUIs on any laptop (Windows,
macOS or Linux):

- `*.ncu-rep` → **Nsight Compute** (File → Open). The *Details* page has
  the Speed Of Light and hierarchical roofline charts; pick a kernel from
  the *Result* dropdown at the top.
- `*.nsys-rep` → **Nsight Systems** (File → Open). The timeline shows the
  per-gate launches and host syncs that dominate stage1, and how stage2+
  remove them.

Or from a terminal with the CLI tools:

```bash
unzip reference_results/profiles/stage4_vqe_roofline_a100_pm.ncu-rep.zip
ncu -i stage4_vqe_roofline_a100_pm.ncu-rep --print-summary per-kernel
python3 plot_hierarchical_roofline.py stage4_vqe_roofline_a100_pm.ncu-rep   # same plots as per_stage/
nsys stats stage4_vqe_timeline_a100_pm.nsys-rep
```

Open the reports with Nsight Compute / Nsight Systems 2026.x (CUDA 13.2)
or newer, since older GUIs may refuse newer report versions. Stage1's ncu
reports cover only the first 60 kernel launches (`--launch-count 60` in
its `profile.sh`).

Your absolute numbers will differ on other GPUs (and somewhat from run to
run). Compare the *shape*, meaning which way each stage moves and why, not
the digits.
