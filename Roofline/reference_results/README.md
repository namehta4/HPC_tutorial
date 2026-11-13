# Reference results

Roofline plots and per-kernel tables from this tutorial's reference runs, so
you can compare your own results (or show them in a live session without
profiling first). README §10 walks through what they show. There are two
sets, one per GPU, told apart by the tag at the end of every file name:
`_a100_pm` (A100 on Perlmutter) and `_gb300` (GB300).

**How the A100 set was made:** NVIDIA A100-SXM4-40GB on dedicated, idle
Perlmutter nodes (2026-10-22; stages 3 and 4 re-profiled 2026-10-23 after
the GQE tensor-core and fusion changes). Every stage's `profile.sh` ran with

```bash
QUBITS=12 D_MODEL=128 BATCH=256 GQE_BATCH=256 ./profile.sh
```

i.e. VQE on a random 12-qubit, 576-term Hamiltonian at depth 3, and GQE
at batch 256, d_model 128. All correctness tests passed for every stage. The
plotted values match the Nsight Compute GUI's roofline charts
(`tools/verify_vs_gui.py`). GQE stages 1–3 were profiled before the fix that
keeps them in FP32 under NumPy 2 (`float()` around the `np.sqrt` scalars), so
FP64 kernels take 66%, 77% and 13% of their GPU time (kernels with
double/float64 in their names, e.g. cuBLAS `gemmSN_*_kernel<double>`, cub
reductions on double and `cupy_*_float64` elementwise kernels). Re-profiled
on an A100 today, those stages would show
FP32 kernels with different FLOP/s and GPU time.

**How the GB300 set was made:** one NVIDIA GB300 GPU (compute capability
10.3, 152 SMs) on a dedicated, idle node (2026-11-13; driver 580.159.03,
CUDA 13.0, Nsight Compute 2025.3.1, Nsight Systems 2025.3.2). One
`./run_all.sh` from a clean checkout, with

```bash
QUBITS=12 D_MODEL=128 BATCH=4096 GQE_BATCH=4096 VQE_BATCH=152 ./run_all.sh
```

The batches are larger than the A100 run's so the work fills the bigger
GPU, and `VQE_BATCH=152` gives stage 4 one statevector per SM. All
correctness tests passed, and `tools/verify_vs_gui.py` found a largest
relative difference of 6.7e-05 between the plots and Nsight Compute's
roofline-chart formulas as of version 2026.1, which count packed FP32. These
reports carry Nsight Compute 2025.3's charts, which don't (see *Packed FP32
and padding* below). `environment_manifest_gb300.txt` lists every version.

Every file name ends in its system tag, so it can't be mistaken for, or
replaced by, the untagged plots `profile.sh` writes when you run the
tutorial on your own system.

| File | What it shows |
|---|---|
| `roofline_hierarchical_vqe_a100_pm.png` | VQE: one aggregate L1 (o) / L2 (s) / HBM (v) point triplet per stage |
| `roofline_hierarchical_gqe_a100_pm.png` | Same for GQE |
| `story/roofline_story_{vqe,gqe}_<n>_a100_pm.png` | Slide sequence: stages 1..n at HBM only, with all ceilings, identical axes, and speedup arrows |
| `per_stage/<report>_all_a100_pm.png` | Each report's top kernels at L1/L2/HBM |
| `per_stage/<report>_kernels_a100_pm.csv` | The numbers behind each plot: per kernel, launches, time (s), FLOPs, bytes per level, arithmetic intensity, GFLOP/s |
| `profiles/<report>_a100_pm.ncu-rep.zip` | The raw Nsight Compute reports behind every plot above (10: 6 VQE, 4 GQE) |
| `profiles/<report>_a100_pm.nsys-rep.zip` | The matching Nsight Systems timelines (11, including stage4's multi-stream GQE run) |

The GB300 set has the same files with `_gb300` in place of `_a100_pm`.

## Reading the GB300 set

- **Clock.** Nsight Compute locks the clocks while it profiles. On GB300
  the SM clock it reports is about 1.25–1.35 GHz, against a 2.07 GHz
  maximum, so the ceilings in the plots vary by a few percent between
  reports. The FP64 CUDA-core roof is about 0.77 TFLOP/s under ncu, and
  about 1.26 TFLOP/s at full clock.
- **One SM.** VQE stage 3 runs each kernel as one thread block, so it can
  use at most one of the 152 SMs. Its real limit is one SM's share of the
  FP64 roof, about 5 GFLOP/s, not the roofs drawn on the plot. That is why
  stage 3 is slower than stage 2 on GB300, and why stage 4 (one block per
  statevector, 152 statevectors) is the step that pays.
- **FP16 tensor roof.** The plots draw the dense FP16 tensor-core peak,
  about 1.57 PFLOP/s under ncu. The un-split FP16 peak that ncu also
  reports on GB300 is twice that, because it counts 2:4 structured
  sparsity. Stage 4's hand-written WMMA kernel uses the older `mma.sync`
  path, which tops out at a quarter of the dense peak (about 390 TFLOP/s).
- **Packed FP32 and padding.** cuBLAS's SGEMM kernels on GB300 issue packed
  FP32 instructions (FFMA2 = two FMAs). `profile.sh` collects their counters,
  and the plots count them as Nsight Compute 2026 does, by the work each
  instruction performs. GQE stage 2's linear layers multiply 8-row operands,
  which cuBLAS pads to 32-row tiles, so about three quarters of their
  counted FP32 FLOPs are padding. That inflates stage 2's GFLOP/s, which is
  why the stage 2→3 story arrow shows a drop in throughput although stage 3
  takes less GPU time (README §10 explains why a throughput ratio misleads
  when FLOPs aren't all useful work). The Nsight Compute 2025.3 GUI's
  roofline chart leaves packed FP32 out entirely, so for these kernels it
  shows far less FP32 throughput than the plots.

## Opening the raw reports

The reports are zipped because ncu reports compress ~10-150x (the largest
is 191 MB unzipped, 13 MB zipped), which keeps them under GitHub's file
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

Open the A100 reports with Nsight Compute / Nsight Systems 2026.x
(CUDA 13.2) or newer, and the GB300 reports with Nsight Compute 2025.3 /
Nsight Systems 2025.3.2.474 or newer (2025.5 or newer is simplest; the
2025.3.2.367 build bundled with CUDA 13.0 refuses the GB300 timelines),
since older GUIs may refuse newer report versions. Stage1's ncu
reports cover only the first 60 kernel launches (`--launch-count 60` in
its `profile.sh`).

Your absolute numbers will differ on other GPUs (and somewhat from run to
run). Compare the *shape*, meaning which way each stage moves and why, not
the digits.
