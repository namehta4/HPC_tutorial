# HPC_tutorial

A hands-on tutorial in parallel and heterogeneous programming for HPC systems,
using a simple matrix-vector multiply as the running example. It progresses
from single-threaded C through OpenMP (CPU threading and GPU offload), MPI
(distributed memory), hybrid MPI+OpenMP-offload, and CUDA — including a set of
CUDA optimization steps (loop unrolling, mixed precision, shared-memory
tiling).

## Layout

### `HelloWorld/`
Minimal "hello world" programs for each parallel model, useful as compile/run
smoke tests on a new system:

| File | Model |
|---|---|
| `HelloWorld.c` | Serial C |
| `HelloWorld_omp.c` | OpenMP (CPU threads) |
| `HelloWorld_mpi.c` | MPI (rank/hostname report) |
| `HelloWorld_p2p.c` | MPI point-to-point (`MPI_Send`/`MPI_Recv`) |
| `HelloWorld.cu` | CUDA kernel launch |

### `MatVec/`
The same dense matrix-vector multiply (`a = W*x + b`, with `n=80000` rows and
`m=500` columns run for 100 iterations), implemented with different
parallelization strategies so their performance and code structure can be
compared directly:

| File | Model |
|---|---|
| `matvec.c` | Serial baseline |
| `matvec_omp.c` | OpenMP parallel-for on the CPU |
| `matvec_ompoff.c` | OpenMP target offload to GPU |
| `matvec_mpi.c` | MPI: row-wise scatter/gather across ranks |
| `matvec_hybrid.c` | MPI + OpenMP target offload (each rank offloads its row block to a GPU) |
| `matvec.cu` | CUDA: one thread per output row |

### `Optimizations/`
A sequence of CUDA matrix-vector kernels (`n=80000`, `m=8000`) showing
successive optimization steps, plus `run.sh` to build and profile each one
with `nsys`:

| File | Optimization |
|---|---|
| `matvec_opt0.cu` | Baseline CUDA kernel (double precision, one thread per row) |
| `matvec_opt1.cu` | + loop unrolling (`#pragma unroll 4`) for instruction-level parallelism |
| `matvec_opt2.cu` | + switch to single precision (`float`) |
| `matvec_opt3.cu` | + shared-memory tiling of the input vector `x` |

## Building and running

Each source file is self-contained and can be compiled directly with the
appropriate toolchain (adjust flags/modules for your system):

```bash
# Serial / OpenMP CPU
cc -O3 -fopenmp MatVec/matvec_omp.c -o matvec_omp

# OpenMP GPU offload (e.g. NVIDIA target via nvc/clang)
nvc -mp=gpu -O3 MatVec/matvec_ompoff.c -o matvec_ompoff

# MPI
cc MatVec/matvec_mpi.c -o matvec_mpi
srun -n <ranks> ./matvec_mpi

# Hybrid MPI + OpenMP offload
cc -mp=gpu MatVec/matvec_hybrid.c -o matvec_hybrid
srun -n <ranks> ./matvec_hybrid

# CUDA
nvcc -O3 MatVec/matvec.cu -o matvec_cuda
./matvec_cuda
```

For the `Optimizations/` kernels, run `Optimizations/run.sh` (via `srun` on a
Slurm-managed GPU node) to build each stage and profile it with `nsys`:

```bash
cd Optimizations
./run.sh
```

### `Roofline/`
A separate, more advanced tutorial (its own git repository) that teaches
roofline analysis — reading GPU performance against memory-bandwidth and
compute (FLOP/s) ceilings — using two quantum-computing workloads (a VQE
statevector simulator and a GQE transformer policy network) as the example
codes instead of matrix-vector multiply. It walks through the same
workloads across five stages, each fixing the bottleneck the previous
stage's profile revealed:

| Stage | Bottleneck fixed | Technique |
|---|---|---|
| `stage0_calibration` | n/a | measure this GPU's real achievable bandwidth/FLOP-s ceilings |
| `stage1_naive` | — (baseline) | a legitimate, unoptimized CuPy port |
| `stage2_gpu_native` | kernel-launch & sync overhead | fuse gates, batch work, remove syncs, use streams |
| `stage3_memory_opt` | global memory traffic | hand-written CUDA kernels, shared memory, mixed precision |
| `stage4_compute_opt` | occupancy / compute throughput | batch across SMs, tensor cores for the GEMM-shaped GQE |

See `Roofline/README.md` for the full participant guide (environment setup,
per-stage run commands, expected results) and `Roofline/ENV_SETUP.md` for
detailed environment setup on NERSC Perlmutter and generic Linux+GPU
machines.
