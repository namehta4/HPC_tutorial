# Environment Setup

This tutorial was built and tested on **NERSC Perlmutter**
(NVIDIA A100-PCIE-40GB, compute capability 8.0, CUDA 13.2, driver
580.178.04). The instructions below are split into "Perlmutter exact steps"
(copy-paste, known to work) and "generic instructions" (any other machine
with an NVIDIA GPU).

If you're new to HPC clusters or CUDA development, read the whole page —
it's written to be followed step by step, with no assumed prior GPU
experience.

## 1. Do you have what you need?

Check these three things before doing anything else:

```bash
nvidia-smi                  # Do you have an NVIDIA GPU? What model? How busy is it?
nvcc --version               # Is the CUDA compiler available?
nsys --version && ncu --version   # Are the profiling tools available?
```

If any of these commands say "command not found," you need to load a
module (on an HPC cluster) or install the CUDA Toolkit (on your own
machine) before continuing — see "Generic instructions" below.

**Look at `nvidia-smi`'s "Volatile GPU-Util" column.** If it already shows
high utilization from someone else's job, every timing number and roofline
plot you generate will be contaminated by contention — see the "Shared GPU
warning" box below.

## 2. Perlmutter exact steps

```bash
# Load a complete Python environment with a working `pip --user`.
# (Perlmutter's bare `python3` on the login node has no pip -- this module
# gives you a real Python 3.12 + NumPy/SciPy/PyTorch environment instead.)
module load pytorch/2.13.0

# Install this tutorial's remaining dependencies into your user site-packages.
# cupy-cuda13x / cuquantum-python-cu13 match Perlmutter's CUDA 13.x toolkit --
# if `nvcc --version` shows CUDA 12.x on your allocation instead, swap these
# for cupy-cuda12x / cuquantum-python-cu12.
pip install --user cupy-cuda13x cuquantum-python-cu13 qutip pytest

# Verify the GPU is visible to CuPy.
python3 -c "import cupy as cp; print(cp.cuda.Device(0).compute_capability); print(cp.arange(5).sum())"
```

**Getting a dedicated GPU for profiling and calibration.** Running on a
Perlmutter *login* node's GPU is fine for quick correctness checks, but
login-node GPUs are often shared with other users' jobs (you can check with
`nvidia-smi` — if "Volatile GPU-Util" is already high before you've run
anything, someone else is using it). For calibration (`stage0`) and any
profiling run whose numbers you care about, request a dedicated compute
node:

```bash
salloc -N 1 -C gpu -G 1 -q interactive -t 00:30:00 -A <your_NERSC_project>
```

Replace `<your_NERSC_project>` with your NERSC allocation/repo name (run
`sacctmgr show associations user=$USER format=account` if you're not sure
which ones you have). Once the allocation starts, you'll be on a compute
node with an exclusive GPU — re-run the `module load` / verification
commands above there too (modules loaded on the login node do NOT
automatically carry over).

## 3. Generic instructions (any Linux machine with an NVIDIA GPU)

1. **Install the CUDA Toolkit** (includes `nvcc`) from
   [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads),
   matching your installed NVIDIA driver's supported CUDA version (check
   with `nvidia-smi` — the top-right of its header shows the maximum CUDA
   version your driver supports).
2. **Install Nsight Systems and Nsight Compute.** These usually ship
   bundled with the CUDA Toolkit installer, or can be downloaded separately
   from [developer.nvidia.com/nsight-systems](https://developer.nvidia.com/nsight-systems)
   and [developer.nvidia.com/nsight-compute](https://developer.nvidia.com/nsight-compute).
   Confirm with `nsys --version` and `ncu --version`.
3. **Set up Python** (3.10+; this tutorial was tested on 3.12) and install
   dependencies:
   ```bash
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```
   Pick the `cupy-cudaXXx` / `cuquantum-python-cuXX` variant in
   `requirements.txt` matching your CUDA major version (12 or 13).
4. **Check GPU compute capability** with
   `nvidia-smi --query-gpu=compute_cap --format=csv` — you'll need this
   number (e.g. `8.0`, `8.6`, `8.9`, `9.0`) to pass as `-arch=sm_XX` when
   compiling stage3/stage4's CUDA kernels (`CUDA_ARCH=sm_86 ./build.sh`,
   for example — see each stage's `build.sh`).

## 4. The DCGM / Nsight Compute permission issue (read this before profiling)

On many shared/HPC systems, NVIDIA's DCGM (Data Center GPU Manager)
background service holds the GPU's hardware performance counters, which
makes Nsight Compute (`ncu`) fail with:

```
==ERROR== Profiling failed because a driver resource was unavailable.
Ensure that no other tool (like DCGM) is concurrently collecting profiling
data.
```

We hit this ourselves while building this tutorial. The fix (no special
privileges needed on Perlmutter):

```bash
dcgmi profile --pause     # before profiling
# ... run your ncu / nsys commands ...
dcgmi profile --resume    # after you're done
```

Every `profile.sh` script in this tutorial does this automatically (pause
at the start, resume on exit via a shell trap, even if the script fails
partway through). If `dcgmi` isn't installed on your system, this is a
harmless no-op — but if `ncu` still fails with the error above, check
whether DCGM (or a similar tool, like the OS-level `nvidia-smi dmon`) is
running (`ps aux | grep dcgm`) and consult your cluster's documentation or
your sysadmin for how to pause it.

## 5. Shared GPU warning (why this matters for a *performance* tutorial)

This tutorial teaches you to read roofline plots and compare kernels
against *achievable* hardware ceilings — but if another job is running on
the same physical GPU while you profile, your measured numbers reflect
**contention**, not your code's actual efficiency. We saw this directly
while writing this tutorial: on a login-node GPU already at 100%
utilization from an unrelated job, our measured memory bandwidth was
**~192 GB/s** against this A100's real achievable ceiling of roughly
**1300-1500 GB/s** — an order of magnitude off, purely from sharing the GPU.

Before trusting any timing, profiling, or roofline number from this
tutorial:
1. Run `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv`
   and confirm the GPU is near-idle.
2. Prefer a dedicated allocation (`salloc` on Perlmutter, or simply "a
   machine nobody else is using") over a shared login node or shared
   workstation.
3. Re-run `stage0_calibration/run_calibration.sh` on that dedicated
   allocation before trusting any stage's roofline plot — see
   `stage0_calibration/RESULTS.md`.

## 6. Quick sanity check

Once everything above is done, confirm the whole pipeline works end to end:

```bash
cd stage1_naive
python3 -m pytest test_correctness.py -v
```

You should see 10 tests pass. If they do, you're ready to start the
tutorial from the top-level `README.md`.
