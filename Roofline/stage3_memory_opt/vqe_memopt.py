#!/usr/bin/env python3
"""Stage 3: memory-traffic-optimized VQE, using hand-written CUDA kernels.

Run ./build.sh first to compile kernels/*.cu into kernels/*.cubin.

This replaces stage2's sequence of many small CuPy-generated kernels (one
per gate, one CuPy-vectorized op for the batched expectation value) with
TWO hand-written CUDA kernels total:

  1. `apply_ansatz_shared` (kernels/gate_apply.cu) -- the entire circuit,
     every gate, in ONE kernel launch. The statevector is loaded into
     shared memory once, every gate updates it in place in shared memory,
     and the result is written back to global memory once. See that file's
     header comment for the coalescing/reuse argument.

  2. `expval_pauli_batched_shared` (kernels/expval_reduce.cu) -- every
     Hamiltonian term's expectation value, in ONE kernel launch, with the
     statevector again loaded into shared memory once and reused across
     all `n_terms` terms instead of stage2's one-global-memory-pass-per-term.

Net effect: for a full VQE energy evaluation, global memory traffic on the
2^n-element statevector drops from O(n_gates + n_terms) full passes
(stage2) to exactly 3 full passes total (load once for gate application,
write back once, then load once for expectation-value reduction) --
independent of circuit depth or Hamiltonian term count.

Mixed precision: `--precision fp32` runs the SAME kernels compiled against
complex64 arithmetic (see kernels build variants below), halving the bytes
moved per statevector element. This tutorial's correctness tests verify the
resulting VQE energy still matches the FP64 CPU reference within a looser
(but still meaningful) tolerance -- mixed precision is a real memory-traffic
optimization, not a free lunch, and the tests say so explicitly.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cupy as cp
import numpy as np

from common.ansatz import Gate, single_qubit_matrix
from common.config import add_vqe_args, get_terms, resolve_qubits

KERNELS_DIR = Path(__file__).resolve().parent / "kernels"

_MODULE_CACHE: dict[str, cp.RawModule] = {}


def _get_module(cubin_name: str) -> cp.RawModule:
    if cubin_name not in _MODULE_CACHE:
        path = KERNELS_DIR / cubin_name
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run ./build.sh first")
        _MODULE_CACHE[cubin_name] = cp.RawModule(path=str(path))
    return _MODULE_CACHE[cubin_name]


def _fused_rotation_matrix(theta: float, phi: float) -> np.ndarray:
    ry = single_qubit_matrix(Gate("RY", (0,), theta))
    rz = single_qubit_matrix(Gate("RZ", (0,), phi))
    return rz @ ry


def _build_fused_matrices(n_qubits: int, depth: int, params: np.ndarray) -> np.ndarray:
    """(depth * n_qubits, 4) complex128 array of flattened 2x2 fused RZ@RY matrices."""
    mats = np.zeros((depth * n_qubits, 4), dtype=np.complex128)
    idx = 0
    for layer in range(depth):
        ry_angles = params[idx : idx + n_qubits]
        idx += n_qubits
        rz_angles = params[idx : idx + n_qubits]
        idx += n_qubits
        for q in range(n_qubits):
            mats[layer * n_qubits + q] = _fused_rotation_matrix(float(ry_angles[q]), float(rz_angles[q])).flatten()
    return mats


def run_circuit_kernel(n_qubits: int, depth: int, params: np.ndarray, precision: str = "fp64") -> cp.ndarray:
    """Apply the whole ansatz via ONE launch of apply_ansatz_shared (or its
    fp32 variant). `precision='fp32'` halves global-memory bytes moved per
    amplitude at the cost of numerical precision -- see NOTES.md."""
    dim = 1 << n_qubits

    if precision == "fp32":
        mod = _get_module("gate_apply_fp32.cubin")
        kernel = mod.get_function("apply_ansatz_shared_fp32")
        dtype = cp.complex64
        bytes_per_amp = 8
    else:
        mod = _get_module("gate_apply.cubin")
        kernel = mod.get_function("apply_ansatz_shared")
        dtype = cp.complex128
        bytes_per_amp = 16

    psi = cp.zeros(dim, dtype=dtype)
    psi[0] = 1.0

    fused_matrices_np = _build_fused_matrices(n_qubits, depth, params)
    if precision == "fp32":
        fused_matrices_np = fused_matrices_np.astype(np.complex64)
    fused_matrices = cp.asarray(fused_matrices_np)

    shared_bytes = dim * bytes_per_amp
    if shared_bytes > 48 * 1024:
        # Opt in to >48KB dynamic shared memory (A100 allows up to ~163KB/block).
        kernel.max_dynamic_shared_size_bytes = shared_bytes

    nthreads = min(512, max(32, dim // 2))
    kernel((1,), (nthreads,), (psi, fused_matrices, n_qubits, depth), shared_mem=shared_bytes)
    return psi


def _pauli_masks(n_qubits: int, pauli_string: str) -> tuple[int, int, int]:
    """flip_mask, z_mask, y_count for one Pauli string (see kernels/expval_reduce.cu)."""
    flip_mask = 0
    z_mask = 0
    y_count = 0
    for q, c in enumerate(pauli_string):
        bit = n_qubits - 1 - q
        if c == "X":
            flip_mask |= 1 << bit
        elif c == "Y":
            flip_mask |= 1 << bit
            z_mask |= 1 << bit
            y_count += 1
        elif c == "Z":
            z_mask |= 1 << bit
    return flip_mask, z_mask, y_count


def batched_expectation_kernel(
    psi: cp.ndarray, n_qubits: int, terms: list[tuple[float, str]], precision: str = "fp64"
) -> cp.ndarray:
    """Every Hamiltonian term's expectation value via ONE launch of
    expval_pauli_batched_shared (or its fp32 variant), reusing the
    shared-memory-resident statevector across all terms."""
    if precision == "fp32":
        mod = _get_module("expval_reduce_fp32.cubin")
        kernel = mod.get_function("expval_pauli_batched_shared_fp32")
        bytes_per_amp = 8
    else:
        mod = _get_module("expval_reduce.cubin")
        kernel = mod.get_function("expval_pauli_batched_shared")
        bytes_per_amp = 16

    n_terms = len(terms)
    dim = 1 << n_qubits
    flip_masks = np.zeros(n_terms, dtype=np.int64)
    z_masks = np.zeros(n_terms, dtype=np.int64)
    y_counts = np.zeros(n_terms, dtype=np.int32)
    for t, (_, pauli) in enumerate(terms):
        fm, zm, yc = _pauli_masks(n_qubits, pauli)
        flip_masks[t] = fm
        z_masks[t] = zm
        y_counts[t] = yc

    flip_masks_gpu = cp.asarray(flip_masks)
    z_masks_gpu = cp.asarray(z_masks)
    y_counts_gpu = cp.asarray(y_counts)
    out = cp.zeros(n_terms, dtype=cp.float64)  # accumulation always in fp64, see kernel comments

    nthreads = min(512, max(32, dim))
    shared_bytes = dim * bytes_per_amp + nthreads * 8
    if shared_bytes > 48 * 1024:
        kernel.max_dynamic_shared_size_bytes = shared_bytes

    kernel(
        (1,), (nthreads,),
        (psi, flip_masks_gpu, z_masks_gpu, y_counts_gpu, out, n_qubits, n_terms),
        shared_mem=shared_bytes,
    )
    return out


def vqe_energy_memopt(
    n_qubits: int, depth: int, params: np.ndarray, terms: list[tuple[float, str]], precision: str = "fp64"
) -> float:
    psi = run_circuit_kernel(n_qubits, depth, params, precision=precision)
    exp_vals = batched_expectation_kernel(psi, n_qubits, terms, precision=precision)
    coeffs = cp.asarray([c for c, _ in terms], dtype=cp.float64)
    energy = cp.sum(coeffs * exp_vals)
    return float(energy.get())


def main():
    ap = argparse.ArgumentParser(description="Stage 3 memory-optimized hand-written-CUDA VQE energy evaluation.")
    add_vqe_args(ap)
    args = ap.parse_args()

    precision = "fp32" if args.precision in ("fp32", "tf32") else "fp64"
    if args.precision == "tf32":
        print("[stage3_memopt] NOTE: --precision tf32 requested; this statevector kernel has no "
              "tensor-core matmul to apply TF32 to, so it runs the fp32 (complex64) kernel variant "
              "instead. TF32 tensor cores show up in stage4's GEMM-shaped kernels.")
    elif args.precision == "fp16":
        print("[stage3_memopt] NOTE: --precision fp16 requested; this stage only ships fp64/fp32 "
              "kernel variants (fp16 statevector storage is numerically too lossy for VQE energies "
              "at the precision this tutorial checks against). Running fp32 instead.")

    n_qubits = resolve_qubits(args.molecule, args.qubits)
    n_qubits, terms = get_terms(args.molecule, n_qubits, args.seed)

    from common.ansatz import random_params

    params = random_params(n_qubits, args.depth, seed=args.seed)

    t0 = time.perf_counter()
    energy = vqe_energy_memopt(n_qubits, args.depth, params, terms, precision=precision)
    t1 = time.perf_counter()

    print(f"[stage3_memopt] molecule={args.molecule} n_qubits={n_qubits} depth={args.depth}")
    print(f"[stage3_memopt] energy = {energy:.10f} Ha")
    print(f"[stage3_memopt] wall time = {(t1 - t0) * 1e3:.3f} ms")


if __name__ == "__main__":
    main()
