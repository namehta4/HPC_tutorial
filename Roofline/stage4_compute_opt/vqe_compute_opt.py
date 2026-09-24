#!/usr/bin/env python3
"""Stage 4: occupancy-tuned, batched VQE energy evaluation.

Run ./build.sh first.

Stage 3's `apply_ansatz_shared` and `expval_pauli_batched_shared` kernels
each launch exactly ONE thread block -- meaning a single VQE energy
evaluation occupies at most 1 of the A100's 108 Streaming Multiprocessors
(SMs). That's a legitimate design for evaluating ONE circuit as fast as
possible, but real VQE workloads rarely need just one evaluation: a
parameter-shift-rule gradient needs 2 evaluations per parameter (so
`2 * n_params` evaluations per optimizer step), and GQE's sampling loop
proposes many candidate ansatz parameter sets per training step.

This module batches many independent statevector evaluations across the
grid: `apply_ansatz_batched` and `expval_pauli_batched_grid` (see
kernels/gate_apply_tc.cu) launch ONE BLOCK PER STATEVECTOR, so a batch of
B evaluations can occupy up to B SMs concurrently instead of running B
single-SM kernels back-to-back. See NOTES.md for the "why not tensor
cores here" discussion -- 2x2 single-qubit gates are the wrong shape for
tensor cores; the compute-bound-tuning story for VQE is occupancy, not
WMMA.

Tracked kernels: 'gate_apply' -> apply_ansatz_batched,
                 'expval_reduce' -> expval_pauli_batched_grid
(same logical roles as every previous stage, now grid-batched).
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


def _build_fused_matrices_batch(n_qubits: int, depth: int, params_batch: np.ndarray) -> np.ndarray:
    """params_batch: (batch_size, n_params) -> (batch_size, depth*n_qubits, 4) complex128."""
    batch_size = params_batch.shape[0]
    mats = np.zeros((batch_size, depth * n_qubits, 4), dtype=np.complex128)
    for b in range(batch_size):
        idx = 0
        params = params_batch[b]
        for layer in range(depth):
            ry_angles = params[idx : idx + n_qubits]
            idx += n_qubits
            rz_angles = params[idx : idx + n_qubits]
            idx += n_qubits
            for q in range(n_qubits):
                mats[b, layer * n_qubits + q] = _fused_rotation_matrix(
                    float(ry_angles[q]), float(rz_angles[q])
                ).flatten()
    return mats


def _pauli_masks(n_qubits: int, pauli_string: str) -> tuple[int, int, int]:
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


def run_circuits_batched(n_qubits: int, depth: int, params_batch: np.ndarray) -> cp.ndarray:
    """params_batch: (batch_size, n_params). Returns (batch_size, dim) statevectors.
    ONE kernel launch computes the ENTIRE batch, one CUDA block per statevector."""
    mod = _get_module("gate_apply_tc.cubin")
    kernel = mod.get_function("apply_ansatz_batched")

    batch_size = params_batch.shape[0]
    dim = 1 << n_qubits

    psi_batch = cp.zeros((batch_size, dim), dtype=cp.complex128)
    psi_batch[:, 0] = 1.0

    fused_matrices = cp.asarray(_build_fused_matrices_batch(n_qubits, depth, params_batch))

    shared_bytes = dim * 16
    if shared_bytes > 48 * 1024:
        kernel.max_dynamic_shared_size_bytes = shared_bytes

    nthreads = min(512, max(32, dim // 2))
    kernel((batch_size,), (nthreads,), (psi_batch, fused_matrices, n_qubits, depth), shared_mem=shared_bytes)
    return psi_batch


def batched_expectation_grid(psi_batch: cp.ndarray, n_qubits: int, terms: list[tuple[float, str]]) -> cp.ndarray:
    """psi_batch: (batch_size, dim). Returns (batch_size, n_terms) expectation values,
    ONE kernel launch for the entire batch x term matrix."""
    mod = _get_module("gate_apply_tc.cubin")
    kernel = mod.get_function("expval_pauli_batched_grid")

    batch_size, dim = psi_batch.shape
    n_terms = len(terms)

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
    out = cp.zeros((batch_size, n_terms), dtype=cp.float64)

    nthreads = min(512, max(32, dim))
    shared_bytes = dim * 16 + nthreads * 8
    if shared_bytes > 48 * 1024:
        kernel.max_dynamic_shared_size_bytes = shared_bytes

    kernel(
        (batch_size,), (nthreads,),
        (psi_batch, flip_masks_gpu, z_masks_gpu, y_counts_gpu, out, n_qubits, n_terms),
        shared_mem=shared_bytes,
    )
    return out


def vqe_energies_batched(
    n_qubits: int, depth: int, params_batch: np.ndarray, terms: list[tuple[float, str]]
) -> np.ndarray:
    """The full batched VQE pipeline: 2 kernel launches total, regardless of batch size."""
    psi_batch = run_circuits_batched(n_qubits, depth, params_batch)
    exp_vals = batched_expectation_grid(psi_batch, n_qubits, terms)  # (batch_size, n_terms)
    coeffs = cp.asarray([c for c, _ in terms], dtype=cp.float64)
    energies = cp.sum(exp_vals * coeffs[None, :], axis=1)
    return cp.asnumpy(energies)


def main():
    ap = argparse.ArgumentParser(description="Stage 4 occupancy-tuned batched VQE energy evaluation.")
    add_vqe_args(ap)
    ap.add_argument("--batch-size", type=int, default=16,
                     help="Number of independent parameter sets to evaluate concurrently "
                     "(models e.g. parameter-shift-rule gradient evaluations).")
    args = ap.parse_args()

    n_qubits = resolve_qubits(args.molecule, args.qubits)
    n_qubits, terms = get_terms(args.molecule, n_qubits, args.seed)

    from common.ansatz import n_params, random_params

    base_params = random_params(n_qubits, args.depth, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    params_batch = np.stack(
        [base_params + rng.normal(scale=0.05, size=base_params.shape) for _ in range(args.batch_size)]
    )

    t0 = time.perf_counter()
    energies = vqe_energies_batched(n_qubits, args.depth, params_batch, terms)
    t1 = time.perf_counter()

    print(f"[stage4_compute_opt] molecule={args.molecule} n_qubits={n_qubits} depth={args.depth} "
          f"batch_size={args.batch_size}")
    print(f"[stage4_compute_opt] energies[:5] = {energies[:5]}")
    print(f"[stage4_compute_opt] wall time = {(t1 - t0) * 1e3:.3f} ms "
          f"({(t1 - t0) * 1e3 / args.batch_size:.4f} ms/evaluation)")


if __name__ == "__main__":
    main()
