#!/usr/bin/env python3
"""Stage 2 comparison point: the same VQE circuit, run with NVIDIA's
cuQuantum cuStateVec library instead of hand-rolled CuPy kernels.

This file exists to answer the question every reader will have after seeing
vqe_fused.py: "why would I ever hand-write this instead of using NVIDIA's
own library?" The answer this tutorial gives is pedagogical, not
practical -- for production VQE work, cuStateVec is almost always the right
choice (it's more general, handles more qubits, and is tuned by NVIDIA
engineers who know the hardware intimately). We hand-roll our own kernels
in stages 2-4 specifically so the roofline STORY is visible: you can watch
memory traffic and compute intensity change as we rewrite our own code, in
a way you can't when the implementation is opaque library internals.

cuStateVec conventions to note (verified empirically, see the tutorial's
build notes): its state-vector bit index 0 is the LEAST significant bit,
which is the OPPOSITE of this tutorial's big-endian convention (qubit 0 =
most significant bit, used throughout common/). So every target/basis bit
index passed to a custatevec call is qubit index (n_qubits - 1 - q).

Two cuStateVec API calls do the heavy lifting:
  - cusv.apply_matrix(...)                     -- the 'gate_apply' kernel
  - cusv.compute_expectations_on_pauli_basis(...) -- the 'expval_reduce' kernel,
    natively batched across all Hamiltonian terms in one call.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cupy as cp
import numpy as np
import cuquantum
from cuquantum.bindings import custatevec as cusv

from common.ansatz import build_circuit
from common.config import add_vqe_args, get_terms, resolve_qubits

_PAULI_CHAR_TO_ENUM = {"I": cusv.Pauli.I, "X": cusv.Pauli.X, "Y": cusv.Pauli.Y, "Z": cusv.Pauli.Z}


def to_custatevec_bit(n_qubits: int, qubit: int) -> int:
    """Map this tutorial's qubit index (0 = MSB) to cuStateVec's bit index (0 = LSB)."""
    return n_qubits - 1 - qubit


def run_circuit_custatevec(handle, sv: cp.ndarray, n_qubits: int, depth: int, params: np.ndarray) -> None:
    """Apply every ansatz gate via cusv.apply_matrix, in place on `sv`."""
    for gate in build_circuit(n_qubits, depth, params):
        if gate.name == "CNOT":
            control, target = gate.qubits
            X = cp.array([[0, 1], [1, 0]], dtype=cp.complex128)
            cusv.apply_matrix(
                handle, sv.data.ptr, cuquantum.cudaDataType.CUDA_C_64F, n_qubits,
                X.data.ptr, cuquantum.cudaDataType.CUDA_C_64F, cusv.MatrixLayout.ROW, 0,
                [to_custatevec_bit(n_qubits, target)], 1,
                [to_custatevec_bit(n_qubits, control)], [1], 1,
                cusv.ComputeType.COMPUTE_64F, 0, 0,
            )
        else:
            from common.ansatz import single_qubit_matrix

            mat = cp.asarray(single_qubit_matrix(gate))
            (qubit,) = gate.qubits
            cusv.apply_matrix(
                handle, sv.data.ptr, cuquantum.cudaDataType.CUDA_C_64F, n_qubits,
                mat.data.ptr, cuquantum.cudaDataType.CUDA_C_64F, cusv.MatrixLayout.ROW, 0,
                [to_custatevec_bit(n_qubits, qubit)], 1, [], [], 0,
                cusv.ComputeType.COMPUTE_64F, 0, 0,
            )


def batched_pauli_expectation_custatevec(
    handle, sv: cp.ndarray, n_qubits: int, terms: list[tuple[float, str]]
) -> np.ndarray:
    """One call to cusv.compute_expectations_on_pauli_basis computes every
    term's expectation value in a single batched library call -- the
    natively-supported equivalent of stage2's hand-rolled
    batched_pauli_expectation in vqe_fused.py."""
    pauli_ops = []
    basis_bits = []
    for _, pauli_string in terms:
        ops = []
        bits = []
        for q, c in enumerate(pauli_string):
            if c == "I":
                continue
            ops.append(_PAULI_CHAR_TO_ENUM[c])
            bits.append(to_custatevec_bit(n_qubits, q))
        if not ops:  # identity term: no basis bits to specify
            ops = [cusv.Pauli.I]
            bits = [0]
        pauli_ops.append(ops)
        basis_bits.append(bits)

    n_basis_bits = [len(b) for b in basis_bits]
    exp_vals = np.zeros(len(terms), dtype=np.float64)

    cusv.compute_expectations_on_pauli_basis(
        handle, sv.data.ptr, cuquantum.cudaDataType.CUDA_C_64F, n_qubits,
        exp_vals.ctypes.data, pauli_ops, len(terms), basis_bits, n_basis_bits,
    )
    return exp_vals


def vqe_energy_custatevec(n_qubits: int, depth: int, params: np.ndarray, terms: list[tuple[float, str]]) -> float:
    handle = cusv.create()
    try:
        sv = cp.zeros(2**n_qubits, dtype=cp.complex128)
        sv[0] = 1.0

        run_circuit_custatevec(handle, sv, n_qubits, depth, params)
        exp_vals = batched_pauli_expectation_custatevec(handle, sv, n_qubits, terms)

        coeffs = np.array([c for c, _ in terms], dtype=np.float64)
        return float(np.sum(coeffs * exp_vals))
    finally:
        cusv.destroy(handle)


def main():
    ap = argparse.ArgumentParser(description="Stage 2 cuStateVec-based VQE energy evaluation (comparison point).")
    add_vqe_args(ap)
    args = ap.parse_args()

    n_qubits = resolve_qubits(args.molecule, args.qubits)
    n_qubits, terms = get_terms(args.molecule, n_qubits, args.seed)

    from common.ansatz import random_params

    params = random_params(n_qubits, args.depth, seed=args.seed)

    t0 = time.perf_counter()
    energy = vqe_energy_custatevec(n_qubits, args.depth, params, terms)
    t1 = time.perf_counter()

    print(f"[stage2_custatevec] molecule={args.molecule} n_qubits={n_qubits} depth={args.depth}")
    print(f"[stage2_custatevec] energy = {energy:.10f} Ha")
    print(f"[stage2_custatevec] wall time = {(t1 - t0) * 1e3:.3f} ms")


if __name__ == "__main__":
    main()
