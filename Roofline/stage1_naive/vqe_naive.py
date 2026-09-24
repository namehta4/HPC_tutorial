#!/usr/bin/env python3
"""Stage 1: naive GPU port of the statevector VQE simulator.

This is a *legitimate* naive baseline, not a strawman. It is what a
computational chemist who knows NumPy but has never done GPU performance
engineering would actually write on their first attempt at "just move it to
CuPy". Every design choice below is realistic, not deliberately sabotaged:

  1. One CuPy kernel launch PER GATE. A circuit with G gates issues G
     separate tiny GPU kernels (via cp.tensordot / cp.moveaxis), each one
     touching the full 2^n-element statevector. There is no fusion: applying
     RY then RZ to the same qubit reads and writes the whole statevector
     from/to global memory twice instead of once.

  2. `cp.cuda.Stream.null.synchronize()` after every single gate. This
     forces the host to wait for the GPU after every tiny kernel, so the
     CPU can never get ahead and queue up work -- kernel-launch overhead
     and PCIe-driven latency dominate wall-clock time instead of the GPU's
     actual compute/memory work.

  3. Each Hamiltonian term's expectation value is computed by building the
     FULL dense 2^n x 2^n Pauli matrix (via repeated cp.kron) and doing a
     dense matrix-vector product, THEN copying the scalar result back to
     host with `.get()` -- once per term. For H2 (15 terms) that is 15
     independent host syncs and 15 redundant statevector reads, when the
     "same" statevector is being reduced against 15 different operators
     that could obviously be batched.

  4. No mixed precision, no shared memory, no attempt at coalescing --
     complex128 throughout, exactly mirroring common/reference_cpu.py's
     algorithm but executed gate-by-gate, term-by-term on the GPU instead
     of the CPU.

What Nsight Systems will show: a long, thin timeline of many small kernel
launches, each followed by a visible sync gap, with the GPU idle most of the
time between kernels (poor "GPU busy %"). What Nsight Compute's roofline
will show: the gate-application and Pauli-matvec kernels sitting far below
BOTH rooflines (achieved GB/s and achieved FLOP/s both tiny relative to what
the hardware can do), because at 4-12 qubits the statevector is small enough
that kernel-launch overhead, not memory or compute throughput, dominates.
See NOTES.md for the full diagnosis and what stage2 fixes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cupy as cp
import numpy as np

from common.ansatz import Gate, build_circuit, single_qubit_matrix
from common.config import add_vqe_args, get_terms, resolve_qubits

_PAULI_GPU = {
    "I": cp.eye(2, dtype=cp.complex128),
    "X": cp.array([[0, 1], [1, 0]], dtype=cp.complex128),
    "Y": cp.array([[0, -1j], [1j, 0]], dtype=cp.complex128),
    "Z": cp.array([[1, 0], [0, -1]], dtype=cp.complex128),
}


def zero_state_gpu(n_qubits: int) -> cp.ndarray:
    psi = cp.zeros(2**n_qubits, dtype=cp.complex128)
    psi[0] = 1.0
    return psi


def apply_single_qubit_gate_naive(psi: cp.ndarray, n_qubits: int, qubit: int, mat: cp.ndarray) -> cp.ndarray:
    """One gate = one tensordot kernel launch + one sync. Deliberately unfused."""
    psi_t = psi.reshape([2] * n_qubits)
    psi_t = cp.moveaxis(psi_t, qubit, 0)
    psi_t = cp.tensordot(mat, psi_t, axes=([1], [0]))
    psi_t = cp.moveaxis(psi_t, 0, qubit)
    out = psi_t.reshape(-1)
    cp.cuda.Stream.null.synchronize()  # naive: sync after every single gate
    return out


def apply_cnot_naive(psi: cp.ndarray, n_qubits: int, control: int, target: int) -> cp.ndarray:
    psi_t = psi.reshape([2] * n_qubits)
    psi_t = cp.moveaxis(psi_t, [control, target], [0, 1])
    out = psi_t.copy()
    out[1, 0] = psi_t[1, 1]
    out[1, 1] = psi_t[1, 0]
    out = cp.moveaxis(out, [0, 1], [control, target])
    result = out.reshape(-1)
    cp.cuda.Stream.null.synchronize()  # naive: sync after every single gate
    return result


def apply_gate_naive(psi: cp.ndarray, n_qubits: int, gate: Gate) -> cp.ndarray:
    if gate.name == "CNOT":
        return apply_cnot_naive(psi, n_qubits, gate.qubits[0], gate.qubits[1])
    mat = cp.asarray(single_qubit_matrix(gate))
    return apply_single_qubit_gate_naive(psi, n_qubits, gate.qubits[0], mat)


def run_circuit_naive(n_qubits: int, depth: int, params: np.ndarray) -> cp.ndarray:
    """Gate-by-gate execution, GPU kernel + host sync per gate. This is the
    'gate application kernel' tracked across all four stages."""
    psi = zero_state_gpu(n_qubits)
    for gate in build_circuit(n_qubits, depth, params):
        psi = apply_gate_naive(psi, n_qubits, gate)
    return psi


def pauli_string_matrix_gpu(pauli_string: str) -> cp.ndarray:
    """Build the FULL dense 2^n x 2^n Pauli matrix via repeated cp.kron.
    Naive: O(4^n) memory, rebuilt independently for every single term even
    though many terms share structure."""
    mat = _PAULI_GPU[pauli_string[0]]
    for c in pauli_string[1:]:
        mat = cp.kron(mat, _PAULI_GPU[c])
    return mat


def expectation_value_naive(psi: cp.ndarray, pauli_string: str) -> float:
    """This is the 'expval_reduce kernel' tracked across all four stages,
    in its most naive form: dense matrix build + dense matvec + dot product,
    with an explicit host round-trip via .get() at the end."""
    mat = pauli_string_matrix_gpu(pauli_string)
    Hpsi = mat @ psi
    val = cp.vdot(psi, Hpsi)
    cp.cuda.Stream.null.synchronize()
    return float(cp.real(val).get())  # naive: one host sync per Hamiltonian term


def vqe_energy_naive(n_qubits: int, depth: int, params: np.ndarray, terms: list[tuple[float, str]]) -> float:
    psi = run_circuit_naive(n_qubits, depth, params)
    energy = 0.0
    for coeff, pauli in terms:
        energy += coeff * expectation_value_naive(psi, pauli)  # one host sync per term, no batching
    return energy


def main():
    ap = argparse.ArgumentParser(description="Stage 1 naive GPU VQE energy evaluation.")
    add_vqe_args(ap)
    args = ap.parse_args()

    n_qubits = resolve_qubits(args.molecule, args.qubits)
    n_qubits, terms = get_terms(args.molecule, n_qubits, args.seed)

    from common.ansatz import random_params

    params = random_params(n_qubits, args.depth, seed=args.seed)

    t0 = time.perf_counter()
    energy = vqe_energy_naive(n_qubits, args.depth, params, terms)
    t1 = time.perf_counter()

    print(f"[stage1_naive] molecule={args.molecule} n_qubits={n_qubits} depth={args.depth}")
    print(f"[stage1_naive] energy = {energy:.10f} Ha")
    print(f"[stage1_naive] wall time = {(t1 - t0) * 1e3:.3f} ms")


if __name__ == "__main__":
    main()
