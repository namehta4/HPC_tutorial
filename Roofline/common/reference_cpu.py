"""NumPy statevector simulator and VQE energy evaluator.

This is the correctness oracle for the entire tutorial. Every stage
(stage1 through stage4, on GPU, in progressively more optimized code) must
reproduce the energy this module computes, within a small numerical
tolerance. If a "faster" GPU version and this module disagree, the GPU
version is *wrong*, not fast -- that's the whole point of having a
dead-simple, obviously-correct reference implementation to check against.

Deliberately not optimized: dense complex128 statevector, gates applied via
explicit tensor reshaping, one Pauli term at a time. Fine for up to ~16
qubits (2^16 = 65536 amplitudes), which comfortably covers this tutorial's
4-12 qubit range.
"""
from __future__ import annotations

import numpy as np

from common.ansatz import Gate, build_circuit, single_qubit_matrix

_PAULI = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def zero_state(n_qubits: int) -> np.ndarray:
    psi = np.zeros(2**n_qubits, dtype=np.complex128)
    psi[0] = 1.0
    return psi


def apply_single_qubit_gate(psi: np.ndarray, n_qubits: int, qubit: int, mat: np.ndarray) -> np.ndarray:
    """Apply a 2x2 unitary to `qubit` via reshape-contract-reshape.

    Qubit 0 is the most significant bit of the state index (big-endian),
    matching the convention used consistently across every stage.
    """
    psi_t = psi.reshape([2] * n_qubits)
    psi_t = np.moveaxis(psi_t, qubit, 0)
    psi_t = np.tensordot(mat, psi_t, axes=([1], [0]))
    psi_t = np.moveaxis(psi_t, 0, qubit)
    return psi_t.reshape(-1)


def apply_cnot(psi: np.ndarray, n_qubits: int, control: int, target: int) -> np.ndarray:
    psi_t = psi.reshape([2] * n_qubits)
    psi_t = np.moveaxis(psi_t, [control, target], [0, 1])
    out = psi_t.copy()
    out[1, 0] = psi_t[1, 1]
    out[1, 1] = psi_t[1, 0]
    out = np.moveaxis(out, [0, 1], [control, target])
    return out.reshape(-1)


def apply_gate(psi: np.ndarray, n_qubits: int, gate: Gate) -> np.ndarray:
    if gate.name == "CNOT":
        return apply_cnot(psi, n_qubits, gate.qubits[0], gate.qubits[1])
    mat = single_qubit_matrix(gate)
    return apply_single_qubit_gate(psi, n_qubits, gate.qubits[0], mat)


def run_circuit(n_qubits: int, depth: int, params: np.ndarray) -> np.ndarray:
    """Return the statevector after applying the ansatz to |0...0>."""
    psi = zero_state(n_qubits)
    for gate in build_circuit(n_qubits, depth, params):
        psi = apply_gate(psi, n_qubits, gate)
    return psi


def pauli_string_matrix(pauli_string: str) -> np.ndarray:
    """Dense 2^n x 2^n matrix for a Pauli string, via Kronecker product."""
    mat = _PAULI[pauli_string[0]]
    for c in pauli_string[1:]:
        mat = np.kron(mat, _PAULI[c])
    return mat


def expectation_value(psi: np.ndarray, pauli_string: str) -> float:
    """<psi| P |psi> for a single Pauli string, dense reference implementation.

    O(4^n) memory / O(4^n) compute -- intentionally naive. Fine up to ~12
    qubits, which is this tutorial's ceiling.
    """
    mat = pauli_string_matrix(pauli_string)
    return float(np.real(np.vdot(psi, mat @ psi)))


def vqe_energy(n_qubits: int, depth: int, params: np.ndarray, terms: list[tuple[float, str]]) -> float:
    """E(theta) = sum_i coeff_i * <psi(theta)| P_i |psi(theta)>."""
    psi = run_circuit(n_qubits, depth, params)
    return sum(coeff * expectation_value(psi, pauli) for coeff, pauli in terms)
