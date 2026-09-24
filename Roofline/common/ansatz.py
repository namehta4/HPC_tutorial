"""Hardware-efficient ansatz (HEA) circuit definition, shared by every stage.

This module defines the ansatz *once* so that stage1 through stage4 all
simulate the literal same circuit -- same gate sequence, same parameter
count, same qubit-wiring -- and differ only in how efficiently they execute
it on the GPU. That's what makes the roofline overlay plot a fair
before/after comparison instead of an apples-to-oranges one.

Circuit structure, per layer:
    RY(theta) on every qubit
    RZ(phi)   on every qubit
    ladder of CNOT(i, i+1) for i in [0, n_qubits-2]   (linear entangling layer)

`depth` layers are stacked. Parameter count = 2 * n_qubits * depth.

This is the same style of ansatz used in Kandala et al. 2017 and most
subsequent VQE hardware demonstrations: cheap to execute, expressive enough
for small molecules, and it maps cleanly onto a GPU gate-application kernel
because every gate is either a single-qubit 2x2 unitary or a fixed CNOT.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Gate:
    name: str  # "RY", "RZ", or "CNOT"
    qubits: tuple  # (target,) for single-qubit gates, (control, target) for CNOT
    param: float = 0.0  # rotation angle; unused for CNOT


def n_params(n_qubits: int, depth: int) -> int:
    return 2 * n_qubits * depth


def build_circuit(n_qubits: int, depth: int, params: np.ndarray) -> list[Gate]:
    """Expand (n_qubits, depth, flat params) into an explicit gate list.

    `params` must have length n_params(n_qubits, depth). Layout: for each
    layer, first all RY angles (one per qubit), then all RZ angles (one per
    qubit) -- this groups same-type gates together, which stage2+ exploit to
    batch/fuse gate application across qubits within a layer.
    """
    expected = n_params(n_qubits, depth)
    if len(params) != expected:
        raise ValueError(f"expected {expected} params for n_qubits={n_qubits}, depth={depth}, got {len(params)}")

    gates = []
    idx = 0
    for _layer in range(depth):
        ry_angles = params[idx : idx + n_qubits]
        idx += n_qubits
        rz_angles = params[idx : idx + n_qubits]
        idx += n_qubits

        for q in range(n_qubits):
            gates.append(Gate("RY", (q,), float(ry_angles[q])))
        for q in range(n_qubits):
            gates.append(Gate("RZ", (q,), float(rz_angles[q])))
        for q in range(n_qubits - 1):
            gates.append(Gate("CNOT", (q, q + 1), 0.0))

    return gates


def single_qubit_matrix(gate: Gate) -> np.ndarray:
    """Dense 2x2 unitary for RY/RZ gates (complex128)."""
    theta = gate.param
    if gate.name == "RY":
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)
    if gate.name == "RZ":
        return np.array(
            [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
            dtype=np.complex128,
        )
    raise ValueError(f"single_qubit_matrix called on non-single-qubit gate {gate.name}")


def random_params(n_qubits: int, depth: int, seed: int = 0) -> np.ndarray:
    """Deterministic pseudo-random parameters, for reproducible correctness tests."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-np.pi, np.pi, size=n_params(n_qubits, depth)).astype(np.float64)
