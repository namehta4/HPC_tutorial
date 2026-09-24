"""Shared CLI argument parsing, used identically by every stage's scripts.

Keeping this in one place means `python vqe_naive.py --qubits 8 --molecule lih`
and `python vqe_compute_opt.py --qubits 8 --molecule lih` run the *same*
logical problem, just with different GPU implementations underneath --
which is what makes the cross-stage roofline comparison meaningful.
"""
from __future__ import annotations

import argparse

VALID_MOLECULES = ("h2", "lih", "random")
VALID_PRECISIONS = ("fp64", "fp32", "tf32", "fp16")


def add_vqe_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--molecule", choices=VALID_MOLECULES, default="h2",
        help="h2 (4 qubits) or lih (8 qubits) use real chemistry; "
        "random generates a synthetic Hamiltonian at --qubits size.",
    )
    parser.add_argument(
        "--qubits", type=int, default=None,
        help="Qubit count. Required for --molecule random (valid 4-12); "
        "ignored for h2/lih, which have a fixed qubit count.",
    )
    parser.add_argument("--depth", type=int, default=3, help="Ansatz layer depth.")
    parser.add_argument(
        "--precision", choices=VALID_PRECISIONS, default="fp64",
        help="Statevector / compute precision. fp64 is the numerically safe "
        "default; fp32/tf32/fp16 are introduced in stage3+ for mixed-precision experiments.",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for ansatz parameters.")
    parser.add_argument(
        "--tol", type=float, default=1e-6,
        help="Absolute energy tolerance vs. the CPU reference, used by test_correctness.py.",
    )
    return parser


def add_gqe_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--batch-size", type=int, default=32, help="Number of gate-sequence samples per forward pass.")
    parser.add_argument("--seq-len", type=int, default=8, help="Gate-sequence length (tokens).")
    parser.add_argument("--vocab-size", type=int, default=16, help="Number of discretized gate-parameter tokens.")
    parser.add_argument("--d-model", type=int, default=32, help="Transformer hidden dimension.")
    parser.add_argument("--n-layers", type=int, default=2, help="Number of transformer blocks.")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for weights/inputs.")
    return parser


def resolve_qubits(molecule: str, qubits: int | None) -> int:
    """Apply the h2=4 / lih=8 / random=--qubits rule consistently."""
    if molecule == "h2":
        return 4
    if molecule == "lih":
        return 8
    if qubits is None:
        raise ValueError("--qubits is required when --molecule random")
    if not (4 <= qubits <= 12):
        raise ValueError(f"--qubits must be in [4, 12] for this tutorial, got {qubits}")
    return qubits


def get_terms(molecule: str, qubits: int, seed: int):
    """Return (n_qubits, terms) for whichever molecule/qubit combination was requested."""
    from common.hamiltonians import get_hamiltonian, random_pauli_hamiltonian

    if molecule == "random":
        return qubits, random_pauli_hamiltonian(qubits, seed=seed)
    return get_hamiltonian(molecule)
