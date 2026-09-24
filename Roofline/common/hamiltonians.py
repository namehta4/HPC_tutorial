"""Molecular Hamiltonians for the VQE workload, as lists of (coeff, pauli_string).

Every Hamiltonian here is a sum of weighted Pauli strings:

    H = sum_i coeff_i * P_i,   P_i in {I, X, Y, Z}^n_qubits

This is exactly what a Jordan-Wigner-mapped electronic-structure Hamiltonian
looks like, and it is also the natural unit of work for a GPU kernel: each
term is one "apply this Pauli string's expectation value" reduction.

Two real molecules are provided with numerically verified coefficients
(see common/_hamiltonian_data.py and tools/generate_hamiltonians.py):

  - "h2":  4 qubits,  15 Pauli terms, ground energy -1.1372701747 Ha
  - "lih": 8 qubits, 105 Pauli terms, ground energy (active space) -7.8638448143 Ha

For qubit counts the real molecules don't cover (the tutorial wants a
configurable 4-12 qubit sweep so you can see how the roofline story changes
with problem size), `random_pauli_hamiltonian` builds a synthetic but
well-formed Hamiltonian: real coefficients, Hermitian by construction (every
Pauli string is its own conjugate transpose), same computational shape as a
real molecular Hamiltonian. It is NOT a physical molecule -- it exists purely
to let you scale the workload up and down and watch the roofline move.
"""
from __future__ import annotations

import numpy as np

from common._hamiltonian_data import H2_N_QUBITS, H2_TERMS, LIH_N_QUBITS, LIH_TERMS

MOLECULES = {
    "h2": {
        "n_qubits": H2_N_QUBITS,
        "terms": H2_TERMS,
        "fci_energy": -1.1372701747,
        "description": "H2, STO-3G, R=0.7414 Angstrom, full Jordan-Wigner map",
    },
    "lih": {
        "n_qubits": LIH_N_QUBITS,
        "terms": LIH_TERMS,
        "fci_energy": -7.8638448143,
        "description": "LiH, STO-3G, R=1.5949 Angstrom, reduced active space "
        "(frozen Li 1s core, 4 active spatial orbitals), Jordan-Wigner map",
    },
}


def get_hamiltonian(molecule: str) -> tuple[int, list[tuple[float, str]]]:
    """Return (n_qubits, terms) for a named molecule ('h2' or 'lih')."""
    if molecule not in MOLECULES:
        raise ValueError(
            f"Unknown molecule '{molecule}'. Choices: {list(MOLECULES)} "
            "(use random_pauli_hamiltonian() for other qubit counts)."
        )
    info = MOLECULES[molecule]
    return info["n_qubits"], info["terms"]


def random_pauli_hamiltonian(
    n_qubits: int, n_terms: int | None = None, seed: int = 0
) -> list[tuple[float, str]]:
    """Synthetic Hermitian Pauli-sum Hamiltonian for arbitrary qubit counts.

    Used only for the stage0 calibration sweep and for exploring how
    arithmetic intensity / roofline position changes as qubit count grows
    beyond what H2 (4) and LiH (8) cover, up to the tutorial's 12-qubit cap.
    Term count defaults to a molecule-like density (~n_qubits**2 terms,
    matching the O(n^4) electronic-integral term count seen in H2/LiH after
    symmetry cancellation, capped for tractability).
    """
    rng = np.random.default_rng(seed)
    if n_terms is None:
        n_terms = min(4 * n_qubits * n_qubits, 4**n_qubits)

    paulis = "IXYZ"
    terms = [(1.0, "I" * n_qubits)]  # constant/identity term, as in real Hamiltonians
    seen = {"I" * n_qubits}
    while len(terms) < n_terms:
        # Bias toward low-weight Pauli strings (mostly I), like real
        # fermion-to-qubit-mapped Hamiltonians, so term counts are realistic.
        weights = rng.choice(4, size=n_qubits, p=[0.6, 0.1333, 0.1333, 0.1334])
        string = "".join(paulis[w] for w in weights)
        if string in seen:
            continue
        seen.add(string)
        coeff = float(rng.uniform(-1.0, 1.0) / (1 + string.count("I")))
        terms.append((coeff, string))
    return terms
