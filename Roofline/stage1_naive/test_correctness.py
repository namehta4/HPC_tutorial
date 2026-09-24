#!/usr/bin/env python3
"""Correctness tests for stage1_naive, run with: pytest test_correctness.py -v

Every GPU implementation in this tutorial is checked against a CPU/NumPy
(or, for GQE, CPU/PyTorch) reference. A stage only "counts" as done if its
numbers match the reference within tolerance -- speed without correctness
is not progress.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cupy as cp
import numpy as np
import pytest
import torch

from common.ansatz import random_params
from common.gqe_policy_ref import GQEConfig, deterministic_weights, export_weights_numpy, sample_tokens
from common.hamiltonians import MOLECULES, get_hamiltonian
from common.reference_cpu import vqe_energy
from stage1_naive.gqe_naive import policy_forward_naive, upload_weights
from stage1_naive.vqe_naive import vqe_energy_naive

ENERGY_TOL = 1e-6
LOGITS_TOL = 1e-3  # GELU/softmax computed slightly differently (tanh-approx vs exact); logits, not probabilities


@pytest.mark.parametrize("molecule", ["h2", "lih"])
@pytest.mark.parametrize("depth", [1, 3])
def test_vqe_energy_matches_cpu_reference(molecule, depth):
    n_qubits, terms = get_hamiltonian(molecule)
    params = random_params(n_qubits, depth, seed=0)

    expected = vqe_energy(n_qubits, depth, params, terms)
    actual = vqe_energy_naive(n_qubits, depth, params, terms)

    assert abs(actual - expected) < ENERGY_TOL, f"{molecule} depth={depth}: {actual} vs {expected}"


@pytest.mark.parametrize("n_qubits", [4, 6, 8])
def test_vqe_energy_random_hamiltonian(n_qubits):
    from common.hamiltonians import random_pauli_hamiltonian

    terms = random_pauli_hamiltonian(n_qubits, seed=1)
    params = random_params(n_qubits, depth=2, seed=1)

    expected = vqe_energy(n_qubits, 2, params, terms)
    actual = vqe_energy_naive(n_qubits, 2, params, terms)

    assert abs(actual - expected) < ENERGY_TOL


@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_gqe_policy_forward_matches_pytorch_reference(batch_size):
    cfg = GQEConfig()
    model = deterministic_weights(cfg, seed=0)
    weights_np = export_weights_numpy(model)

    tokens_torch = sample_tokens(batch_size, cfg, seed=1)
    with torch.no_grad():
        expected = model(tokens_torch).numpy()

    tokens_gpu = cp.asarray(tokens_torch.numpy())
    weights_gpu = upload_weights(weights_np)
    actual = cp.asnumpy(policy_forward_naive(tokens_gpu, weights_gpu, cfg))

    np.testing.assert_allclose(actual, expected, atol=LOGITS_TOL, rtol=LOGITS_TOL)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
