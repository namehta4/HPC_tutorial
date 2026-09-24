#!/usr/bin/env python3
"""Correctness tests for stage2_gpu_native, run with: pytest test_correctness.py -v"""
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
from common.hamiltonians import get_hamiltonian, random_pauli_hamiltonian
from common.reference_cpu import vqe_energy
from stage2_gpu_native.gqe_batched import policy_forward_batched, upload_weights_cached
from stage2_gpu_native.vqe_custatevec import vqe_energy_custatevec
from stage2_gpu_native.vqe_fused import vqe_energy_fused, vqe_energy_multi_params_streamed

ENERGY_TOL = 1e-6
LOGITS_TOL = 1e-3


@pytest.mark.parametrize("molecule", ["h2", "lih"])
@pytest.mark.parametrize("depth", [1, 3])
def test_vqe_fused_matches_cpu_reference(molecule, depth):
    n_qubits, terms = get_hamiltonian(molecule)
    params = random_params(n_qubits, depth, seed=0)

    expected = vqe_energy(n_qubits, depth, params, terms)
    actual = vqe_energy_fused(n_qubits, depth, params, terms)

    assert abs(actual - expected) < ENERGY_TOL, f"{molecule} depth={depth}: {actual} vs {expected}"


@pytest.mark.parametrize("molecule", ["h2", "lih"])
def test_vqe_custatevec_matches_cpu_reference(molecule):
    n_qubits, terms = get_hamiltonian(molecule)
    depth = 2
    params = random_params(n_qubits, depth, seed=0)

    expected = vqe_energy(n_qubits, depth, params, terms)
    actual = vqe_energy_custatevec(n_qubits, depth, params, terms)

    assert abs(actual - expected) < ENERGY_TOL, f"{molecule}: {actual} vs {expected}"


@pytest.mark.parametrize("n_qubits", [4, 6, 8, 10])
def test_vqe_fused_random_hamiltonian(n_qubits):
    terms = random_pauli_hamiltonian(n_qubits, seed=1)
    params = random_params(n_qubits, depth=2, seed=1)

    expected = vqe_energy(n_qubits, 2, params, terms)
    actual = vqe_energy_fused(n_qubits, 2, params, terms)

    assert abs(actual - expected) < ENERGY_TOL


def test_vqe_streamed_multi_params_matches_sequential():
    n_qubits, terms = get_hamiltonian("h2")
    depth = 2
    rng = np.random.default_rng(0)
    params_list = [random_params(n_qubits, depth, seed=i) for i in range(4)]

    expected = [vqe_energy(n_qubits, depth, p, terms) for p in params_list]
    actual = vqe_energy_multi_params_streamed(n_qubits, depth, params_list, terms)

    for e, a in zip(expected, actual):
        assert abs(a - e) < ENERGY_TOL


@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_gqe_policy_forward_batched_matches_pytorch(batch_size):
    cfg = GQEConfig()
    model = deterministic_weights(cfg, seed=0)
    weights_np = export_weights_numpy(model)

    tokens_torch = sample_tokens(batch_size, cfg, seed=1)
    with torch.no_grad():
        expected = model(tokens_torch).numpy()

    weights_gpu = upload_weights_cached(weights_np, cache_key=99)
    tokens_gpu = cp.asarray(tokens_torch.numpy())
    actual = cp.asnumpy(policy_forward_batched(tokens_gpu, weights_gpu, cfg))

    np.testing.assert_allclose(actual, expected, atol=LOGITS_TOL, rtol=LOGITS_TOL)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
