#!/usr/bin/env python3
"""Correctness tests for stage4_compute_opt, run with: pytest test_correctness.py -v

Run ./build.sh first to compile the CUDA kernels this stage depends on.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import torch

from common.ansatz import random_params
from common.gqe_policy_ref import GQEConfig, deterministic_weights, export_weights_numpy, sample_tokens
from common.hamiltonians import get_hamiltonian, random_pauli_hamiltonian
from common.reference_cpu import vqe_energy
from stage4_compute_opt.gqe_multistream import policy_forward_tc, run_multistream_sampling_loop, upload_weights
from stage4_compute_opt.vqe_compute_opt import vqe_energies_batched

ENERGY_TOL = 1e-6
LOGITS_TOL = 0.05  # fp16 tensor-core GEMM path


@pytest.mark.parametrize("molecule", ["h2", "lih"])
@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_vqe_batched_matches_cpu_reference(molecule, batch_size):
    n_qubits, terms = get_hamiltonian(molecule)
    depth = 2
    rng = np.random.default_rng(0)
    base = random_params(n_qubits, depth, seed=0)
    params_batch = np.stack([base + rng.normal(scale=0.05, size=base.shape) for _ in range(batch_size)])

    expected = np.array([vqe_energy(n_qubits, depth, p, terms) for p in params_batch])
    actual = vqe_energies_batched(n_qubits, depth, params_batch, terms)

    np.testing.assert_allclose(actual, expected, atol=ENERGY_TOL)


@pytest.mark.parametrize("n_qubits", [4, 8, 10, 12])
def test_vqe_batched_full_qubit_range(n_qubits):
    terms = random_pauli_hamiltonian(n_qubits, n_terms=20, seed=1)
    depth = 2
    rng = np.random.default_rng(1)
    base = random_params(n_qubits, depth, seed=1)
    params_batch = np.stack([base + rng.normal(scale=0.05, size=base.shape) for _ in range(4)])

    expected = np.array([vqe_energy(n_qubits, depth, p, terms) for p in params_batch])
    actual = vqe_energies_batched(n_qubits, depth, params_batch, terms)

    np.testing.assert_allclose(actual, expected, atol=ENERGY_TOL)


@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_gqe_policy_forward_tensorcore_matches_pytorch(batch_size):
    cfg = GQEConfig()
    model = deterministic_weights(cfg, seed=0)
    weights_np = export_weights_numpy(model)
    weights_gpu = upload_weights(weights_np)

    tokens_torch = sample_tokens(batch_size, cfg, seed=1)
    with torch.no_grad():
        expected = model(tokens_torch).numpy()

    import cupy as cp

    tokens_gpu = cp.asarray(tokens_torch.numpy())
    actual = cp.asnumpy(policy_forward_tc(tokens_gpu, weights_gpu, cfg))

    max_abs_diff = np.max(np.abs(actual - expected))
    assert max_abs_diff < LOGITS_TOL, f"batch={batch_size}: max abs diff {max_abs_diff}"


def test_gqe_multistream_produces_correct_shapes_and_values():
    cfg = GQEConfig()
    model = deterministic_weights(cfg, seed=0)
    weights_np = export_weights_numpy(model)
    weights_gpu = upload_weights(weights_np)

    n_streams, batch_size = 3, 4
    results = run_multistream_sampling_loop(n_streams, batch_size, weights_gpu, cfg, seed=0)

    assert len(results) == n_streams
    for r in results:
        assert r.shape == (batch_size, cfg.seq_len, cfg.vocab_size)
        assert bool((r == r).all())  # no NaNs


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
