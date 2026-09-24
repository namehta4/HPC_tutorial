#!/usr/bin/env python3
"""Correctness tests for stage3_memory_opt, run with: pytest test_correctness.py -v

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
from stage3_memory_opt.gqe_batched_fp16 import policy_forward_fp16, upload_weights_fp16
from stage3_memory_opt.vqe_memopt import vqe_energy_memopt

FP64_TOL = 1e-6
FP32_TOL = 1e-5  # fp32 statevector storage: energies still match to ~7-8 significant figures
FP16_ABS_TOL = 0.01  # fp16 logits: absolute tolerance only, matching the storage format's precision


@pytest.mark.parametrize("molecule", ["h2", "lih"])
@pytest.mark.parametrize("depth", [1, 3])
def test_vqe_memopt_fp64_matches_cpu_reference(molecule, depth):
    n_qubits, terms = get_hamiltonian(molecule)
    params = random_params(n_qubits, depth, seed=0)

    expected = vqe_energy(n_qubits, depth, params, terms)
    actual = vqe_energy_memopt(n_qubits, depth, params, terms, precision="fp64")

    assert abs(actual - expected) < FP64_TOL, f"{molecule} depth={depth}: {actual} vs {expected}"


@pytest.mark.parametrize("molecule", ["h2", "lih"])
def test_vqe_memopt_fp32_matches_cpu_reference_loosely(molecule):
    n_qubits, terms = get_hamiltonian(molecule)
    depth = 3
    params = random_params(n_qubits, depth, seed=0)

    expected = vqe_energy(n_qubits, depth, params, terms)
    actual = vqe_energy_memopt(n_qubits, depth, params, terms, precision="fp32")

    assert abs(actual - expected) < FP32_TOL, f"{molecule}: {actual} vs {expected}"


@pytest.mark.parametrize("n_qubits", [4, 8, 10, 12])
def test_vqe_memopt_fp64_random_hamiltonian_full_qubit_range(n_qubits):
    """Confirms the whole-circuit-in-shared-memory kernel fits within the
    dynamic shared memory budget across this tutorial's entire 4-12 qubit
    range. Term count is capped: the CPU reference builds a dense 4^n Pauli
    matrix PER TERM (see common/reference_cpu.py's docstring -- it's
    deliberately naive), so an uncapped term count makes the *reference*,
    not the GPU kernel, the bottleneck at higher qubit counts."""
    terms = random_pauli_hamiltonian(n_qubits, n_terms=20, seed=1)
    params = random_params(n_qubits, depth=2, seed=1)

    expected = vqe_energy(n_qubits, 2, params, terms)
    actual = vqe_energy_memopt(n_qubits, 2, params, terms, precision="fp64")

    assert abs(actual - expected) < FP64_TOL


@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_gqe_policy_forward_fp16_matches_pytorch_reference(batch_size):
    cfg = GQEConfig()
    model = deterministic_weights(cfg, seed=0)
    weights_np = export_weights_numpy(model)
    weights_gpu = upload_weights_fp16(weights_np)

    tokens_torch = sample_tokens(batch_size, cfg, seed=1)
    with torch.no_grad():
        expected = model(tokens_torch).numpy()

    import cupy as cp

    tokens_gpu = cp.asarray(tokens_torch.numpy())
    actual = cp.asnumpy(policy_forward_fp16(tokens_gpu, weights_gpu, cfg))

    max_abs_diff = np.max(np.abs(actual - expected))
    assert max_abs_diff < FP16_ABS_TOL, f"batch={batch_size}: max abs diff {max_abs_diff}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
