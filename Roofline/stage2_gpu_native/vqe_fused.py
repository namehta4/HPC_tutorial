#!/usr/bin/env python3
"""Stage 2: GPU-native restructuring of the hand-rolled VQE simulator.

Same physics, same ansatz, same Hamiltonians as stage1 -- the energy this
computes must match stage1 (and the CPU reference) to numerical precision.
What changes is *how* the GPU is used:

  1. FUSED single-qubit gate application. In our ansatz (common/ansatz.py),
     each layer applies RY(q) then RZ(q) to every qubit q, then a CNOT
     ladder. RY(q) and RZ(q) act on the same qubit with no other gate
     touching q in between (other qubits' RY/RZ gates commute with q's, since
     they act on disjoint qubits) -- so instead of two separate statevector
     passes (RY, then RZ), we precompute the combined 2x2 unitary
     RZ(phi) @ RY(theta) on the HOST (cheap: 2x2 complex matmul) and apply
     it to the statevector in ONE pass. This halves the number of
     statevector-touching kernel launches for the rotation layers.

  2. NO host-device sync until the very end. Stage1 called
     cp.cuda.Stream.null.synchronize() after every gate. Here we let CUDA's
     own stream-ordering guarantee correctness -- operations enqueued on the
     same stream execute in order without the host needing to wait for each
     one individually. The host only blocks once, when it needs the final
     scalar energy value.

  3. BATCHED, ALGEBRAIC Pauli-term expectation values. Stage1 built a dense
     2^n x 2^n matrix per term (O(4^n) memory) and did a dense matvec.
     Here we exploit the structure of Pauli operators directly: applying a
     Pauli string to a computational basis state |b> just flips the bits
     where X or Y act and multiplies by a phase that depends on the
     PRE-FLIP bit values at the Y and Z positions. Both the bit-flip and the
     phase are expressible as vectorized elementwise operations over the
     full 2^n-element index array -- O(2^n) per term instead of O(4^n), and
     because it's pure elementwise CuPy, ALL Hamiltonian terms' phase/flip
     patterns can be computed as one batched (n_terms, 2^n)-shaped tensor
     operation instead of a Python loop over terms with a host round-trip
     per term.

  4. CUDA STREAMS. VQE in practice needs many energy evaluations per
     optimizer step (e.g. parameter-shift-rule gradients: 2 evaluations per
     parameter). We expose `vqe_energy_multi_params_streamed`, which runs
     several independent parameter sets concurrently on separate CUDA
     streams so the GPU has multiple independent instruction streams to
     interleave -- useful once qubit count is small enough that a single
     circuit's kernels don't fill the GPU by themselves.

Tracked kernels (same role as stage1, now fused/batched):
  - 'gate_apply'    -> apply_fused_rotation_layer / apply_cnot_layer
  - 'expval_reduce' -> batched_pauli_expectation
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cupy as cp
import numpy as np

from common.ansatz import n_params, single_qubit_matrix, Gate
from common.config import add_vqe_args, get_terms, resolve_qubits


def zero_state_gpu(n_qubits: int) -> cp.ndarray:
    psi = cp.zeros(2**n_qubits, dtype=cp.complex128)
    psi[0] = 1.0
    return psi


def fused_rotation_matrix(theta: float, phi: float) -> np.ndarray:
    """Combine RY(theta) then RZ(phi) into a single 2x2 unitary: RZ @ RY.
    Computed on host (trivial cost) -- this is the whole point of fusion,
    we pay a cheap 2x2x2 matmul once instead of two full statevector passes."""
    ry = single_qubit_matrix(Gate("RY", (0,), theta))
    rz = single_qubit_matrix(Gate("RZ", (0,), phi))
    return rz @ ry


def apply_single_qubit_gate(psi: cp.ndarray, n_qubits: int, qubit: int, mat: cp.ndarray) -> cp.ndarray:
    """Identical math to stage1's version, but callers no longer sync after
    every call -- kernels queue up on the default stream and only the final
    result is waited on."""
    psi_t = psi.reshape([2] * n_qubits)
    psi_t = cp.moveaxis(psi_t, qubit, 0)
    psi_t = cp.tensordot(mat, psi_t, axes=([1], [0]))
    psi_t = cp.moveaxis(psi_t, 0, qubit)
    return psi_t.reshape(-1)


def apply_cnot(psi: cp.ndarray, n_qubits: int, control: int, target: int) -> cp.ndarray:
    psi_t = psi.reshape([2] * n_qubits)
    psi_t = cp.moveaxis(psi_t, [control, target], [0, 1])
    out = psi_t.copy()
    out[1, 0] = psi_t[1, 1]
    out[1, 1] = psi_t[1, 0]
    out = cp.moveaxis(out, [0, 1], [control, target])
    return out.reshape(-1)


def run_circuit_fused(n_qubits: int, depth: int, params: np.ndarray) -> cp.ndarray:
    """Same ansatz as common/ansatz.build_circuit, but RY+RZ per qubit are
    pre-combined into one matrix before ever touching the statevector, and
    no synchronization happens until the caller asks for a result."""
    psi = zero_state_gpu(n_qubits)
    idx = 0
    for _layer in range(depth):
        ry_angles = params[idx : idx + n_qubits]
        idx += n_qubits
        rz_angles = params[idx : idx + n_qubits]
        idx += n_qubits

        for q in range(n_qubits):
            mat = cp.asarray(fused_rotation_matrix(float(ry_angles[q]), float(rz_angles[q])))
            psi = apply_single_qubit_gate(psi, n_qubits, q, mat)

        for q in range(n_qubits - 1):
            psi = apply_cnot(psi, n_qubits, q, q + 1)

    return psi


def _pauli_flip_and_phase_masks(n_qubits: int, pauli_string: str):
    """For one Pauli string, return:
      flip_mask: int, XOR mask of bits flipped by X/Y positions
      z_mask:    int, OR mask of bits contributing a (-1)^bit phase (Z or Y)
      y_mask:    int, OR mask of bits contributing an extra +/-i phase (Y)
    Qubit q maps to bit position (n_qubits - 1 - q) to match the tutorial's
    big-endian convention (qubit 0 = most significant bit of the index).
    """
    flip_mask = 0
    z_mask = 0
    y_mask = 0
    for q, c in enumerate(pauli_string):
        bit = n_qubits - 1 - q
        if c == "X":
            flip_mask |= 1 << bit
        elif c == "Y":
            flip_mask |= 1 << bit
            z_mask |= 1 << bit
            y_mask |= 1 << bit
        elif c == "Z":
            z_mask |= 1 << bit
    return flip_mask, z_mask, y_mask


def batched_pauli_expectation(psi: cp.ndarray, n_qubits: int, terms: list[tuple[float, str]]) -> cp.ndarray:
    """Compute <psi|P_i|psi> for every term, vectorized over BOTH the
    2^n basis states and the n_terms Hamiltonian terms at once -- this is
    the 'expval_reduce' kernel: one batched GPU computation instead of
    stage1's one-dense-matvec-plus-host-sync per term.

    Algebra: for a computational basis index `idx` with amplitude
    psi[idx], applying Pauli string P sends it to
    phase(idx) * psi[idx] at position (idx XOR flip_mask), where phase(idx)
    is a product of (-1) for each set Z/Y bit in idx, times i/-i for each Y
    position depending on the original bit value. So
      <psi|P|psi> = sum_idx conj(psi[idx]) * phase(idx) * psi[idx XOR flip_mask]
    computed for all terms simultaneously via broadcasting.
    """
    dim = 1 << n_qubits
    idx = cp.arange(dim, dtype=cp.int64)

    n_terms = len(terms)
    coeffs = np.array([c for c, _ in terms], dtype=np.float64)
    flip_masks = np.zeros(n_terms, dtype=np.int64)
    z_masks = np.zeros(n_terms, dtype=np.int64)
    y_masks = np.zeros(n_terms, dtype=np.int64)
    for t, (_, pauli) in enumerate(terms):
        fm, zm, ym = _pauli_flip_and_phase_masks(n_qubits, pauli)
        flip_masks[t] = fm
        z_masks[t] = zm
        y_masks[t] = ym

    flip_masks_gpu = cp.asarray(flip_masks).reshape(n_terms, 1)  # (T, 1)
    z_masks_gpu = cp.asarray(z_masks).reshape(n_terms, 1)
    y_masks_gpu = cp.asarray(y_masks).reshape(n_terms, 1)
    idx_row = idx.reshape(1, dim)  # (1, D)
    flipped_idx = idx_row ^ flip_masks_gpu  # (T, D); the pre-flip ket k = idx XOR flip_mask

    # Per-qubit single-Pauli action on a bit value b (see module docstring
    # derivation): X|b>=|1-b> (no phase), Z|b>=(-1)^b|b>, Y|b>=i*(-1)^b|1-b>
    # (since Y = i*X*Z as operators). For a multi-qubit Pauli string P|k>:
    #   P|k> = (-1)^popcount(k & z_mask) * i^popcount(y_mask) * |k XOR flip_mask>
    # where z_mask includes BOTH plain-Z and Y positions (both contribute a
    # (-1)^k_q factor), and the i^(#Y) factor is a per-term CONSTANT --
    # it does not depend on k, because the "i" in Y = i*X*Z is not
    # conditional on the bit value (only the (-1)^b sign is, and that's
    # already folded into z_mask).
    z_bits_set = flipped_idx & z_masks_gpu  # (T, D); bit values of pre-flip ket k
    z_parity = _popcount64(z_bits_set) & 1
    sign = 1.0 - 2.0 * z_parity.astype(cp.float64)  # (T, D), +1 or -1

    y_total = _popcount64(y_masks_gpu)  # (T, 1): number of Y operators in this term
    y_power = y_total % 4  # exponent of i, reduced mod 4; constant per term
    i_pow_table = cp.array([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=cp.complex128)  # i^0..i^3
    y_phase = i_pow_table[y_power]  # (T, 1), broadcasts over D

    phase = sign.astype(cp.complex128) * y_phase  # (T, D)

    psi_row = psi.reshape(1, dim)
    psi_flipped = cp.take_along_axis(cp.broadcast_to(psi_row, (n_terms, dim)), flipped_idx, axis=1)
    psi_conj = cp.conj(psi_row)  # (1, D)

    contributions = psi_conj * phase * psi_flipped  # (T, D)
    exp_vals = cp.real(cp.sum(contributions, axis=1))  # (T,)
    return exp_vals


def _popcount64(x: cp.ndarray) -> cp.ndarray:
    """Vectorized popcount for int64 CuPy arrays (bit-twiddling, no native intrinsic in CuPy)."""
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    x = x + (x >> 8)
    x = x + (x >> 16)
    x = x + (x >> 32)
    return x & 0x7F


def vqe_energy_fused(n_qubits: int, depth: int, params: np.ndarray, terms: list[tuple[float, str]]) -> float:
    psi = run_circuit_fused(n_qubits, depth, params)
    coeffs = cp.asarray([c for c, _ in terms], dtype=cp.float64)
    exp_vals = batched_pauli_expectation(psi, n_qubits, terms)
    energy = cp.sum(coeffs * exp_vals)
    return float(energy.get())  # single host sync for the whole computation


def vqe_energy_multi_params_streamed(
    n_qubits: int, depth: int, params_list: list[np.ndarray], terms: list[tuple[float, str]]
) -> list[float]:
    """Evaluate several independent parameter sets concurrently on separate
    CUDA streams. Models the realistic VQE need for multiple energy
    evaluations per optimizer step (e.g. parameter-shift gradients, or
    multiple random restarts) -- work that is embarrassingly parallel across
    parameter sets but was serialized in stage1."""
    n_evals = len(params_list)
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(n_evals)]
    results = [None] * n_evals

    coeffs = cp.asarray([c for c, _ in terms], dtype=cp.float64)

    for i, (stream, params) in enumerate(zip(streams, params_list)):
        with stream:
            psi = run_circuit_fused(n_qubits, depth, params)
            exp_vals = batched_pauli_expectation(psi, n_qubits, terms)
            energy = cp.sum(coeffs * exp_vals)
            results[i] = energy  # still a device scalar; not synced yet

    for stream in streams:
        stream.synchronize()

    return [float(r.get()) for r in results]


def main():
    ap = argparse.ArgumentParser(description="Stage 2 GPU-native fused VQE energy evaluation.")
    add_vqe_args(ap)
    ap.add_argument("--n-stream-evals", type=int, default=1,
                     help="If >1, evaluate this many (perturbed) parameter sets concurrently on separate streams.")
    args = ap.parse_args()

    n_qubits = resolve_qubits(args.molecule, args.qubits)
    n_qubits, terms = get_terms(args.molecule, n_qubits, args.seed)

    from common.ansatz import random_params

    params = random_params(n_qubits, args.depth, seed=args.seed)

    if args.n_stream_evals <= 1:
        t0 = time.perf_counter()
        energy = vqe_energy_fused(n_qubits, args.depth, params, terms)
        t1 = time.perf_counter()
        print(f"[stage2_fused] molecule={args.molecule} n_qubits={n_qubits} depth={args.depth}")
        print(f"[stage2_fused] energy = {energy:.10f} Ha")
        print(f"[stage2_fused] wall time = {(t1 - t0) * 1e3:.3f} ms")
    else:
        rng = np.random.default_rng(args.seed)
        params_list = [params + rng.normal(scale=0.01, size=params.shape) for _ in range(args.n_stream_evals)]
        t0 = time.perf_counter()
        energies = vqe_energy_multi_params_streamed(n_qubits, args.depth, params_list, terms)
        t1 = time.perf_counter()
        print(f"[stage2_fused] {args.n_stream_evals} streamed evaluations, molecule={args.molecule} "
              f"n_qubits={n_qubits} depth={args.depth}")
        print(f"[stage2_fused] energies = {[f'{e:.6f}' for e in energies]}")
        print(f"[stage2_fused] wall time = {(t1 - t0) * 1e3:.3f} ms")


if __name__ == "__main__":
    main()
