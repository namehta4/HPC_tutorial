// Stage 3 expectation-value kernel: statevector loaded into shared memory
// ONCE, then reused across EVERY Hamiltonian term.
//
// Stage2's batched_pauli_expectation was already a big improvement over
// stage1 (one vectorized CuPy operation instead of one dense matvec per
// term), but it still reads the full statevector from GLOBAL memory once
// per term via cp.take_along_axis (an XOR-indexed gather) -- for LiH's 105
// terms, that's 105 separate global-memory passes over the same data.
//
// This kernel instead loads the statevector into shared memory once and
// loops over all `n_terms` Hamiltonian terms entirely out of shared memory,
// doing a block-wide parallel reduction (sum over basis states) for each
// term. Global memory traffic drops from O(n_terms * 2^n) to O(2^n) for the
// state vector (the mask/coefficient arrays are tiny by comparison,
// O(n_terms) each). This is the concrete instance of "reuse statevector
// tiles across Pauli-term evaluations" called out in the top-level README's
// stage-3 description.
//
// Algebra: applying Pauli string P to basis ket |k> gives
//   P|k> = (-1)^popcount(k & z_mask) * i^(#Y ops) * |k XOR flip_mask>
// so  <psi|P|psi> = sum_k conj(psi[k]) * phase(k) * psi[k XOR flip_mask]
// (see stage2_gpu_native/vqe_fused.py's docstring for the full derivation;
// this kernel is the same math, hand-written instead of vectorized CuPy).

#include <cuComplex.h>

extern "C" __global__
void expval_pauli_batched_shared(
    const cuDoubleComplex* psi_global,
    const long long* flip_masks,   // (n_terms,)
    const long long* z_masks,      // (n_terms,)
    const int* y_counts,           // (n_terms,): number of Y operators in the term
    double* out_expvals,           // (n_terms,): <psi|P_term|psi>, real part only (Hermitian guarantee)
    int n_qubits,
    int n_terms
) {
    extern __shared__ unsigned char smem_raw[];
    cuDoubleComplex* psi_shared = (cuDoubleComplex*)smem_raw;
    const int dim = 1 << n_qubits;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    // Partial-sum scratch space lives right after the statevector in the
    // same dynamic shared-memory allocation (caller must size shared_mem =
    // dim*sizeof(cuDoubleComplex) + nthreads*sizeof(double)).
    double* partial = (double*)(smem_raw + dim * sizeof(cuDoubleComplex));

    for (int i = tid; i < dim; i += nthreads) {
        psi_shared[i] = psi_global[i];
    }
    __syncthreads();

    const cuDoubleComplex i_pow[4] = {
        make_cuDoubleComplex(1.0, 0.0),
        make_cuDoubleComplex(0.0, 1.0),
        make_cuDoubleComplex(-1.0, 0.0),
        make_cuDoubleComplex(0.0, -1.0),
    };

    for (int term = 0; term < n_terms; term++) {
        const long long flip_mask = flip_masks[term];
        const long long z_mask = z_masks[term];
        const cuDoubleComplex y_phase = i_pow[y_counts[term] & 3];

        double local_sum = 0.0;
        for (int idx = tid; idx < dim; idx += nthreads) {
            const int flipped = idx ^ (int)flip_mask;
            const int z_bits = flipped & (int)z_mask;
            const int parity = __popc(z_bits) & 1;
            const double sign = parity ? -1.0 : 1.0;

            const cuDoubleComplex phase = cuCmul(make_cuDoubleComplex(sign, 0.0), y_phase);
            const cuDoubleComplex psi_conj = cuConj(psi_shared[idx]);
            const cuDoubleComplex psi_flip = psi_shared[flipped];
            const cuDoubleComplex contrib = cuCmul(psi_conj, cuCmul(phase, psi_flip));
            local_sum += cuCreal(contrib);
        }

        // Block-wide tree reduction of this term's partial sums.
        partial[tid] = local_sum;
        __syncthreads();
        for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            out_expvals[term] = partial[0];
        }
        __syncthreads();  // ensure partial[] is safe to reuse for the next term
    }
}
