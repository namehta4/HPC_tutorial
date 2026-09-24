// Stage 4 gate-application kernel: batched across MANY independent
// statevectors, one CUDA block per statevector, to fix stage3's occupancy
// problem.
//
// Why this file is named "_tc" (tensor-core) but doesn't use WMMA: a
// single-qubit gate is a 2x2 unitary. Tensor cores operate on >=16x16x16
// matrix tiles (see kernels/policy_forward_gemm.cu, which DOES use WMMA,
// for a case where the natural operation size fits). Trying to force a 2x2
// gate through a tensor core would waste >99% of the tile on padding -- so
// the honest "compute-bound tuning" story for VQE gate application is NOT
// tensor cores, it's OCCUPANCY: stage3's apply_ansatz_shared launched
// exactly ONE thread block, using at most one Streaming Multiprocessor
// (SM) of the GPU's 108 (A100). For a single small circuit that's
// inherent -- but VQE in practice needs MANY circuit evaluations per
// optimizer step (e.g. parameter-shift-rule gradients: 2 evaluations per
// parameter, so 2 * n_params per gradient step; or a batch of GQE-sampled
// candidate ansatz parameter sets to evaluate in parallel).
//
// This kernel launches ONE BLOCK PER STATEVECTOR (grid.x = batch_size), so
// a batch of B independent VQE evaluations occupies up to B SMs
// concurrently instead of stage3's B sequential single-SM kernel launches.
// This is the concrete "occupancy tuning" and "multi-stream-like scaling"
// this tutorial's stage4 promises for the VQE side, achieved via grid
// parallelism rather than multiple CUDA streams (grid parallelism is
// strictly more efficient here: the GPU scheduler packs blocks onto
// whichever SMs are free, without any host-side stream bookkeeping).

#include <cuComplex.h>

extern "C" __global__
void apply_ansatz_batched(
    cuDoubleComplex* psi_batch,             // (batch_size, dim), in place
    const cuDoubleComplex* fused_matrices,  // (batch_size, depth * n_qubits, 4)
    int n_qubits,
    int depth
) {
    extern __shared__ cuDoubleComplex psi_shared[];
    const int dim = 1 << n_qubits;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int b = blockIdx.x;  // one block per statevector in the batch

    cuDoubleComplex* psi = psi_batch + (size_t)b * dim;
    const cuDoubleComplex* mats = fused_matrices + (size_t)b * depth * n_qubits * 4;

    for (int i = tid; i < dim; i += nthreads) psi_shared[i] = psi[i];
    __syncthreads();

    const int half = dim >> 1;
    for (int layer = 0; layer < depth; layer++) {
        for (int q = 0; q < n_qubits; q++) {
            const int bit = n_qubits - 1 - q;
            const int mat_idx = (layer * n_qubits + q) * 4;
            const cuDoubleComplex m00 = mats[mat_idx + 0];
            const cuDoubleComplex m01 = mats[mat_idx + 1];
            const cuDoubleComplex m10 = mats[mat_idx + 2];
            const cuDoubleComplex m11 = mats[mat_idx + 3];
            const int low_mask = (1 << bit) - 1;
            for (int t = tid; t < half; t += nthreads) {
                const int low = t & low_mask;
                const int high = t >> bit;
                const int i0 = (high << (bit + 1)) | low;
                const int i1 = i0 | (1 << bit);
                const cuDoubleComplex a0 = psi_shared[i0];
                const cuDoubleComplex a1 = psi_shared[i1];
                psi_shared[i0] = cuCadd(cuCmul(m00, a0), cuCmul(m01, a1));
                psi_shared[i1] = cuCadd(cuCmul(m10, a0), cuCmul(m11, a1));
            }
            __syncthreads();
        }
        for (int q = 0; q < n_qubits - 1; q++) {
            const int control_bit = n_qubits - 1 - q;
            const int target_bit = n_qubits - 1 - (q + 1);
            for (int i = tid; i < dim; i += nthreads) {
                const int cv = (i >> control_bit) & 1;
                const int tv = (i >> target_bit) & 1;
                if (cv == 1 && tv == 0) {
                    const int partner = i | (1 << target_bit);
                    const cuDoubleComplex tmp = psi_shared[i];
                    psi_shared[i] = psi_shared[partner];
                    psi_shared[partner] = tmp;
                }
            }
            __syncthreads();
        }
    }

    for (int i = tid; i < dim; i += nthreads) psi[i] = psi_shared[i];
}

// Batched expectation-value reduction, same batching strategy: one block
// per statevector, reusing that statevector across every Hamiltonian term
// within the block (same shared-memory-reuse idea as stage3, now combined
// with cross-statevector grid parallelism).
extern "C" __global__
void expval_pauli_batched_grid(
    const cuDoubleComplex* psi_batch,  // (batch_size, dim)
    const long long* flip_masks,       // (n_terms,)
    const long long* z_masks,          // (n_terms,)
    const int* y_counts,               // (n_terms,)
    double* out_expvals,               // (batch_size, n_terms)
    int n_qubits,
    int n_terms
) {
    extern __shared__ unsigned char smem_raw[];
    cuDoubleComplex* psi_shared = (cuDoubleComplex*)smem_raw;
    const int dim = 1 << n_qubits;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int b = blockIdx.x;

    double* partial = (double*)(smem_raw + dim * sizeof(cuDoubleComplex));
    const cuDoubleComplex* psi = psi_batch + (size_t)b * dim;
    double* out = out_expvals + (size_t)b * n_terms;

    for (int i = tid; i < dim; i += nthreads) psi_shared[i] = psi[i];
    __syncthreads();

    const cuDoubleComplex i_pow[4] = {
        make_cuDoubleComplex(1.0, 0.0), make_cuDoubleComplex(0.0, 1.0),
        make_cuDoubleComplex(-1.0, 0.0), make_cuDoubleComplex(0.0, -1.0),
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

        partial[tid] = local_sum;
        __syncthreads();
        for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) partial[tid] += partial[tid + stride];
            __syncthreads();
        }
        if (tid == 0) out[term] = partial[0];
        __syncthreads();
    }
}
