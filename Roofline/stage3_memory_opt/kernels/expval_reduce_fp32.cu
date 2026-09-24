// FP32 (complex64) variant of expval_reduce.cu -- see that file for the
// full design rationale. Accumulation is kept in double precision
// (`local_sum`, `partial[]`) even though the statevector itself is loaded
// as complex64 -- summing thousands of single-precision products in
// single precision would lose much more accuracy than the storage format
// itself costs, and accumulation cost is O(n_terms) doubles, not O(2^n),
// so it's cheap to keep accurate.
#include <cuComplex.h>

extern "C" __global__
void expval_pauli_batched_shared_fp32(
    const cuFloatComplex* psi_global,
    const long long* flip_masks,
    const long long* z_masks,
    const int* y_counts,
    double* out_expvals,
    int n_qubits,
    int n_terms
) {
    extern __shared__ unsigned char smem_raw_f32[];
    cuFloatComplex* psi_shared = (cuFloatComplex*)smem_raw_f32;
    const int dim = 1 << n_qubits;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    double* partial = (double*)(smem_raw_f32 + dim * sizeof(cuFloatComplex));

    for (int i = tid; i < dim; i += nthreads) {
        psi_shared[i] = psi_global[i];
    }
    __syncthreads();

    const cuFloatComplex i_pow[4] = {
        make_cuFloatComplex(1.0f, 0.0f),
        make_cuFloatComplex(0.0f, 1.0f),
        make_cuFloatComplex(-1.0f, 0.0f),
        make_cuFloatComplex(0.0f, -1.0f),
    };

    for (int term = 0; term < n_terms; term++) {
        const long long flip_mask = flip_masks[term];
        const long long z_mask = z_masks[term];
        const cuFloatComplex y_phase = i_pow[y_counts[term] & 3];

        double local_sum = 0.0;
        for (int idx = tid; idx < dim; idx += nthreads) {
            const int flipped = idx ^ (int)flip_mask;
            const int z_bits = flipped & (int)z_mask;
            const int parity = __popc(z_bits) & 1;
            const float sign = parity ? -1.0f : 1.0f;

            const cuFloatComplex phase = cuCmulf(make_cuFloatComplex(sign, 0.0f), y_phase);
            const cuFloatComplex psi_conj = cuConjf(psi_shared[idx]);
            const cuFloatComplex psi_flip = psi_shared[flipped];
            const cuFloatComplex contrib = cuCmulf(psi_conj, cuCmulf(phase, psi_flip));
            local_sum += (double)cuCrealf(contrib);
        }

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
        __syncthreads();
    }
}
