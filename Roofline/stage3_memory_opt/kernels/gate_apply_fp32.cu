// FP32 (complex64) variant of gate_apply.cu -- see that file for the full
// design rationale (whole-circuit-in-shared-memory, single global load/store).
//
// This variant exists to demonstrate mixed precision as a MEMORY-TRAFFIC
// optimization: cuFloatComplex is 8 bytes/amplitude vs. cuDoubleComplex's
// 16 bytes/amplitude, so the single global load and single global store at
// the start/end of the kernel move HALF as many bytes. For a
// memory-bound-ish kernel (which our shared-memory-resident gate
// application increasingly is not, but the global load/store bookends
// still are), this is a real, measurable win -- at the cost of
// precision, which is why test_correctness.py checks FP32 VQE energies
// against a LOOSER tolerance than FP64, and why this tutorial doesn't
// pretend precision loss is free.
#include <cuComplex.h>

extern "C" __global__
void apply_ansatz_shared_fp32(
    cuFloatComplex* psi_global,
    const cuFloatComplex* fused_matrices,  // (depth * n_qubits, 4)
    int n_qubits,
    int depth
) {
    extern __shared__ cuFloatComplex psi_shared_f32[];
    const int dim = 1 << n_qubits;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    for (int i = tid; i < dim; i += nthreads) {
        psi_shared_f32[i] = psi_global[i];
    }
    __syncthreads();

    const int half = dim >> 1;

    for (int layer = 0; layer < depth; layer++) {
        for (int q = 0; q < n_qubits; q++) {
            const int bit = n_qubits - 1 - q;
            const int mat_idx = (layer * n_qubits + q) * 4;
            const cuFloatComplex m00 = fused_matrices[mat_idx + 0];
            const cuFloatComplex m01 = fused_matrices[mat_idx + 1];
            const cuFloatComplex m10 = fused_matrices[mat_idx + 2];
            const cuFloatComplex m11 = fused_matrices[mat_idx + 3];

            const int low_mask = (1 << bit) - 1;
            for (int t = tid; t < half; t += nthreads) {
                const int low = t & low_mask;
                const int high = t >> bit;
                const int i0 = (high << (bit + 1)) | low;
                const int i1 = i0 | (1 << bit);

                const cuFloatComplex a0 = psi_shared_f32[i0];
                const cuFloatComplex a1 = psi_shared_f32[i1];
                psi_shared_f32[i0] = cuCaddf(cuCmulf(m00, a0), cuCmulf(m01, a1));
                psi_shared_f32[i1] = cuCaddf(cuCmulf(m10, a0), cuCmulf(m11, a1));
            }
            __syncthreads();
        }

        for (int q = 0; q < n_qubits - 1; q++) {
            const int control_bit = n_qubits - 1 - q;
            const int target_bit = n_qubits - 1 - (q + 1);
            for (int i = tid; i < dim; i += nthreads) {
                const int control_val = (i >> control_bit) & 1;
                const int target_val = (i >> target_bit) & 1;
                if (control_val == 1 && target_val == 0) {
                    const int partner = i | (1 << target_bit);
                    const cuFloatComplex tmp = psi_shared_f32[i];
                    psi_shared_f32[i] = psi_shared_f32[partner];
                    psi_shared_f32[partner] = tmp;
                }
            }
            __syncthreads();
        }
    }

    for (int i = tid; i < dim; i += nthreads) {
        psi_global[i] = psi_shared_f32[i];
    }
}
