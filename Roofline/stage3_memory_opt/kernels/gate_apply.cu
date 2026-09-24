// Stage 3 gate-application kernel: the WHOLE ansatz circuit in ONE kernel launch.
//
// Stage2's hand-rolled version still launched one kernel PER GATE and read/
// wrote the full 2^n-element statevector from GLOBAL memory on every gate --
// even with fusion (RY+RZ combined) and no host syncs, that's still
// `n_qubits * depth + (n_qubits-1) * depth` separate global-memory round
// trips over the entire statevector.
//
// This kernel instead:
//   1. Loads the ENTIRE statevector into SHARED memory ONCE, at kernel start.
//   2. Applies every rotation and every CNOT in the whole circuit as
//      in-place updates to the shared-memory copy, with a __syncthreads()
//      between gates (needed for correctness: gate g+1 must see gate g's
//      fully-written result) but WITHOUT ever touching global memory.
//   3. Writes the final statevector back to global memory ONCE, at the end.
//
// Global memory traffic drops from O(n_gates * 2^n) to O(2^n) -- exactly the
// "reuse statevector tiles across gate applications" optimization this
// tutorial is about. Shared memory access is ~100x faster than global
// memory and doesn't count against achieved-bandwidth measurements the same
// way, which is what should move this kernel's ncu roofline point closer to
// (and for large-enough problems, near) the memory roofline.
//
// Practical limit: shared memory capacity. An A100 allows up to ~163KB of
// shared memory per block (opt-in, see build.sh / vqe_memopt.py for the
// cudaFuncAttributeMaxDynamicSharedMemorySize call this requires) -- at
// complex128 (16 bytes/amplitude), that's enough for up to 2^13 = 8192
// amplitude a hair over our 12-qubit ceiling (2^12 = 4096 amplitudes,
// 64KB), so this whole-circuit-in-shared-memory approach fits this
// tutorial's entire qubit range without spilling to global memory mid-circuit.

#include <cuComplex.h>

extern "C" __global__
void apply_ansatz_shared(
    cuDoubleComplex* psi_global,
    const cuDoubleComplex* fused_matrices,  // (depth * n_qubits, 4): row-major [m00,m01,m10,m11] per (layer,qubit)
    int n_qubits,
    int depth
) {
    extern __shared__ cuDoubleComplex psi_shared[];
    const int dim = 1 << n_qubits;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    // Single coalesced load of the whole statevector into shared memory.
    // Consecutive threads read consecutive global addresses -> coalesced.
    for (int i = tid; i < dim; i += nthreads) {
        psi_shared[i] = psi_global[i];
    }
    __syncthreads();

    const int half = dim >> 1;

    for (int layer = 0; layer < depth; layer++) {
        // --- Rotation layer: apply the pre-fused RZ@RY 2x2 matrix to every qubit ---
        for (int q = 0; q < n_qubits; q++) {
            const int bit = n_qubits - 1 - q;  // big-endian convention: qubit 0 = MSB
            const int mat_idx = (layer * n_qubits + q) * 4;
            const cuDoubleComplex m00 = fused_matrices[mat_idx + 0];
            const cuDoubleComplex m01 = fused_matrices[mat_idx + 1];
            const cuDoubleComplex m10 = fused_matrices[mat_idx + 2];
            const cuDoubleComplex m11 = fused_matrices[mat_idx + 3];

            const int low_mask = (1 << bit) - 1;
            for (int t = tid; t < half; t += nthreads) {
                // t indexes pairs of amplitudes that differ only in `bit`;
                // reconstruct the two full indices i0 (bit=0) and i1 (bit=1).
                const int low = t & low_mask;
                const int high = t >> bit;
                const int i0 = (high << (bit + 1)) | low;
                const int i1 = i0 | (1 << bit);

                const cuDoubleComplex a0 = psi_shared[i0];
                const cuDoubleComplex a1 = psi_shared[i1];
                psi_shared[i0] = cuCadd(cuCmul(m00, a0), cuCmul(m01, a1));
                psi_shared[i1] = cuCadd(cuCmul(m10, a0), cuCmul(m11, a1));
            }
            __syncthreads();  // next gate must see this gate's fully-written result
        }

        // --- Entangling CNOT ladder: CNOT(q, q+1) for q in [0, n_qubits-2] ---
        for (int q = 0; q < n_qubits - 1; q++) {
            const int control_bit = n_qubits - 1 - q;
            const int target_bit = n_qubits - 1 - (q + 1);
            for (int i = tid; i < dim; i += nthreads) {
                const int control_val = (i >> control_bit) & 1;
                const int target_val = (i >> target_bit) & 1;
                // Swap amplitude pairs where control=1: (target=0) <-> (target=1).
                // Only one thread per pair performs the swap (the one that
                // currently sees target=0) to avoid a race between the two
                // threads that "own" i and its partner.
                if (control_val == 1 && target_val == 0) {
                    const int partner = i | (1 << target_bit);
                    const cuDoubleComplex tmp = psi_shared[i];
                    psi_shared[i] = psi_shared[partner];
                    psi_shared[partner] = tmp;
                }
            }
            __syncthreads();
        }
    }

    // Single coalesced store of the final statevector back to global memory.
    for (int i = tid; i < dim; i += nthreads) {
        psi_global[i] = psi_shared[i];
    }
}
