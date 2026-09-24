// Stage 4 tensor-core GEMM kernel for the GQE policy forward pass.
//
// Unlike VQE's 2x2 single-qubit gates (see gate_apply_tc.cu's header for
// why those are a poor fit for tensor cores), the GQE policy network's
// linear layers and attention score/value matmuls are genuine matrix
// multiplies -- exactly the shape NVIDIA tensor cores are built for. This
// kernel implements a batched half-precision GEMM using the WMMA
// (Warp Matrix Multiply-Accumulate) C++ API: each warp cooperatively
// computes one 16x16 output tile using the tensor core's native
// 16x16x16 fp16-multiply/fp32-accumulate primitive.
//
// This is deliberately a SINGLE reusable batched-GEMM primitive (C = A @ B
// for a batch of A matrices against one shared B, which is exactly the
// shape of "batch of activations times one weight matrix" used by every
// linear layer in the policy network) rather than a full custom transformer
// kernel -- the teaching point is tensor-core utilization, not writing a
// from-scratch flash-attention implementation.
//
// Precision: inputs are fp16 (half), accumulation is fp32 (standard mixed-
// precision GEMM practice -- fp16 accumulation loses far more accuracy for
// the sums this network computes than fp16 storage of the inputs does).
//
// Tile/shape constraints of the WMMA fp16 path used here: M, N, K must each
// be multiples of 16. policy_forward_gemm.py pads any incompatible
// dimension up to the next multiple of 16 with zeros before calling this
// kernel, and slices the result back down -- correctness-preserving since
// padding with zero rows/columns doesn't change the product's non-padded
// entries.

#include <mma.h>
using namespace nvcuda;

#define TILE 16

// C[b] = A[b] @ B  for b in [0, batch_size), A: (batch_size, M, K) row-major
// half, B: (K, N) row-major half (shared across the batch -- this is a
// linear layer's weight matrix), C: (batch_size, M, N) row-major float.
//
// Grid: (M/16, N/16, batch_size). Block: 32 threads (exactly one warp) --
// one warp computes one 16x16 output tile via wmma::mma_sync.
extern "C" __global__
void batched_gemm_tc_fp16(
    const half* A, const half* B, float* C,
    int batch_size, int M, int N, int K
) {
    const int tile_m = blockIdx.x;
    const int tile_n = blockIdx.y;
    const int b = blockIdx.z;

    const int row0 = tile_m * TILE;
    const int col0 = tile_n * TILE;
    if (row0 >= M || col0 >= N) return;

    wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    const half* A_b = A + (size_t)b * M * K;

    for (int k0 = 0; k0 < K; k0 += TILE) {
        wmma::load_matrix_sync(a_frag, A_b + row0 * K + k0, K);
        wmma::load_matrix_sync(b_frag, B + k0 * N + col0, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float* C_b = C + (size_t)b * M * N;
    wmma::store_matrix_sync(C_b + row0 * N + col0, c_frag, N, wmma::mem_row_major);
}
