// Stage 4 fused GQE policy forward pass: a WMMA tensor-core GEMM with a
// fused epilogue, plus fused layernorm and fused causal attention.
//
// Why fusion is the stage4 step: once stage3 put the linear layers on tensor
// cores, those GEMMs were only ~7% of the forward pass's GPU time. The rest
// was ~175 tiny memory-bound CuPy kernels (bias adds, GELU, layernorm
// pieces, softmax pieces, fp16<->fp32 casts), each reading and writing the
// activations from HBM. These kernels do that glue work inside the kernel
// that already has the data in registers, so a forward pass is ~16 launches
// instead of ~180.
//
// Unlike VQE's 2x2 single-qubit gates (see gate_apply_tc.cu's header for
// why those are a poor fit for tensor cores), the linear layers ARE genuine
// matrix multiplies. gemm_tc_fp16_fused issues each one as ONE GEMM over all
// batch*seq_len rows -- multiplying each row separately as a 16-row-padded
// tile would waste 15/16 of the tensor-core work on zeros.
//
// Precision: GEMM inputs are fp16, accumulation and all epilogue/layernorm/
// softmax math is fp32, and the residual stream is kept in fp32.
//
// Shape constraints: M (rows), N and K must be multiples of 16 (WMMA's fp16
// tile). gqe_multistream.py pads the row count up to a multiple of 16 and
// checks that d_model, d_ff and vocab_size are multiples of 16.

#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

#define TILE 16

// Epilogue applied to each GEMM output element v (= A @ W^T, fp32):
#define EPI_STORE_F16 0           // out16 = v (+ bias)
#define EPI_BIAS_GELU_F16 1       // out16 = gelu(v + bias)
#define EPI_RESIDUAL_ADD_F32 2    // out32 += v (+ bias)      (residual stream)
#define EPI_STORE_F32 3           // out32 = v (+ bias)       (final logits)

// out = epilogue(A @ W^T) for A: (M, K) row-major half, W: (N, K) row-major
// half (a PyTorch Linear weight, read as a column-major (K, N) matrix, so no
// transpose is needed), out: (M, N) row-major, half or float per `mode`.
//
// Block: 128 threads = 4 warps computing a 32x32 output tile (2x2 warp
// tiles of 16x16). Grid: (ceil(M/32), ceil(N/32)).
extern "C" __global__
void gemm_tc_fp16_fused(
    const half* __restrict__ A, const half* __restrict__ W, const float* __restrict__ bias,
    void* out, int M, int N, int K, int mode, int use_bias
) {
    __shared__ float acc_tile[4][TILE * TILE];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row0 = blockIdx.x * 32 + (warp / 2) * TILE;
    const int col0 = blockIdx.y * 32 + (warp % 2) * TILE;
    if (row0 >= M || col0 >= N) return;  // per-warp exit: no block-wide sync below

    wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += TILE) {
        wmma::load_matrix_sync(a_frag, A + (size_t)row0 * K + k0, K);
        wmma::load_matrix_sync(b_frag, W + (size_t)col0 * K + k0, K);  // element (k, n) = W[n*K + k]
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Accumulator fragment layout is opaque, so stage it through shared
    // memory to apply the epilogue element by element.
    float* tile = acc_tile[warp];
    wmma::store_matrix_sync(tile, c_frag, TILE, wmma::mem_row_major);
    __syncwarp();

    for (int i = lane; i < TILE * TILE; i += 32) {
        const int r = row0 + i / TILE;
        const int c = col0 + i % TILE;
        float v = tile[i];
        if (use_bias) v += bias[c];
        const size_t idx = (size_t)r * N + c;
        switch (mode) {
            case EPI_STORE_F16:
                ((half*)out)[idx] = __float2half(v);
                break;
            case EPI_BIAS_GELU_F16:  // exact (erf) GELU, as in torch.nn.GELU()
                ((half*)out)[idx] = __float2half(0.5f * v * (1.0f + erff(v * 0.70710678f)));
                break;
            case EPI_RESIDUAL_ADD_F32:
                ((float*)out)[idx] += v;
                break;
            case EPI_STORE_F32:
                ((float*)out)[idx] = v;
                break;
        }
    }
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

// One warp normalizes one row of `x` (fp32, already in memory/L1) into y16.
__device__ __forceinline__ void layernorm_row(
    const float* xr, const float* g, const float* b, half* yr, int D, float eps, int lane
) {
    float s = 0.0f;
    for (int i = lane; i < D; i += 32) s += xr[i];
    const float mean = warp_sum(s) / D;
    float s2 = 0.0f;
    for (int i = lane; i < D; i += 32) { const float d = xr[i] - mean; s2 += d * d; }
    const float rstd = rsqrtf(warp_sum(s2) / D + eps);
    for (int i = lane; i < D; i += 32) yr[i] = __float2half((xr[i] - mean) * rstd * g[i] + b[i]);
}

// y16[row] = layernorm(x32[row]) for rows [0, M). One warp per row;
// grid: ceil(M / (blockDim.x / 32)).
extern "C" __global__
void layernorm_f32_to_f16(
    const float* __restrict__ x, const float* __restrict__ g, const float* __restrict__ b,
    half* __restrict__ y, int M, int D, float eps
) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= M) return;
    layernorm_row(x + (size_t)row * D, g, b, y + (size_t)row * D, D, eps, threadIdx.x % 32);
}

// Token + position embedding into the fp32 residual stream x, fused with the
// first layernorm into y16. Rows [M, Mp) are row padding: zeroed so they stay
// finite (rows never mix in a GEMM, so padding can't affect real rows).
extern "C" __global__
void embed_layernorm(
    const long long* __restrict__ tokens, const float* __restrict__ tok_emb,
    const float* __restrict__ pos_emb, const float* __restrict__ g, const float* __restrict__ b,
    float* __restrict__ x, half* __restrict__ y, int M, int Mp, int T, int D, float eps
) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (row >= Mp) return;
    float* xr = x + (size_t)row * D;
    half* yr = y + (size_t)row * D;
    if (row >= M) {
        for (int i = lane; i < D; i += 32) { xr[i] = 0.0f; yr[i] = __float2half(0.0f); }
        return;
    }
    const float* te = tok_emb + tokens[row] * D;
    const float* pe = pos_emb + (size_t)(row % T) * D;
    for (int i = lane; i < D; i += 32) xr[i] = te[i] + pe[i];
    __syncwarp();
    layernorm_row(xr, g, b, yr, D, eps, lane);
}

// Causal self-attention for one (batch, head): scores = q k^T * scale,
// causal mask, softmax, out = attn @ v -- all in shared memory. qkv16 is
// (B*T, 3*D) laid out as [q | k | v], each split into H heads of Dh = D/H
// (PyTorch's qkv.view(B, T, 3, H, Dh)); out16 is (B*T, D).
//
// At seq_len 8 each head is an 8x8x32 problem, far too small for tensor
// cores to pay off, so this runs on CUDA cores; the win is fusing ~15 CuPy
// kernels per layer into one. Grid: B*H; block: 128;
// dynamic shared memory: (3*T*Dh + T*T) floats.
extern "C" __global__
void causal_attention(
    const half* __restrict__ qkv, half* __restrict__ out, int T, int D, int H, float scale
) {
    extern __shared__ float sm[];
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int Dh = D / H;
    float* q = sm;
    float* k = q + T * Dh;
    float* v = k + T * Dh;
    float* s = v + T * Dh;

    for (int idx = threadIdx.x; idx < T * Dh; idx += blockDim.x) {
        const int t = idx / Dh, d = idx % Dh;
        const size_t base = (size_t)(b * T + t) * 3 * D + h * Dh + d;
        q[idx] = __half2float(qkv[base]);
        k[idx] = __half2float(qkv[base + D]);
        v[idx] = __half2float(qkv[base + 2 * D]);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < T * T; idx += blockDim.x) {
        const int i = idx / T, j = idx % T;
        float dot = 0.0f;
        if (j <= i) {
            for (int d = 0; d < Dh; ++d) dot += q[i * Dh + d] * k[j * Dh + d];
        }
        s[idx] = (j <= i) ? dot * scale : -INFINITY;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < T; i += blockDim.x) {  // one thread per (short) row
        float mx = -INFINITY;
        for (int j = 0; j < T; ++j) mx = fmaxf(mx, s[i * T + j]);
        float sum = 0.0f;
        for (int j = 0; j < T; ++j) { const float e = __expf(s[i * T + j] - mx); s[i * T + j] = e; sum += e; }
        const float inv = 1.0f / sum;
        for (int j = 0; j < T; ++j) s[i * T + j] *= inv;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < T * Dh; idx += blockDim.x) {
        const int i = idx / Dh, d = idx % Dh;
        float o = 0.0f;
        for (int j = 0; j <= i; ++j) o += s[i * T + j] * v[j * Dh + d];
        out[(size_t)(b * T + i) * D + h * Dh + d] = __float2half(o);
    }
}
