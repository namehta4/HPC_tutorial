#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define TILE_SIZE 256  // number of elements of x loaded into shared memory at once

// CUDA kernel: matrix-vector multiplication with bias addition using shared memory tiling
__global__ void matvec_kernel_tiled(int n, int m,
                                    const float* __restrict__ x,
                                    const float* __restrict__ b,
                                    const float* __restrict__ w,
                                    float* __restrict__ a) {
    extern __shared__ float x_tile[];  // shared memory buffer for a tile of x

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    // Use local accumulator
    float sum = 0.0f;

    // Loop over tiles of x
    for (int tile_start = 0; tile_start < m; tile_start += TILE_SIZE) {
        int tile_size = min(TILE_SIZE, m - tile_start);

        // Load a tile of x into shared memory (coalesced)
        if (threadIdx.x < tile_size) {
            x_tile[threadIdx.x] = x[tile_start + threadIdx.x];
        }
        __syncthreads();

        // Each thread processes part of its row using the shared x_tile
        const float* w_row = &w[row * m + tile_start];
        for (int j = 0; j < tile_size; ++j) {
            sum += w_row[j] * x_tile[j];
        }
        __syncthreads();
    }
    a[row] = sum + b[row];
}

int main() {
    const int n = 80000;  // rows
    const int m = 8000;   // columns
    const int nsteps = 100;

    // Host allocations
    float *x = (float*)malloc(m * sizeof(float));
    float *b = (float*)malloc(n * sizeof(float));
    float *a = (float*)malloc(n * sizeof(float));
    float *w = (float*)malloc((size_t)n * m * sizeof(float));

    // Initialize data
    for (int i = 0; i < m; ++i) x[i] = 2.0f;
    for (int i = 0; i < n; ++i) {
        b[i] = 1.0f;
        a[i] = 0.0f;
        for (int j = 0; j < m; ++j)
            w[i * m + j] = 1.0f;
    }

    // Device allocations
    float *d_x, *d_b, *d_w, *d_a;
    cudaMalloc(&d_x, m * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_w, (size_t)n * m * sizeof(float));
    cudaMalloc(&d_a, n * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_x, x, m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w, (size_t)n * m * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel configuration
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    size_t shared_bytes = TILE_SIZE * sizeof(float);  // shared memory per block

    // Timing with CUDA events (GPU-only timing)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < nsteps; ++i) {
        matvec_kernel_tiled<<<grid, block, shared_bytes>>>(n, m, d_x, d_b, d_w, d_a);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Time taken for %d steps: %.3f sec\n", nsteps, ms / 1000.0f);

    // Copy result back
    cudaMemcpy(a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_x);
    cudaFree(d_b);
    cudaFree(d_w);
    cudaFree(d_a);
    free(x);
    free(b);
    free(a);
    free(w);

    return 0;
}

