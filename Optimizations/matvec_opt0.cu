#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// CUDA kernel for matrix-vector multiplication and bias addition
__global__ void matvec_kernel(int n, int m,
		              const double* x,
			      const double* b,
			      const double* w,
			      double* a) {
    // Each thread computes one output row (dot product)
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    // Use local accumulator
    double sum = 0.0;

    // Loop calculating the matrix-vector dot product
    int j = 0;
    for (j = 0; j < m; j++) {
        sum += w[row * m + j] * x[j];
    }
    a[row] = sum + b[row];
}

int main() {
    const int n = 80000;  // rows
    const int m = 8000;   // columns
    const int nsteps = 100;

    // Allocate host memory
    double *x = (double*)malloc(m * sizeof(double));
    double *b = (double*)malloc(n * sizeof(double));
    double *a = (double*)malloc(n * sizeof(double));
    double *w = (double*)malloc((size_t)n * m * sizeof(double));

    // Initialize data
    for (int i = 0; i < m; ++i) x[i] = 2.0;
    for (int i = 0; i < n; ++i) {
        b[i] = 1.0;
        a[i] = 0.0;
        for (int j = 0; j < m; ++j)
            w[i * m + j] = 1.0;
    }

    // Device allocations
    double *d_x, *d_b, *d_w, *d_a;
    cudaMalloc(&d_x, m * sizeof(double));
    cudaMalloc(&d_b, n * sizeof(double));
    cudaMalloc(&d_w, (size_t)n * m * sizeof(double));
    cudaMalloc(&d_a, n * sizeof(double));

    // Copy data to device
    cudaMemcpy(d_x, x, m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w, (size_t)n * m * sizeof(double), cudaMemcpyHostToDevice);

    // Kernel configuration
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Timing with CUDA events (GPU-only timing)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < nsteps; ++i) {
        matvec_kernel<<<grid, block>>>(n, m, d_x, d_b, d_w, d_a);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Time taken for %d steps: %.3f sec\n", nsteps, ms / 1000.0f);

    // Copy result back
    cudaMemcpy(a, d_a, n * sizeof(double), cudaMemcpyDeviceToHost);

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

