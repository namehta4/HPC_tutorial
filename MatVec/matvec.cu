#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 256

// CUDA kernel for matrix-vector multiplication and bias addition
__global__ void matvec_kernel(int n, int m, const double* x, const double* b, const double* w, double* a) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        double dot = 0.0;
        for (int j = 0; j < m; j++) {
            dot += w[row * m + j] * x[j];
        }
        a[row] = dot + b[row];
    }
}

int main() {
    int n = 80000; // rows
    int m = 500;   // columns

    // Host pointers
    double *x, *b, *a, *p;
    double *w;

    // Allocate host memory
    x = (double*)malloc(m * sizeof(double));
    b = (double*)malloc(n * sizeof(double));
    a = (double*)malloc(n * sizeof(double));
    p = (double*)malloc(n * sizeof(double)); // not needed, replaced by a
    w = (double*)malloc(n * m * sizeof(double));

    // Initialize data
    for (int i = 0; i < m; i++)
        x[i] = 2.0;

    for (int i = 0; i < n; i++) {
        b[i] = 1.0;
        a[i] = 0.0;
        for (int j = 0; j < m; j++)
            w[i * m + j] = 1.0;
    }

    // Device pointers
    double *d_x, *d_b, *d_w, *d_a;

    // Allocate device memory
    cudaMalloc((void**)&d_x, m * sizeof(double));
    cudaMalloc((void**)&d_b, n * sizeof(double));
    cudaMalloc((void**)&d_w, n * m * sizeof(double));
    cudaMalloc((void**)&d_a, n * sizeof(double));

    // Copy data to device
    cudaMemcpy(d_x, x, m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w, n * m * sizeof(double), cudaMemcpyHostToDevice);

    // Timing
    clock_t t = clock();

    // Kernel launch parameters
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int nstep = 0; nstep < 100; nstep++) {
        matvec_kernel<<<gridSize, BLOCK_SIZE>>>(n, m, d_x, d_b, d_w, d_a);
        cudaDeviceSynchronize(); // ensure kernel completes
    }

    t = clock() - t;
    double time_taken = ((double)t) / CLOCKS_PER_SEC;
    printf("Time taken: %f (sec)\n", time_taken);

    // Copy result back
    cudaMemcpy(a, d_a, n * sizeof(double), cudaMemcpyDeviceToHost);

    // Free memory
    free(x);
    free(b);
    free(a);
    free(p);
    free(w);
    cudaFree(d_x);
    cudaFree(d_b);
    cudaFree(d_w);
    cudaFree(d_a);

    return 0;
}

