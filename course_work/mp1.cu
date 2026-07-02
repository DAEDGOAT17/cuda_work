#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>

// 1. Kernel uses float so we don't lose the decimals!
__global__ void add_vectors(float* a_d, float* b_d, float* c_d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c_d[i] = a_d[i] + b_d[i];
    }
}

// 2. Changed parameters to float to match the internal allocation
__host__ int add_vectors_host(int n) {
    int N = n;
    size_t size = N * sizeof(float);

    // Clean initialization of device pointers
    float* a_d = nullptr, * b_d = nullptr, * c_d = nullptr;

    // Allocate Host Memory
    float* a_h = new float[N];
    float* b_h = new float[N];
    float* c_h = new float[N];

    // Assign decimal values
    for (int i = 0; i < N; i++) {
        a_h[i] = 1.2f;
        b_h[i] = 2.7f;
    }

    // Allocate GPU VRAM (Using your perfectly mastered typecast!)
    cudaMalloc((void**)&a_d, size);
    cudaMalloc((void**)&b_d, size);
    cudaMalloc((void**)&c_d, size);

    // Copy to GPU
    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);

    // Grid and Block dimensions setup
    dim3 gridSize(N / 256, 1, 1);
    if (N % 256 != 0) {
        gridSize.x++;
    }
    dim3 blockSize(256, 1, 1);

    // Launch Kernel
    add_vectors << <gridSize, blockSize >> > (a_d, b_d, c_d, N);

    // Copy back to CPU RAM
    cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);

    printf("GPU Calculation Complete!\n");
    // %f works perfectly now because c_h holds floats!
    printf("Result check: C_h[0] = %f\n", c_h[0]);
    printf("Result check: C_h[%d] = %f\n", N - 1, c_h[N - 1]);

    // Free VRAM (No ampersand needed!)
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    // Free CPU RAM (Use delete[] to match new[])
    delete[] a_h;
    delete[] b_h;
    delete[] c_h;

    return 0;
}

int main() {
    // Run it with 10,000 elements
    add_vectors_host(10000);
    return 0;
}