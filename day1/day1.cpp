#include <cuda_runtime.h>
#include <iostream>

#define N 1024

__global__ void vectorAdd(float *a, float *b, float *c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    float *a, *b, *c;
    float *dA, *dB, *dC;

    // Allocate memory on the host
    a = (float *)malloc(N * sizeof(float));
    b = (float *)malloc(N * sizeof(float));
    c = (float *)malloc(N * sizeof(float));

    // Initialize the vectors
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i;
    }

    // Allocate memory on the device
    cudaMalloc((void **)&dA, N * sizeof(float));
    cudaMalloc((void **)&dB, N * sizeof(float));
    cudaMalloc((void **)&dC, N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(dA, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, b, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel with proper configuration
    // Using 256 threads per block, and calculate required number of blocks
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Wait for kernel to complete
    cudaDeviceSynchronize();

    // Copy data from device to host
    cudaMemcpy(c, dC, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the first 10 and last 10 results to verify correctness
    std::cout << "\nFirst 10 results:" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
    }

    std::cout << "\nLast 10 results:" << std::endl;
    for (int i = N-10; i < N; i++) {
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
    }

    // Free memory
    free(a);
    free(b);
    free(c);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}
