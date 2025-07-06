#include<stdio.h>
#define BLOCK_SIZE 4

__global__ void matmul(float *A, float *B, float *C, int size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= size || y >= size) return;

    float sum = 0.0f;
    for(int i = 0; i < size; ++i){
        sum += A[y * size + i] * B[i * size + x];
    }
    C[y * size + x] = sum;
}

int main(){
    // initialize values
    int mat_size = 4;
    float *A, *B, *C, *dA, *dB, *dC;

    // memory allocation
    A = (float*)malloc(mat_size*mat_size*sizeof(float));
    B = (float*)malloc(mat_size*mat_size*sizeof(float));
    C = (float*)malloc(mat_size*mat_size*sizeof(float));

    // Initialize matrices
    for(int i=0; i<mat_size*mat_size; ++i){
        A[i] = i;
        B[i] = i;
        C[i] = 0.0f;  // Initialize C to zero
    }

    cudaMalloc(&dA, mat_size*mat_size*sizeof(float));
    cudaMalloc(&dB, mat_size*mat_size*sizeof(float));
    cudaMalloc(&dC, mat_size*mat_size*sizeof(float));

    // copy - Fix: copy B to dB, not A to dB
    cudaMemcpy(dA, A, mat_size*mat_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, mat_size*mat_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, mat_size*mat_size*sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((mat_size + BLOCK_SIZE - 1) / BLOCK_SIZE, (mat_size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // kernel
    matmul<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC, mat_size);

    // Wait for kernel to complete
    cudaDeviceSynchronize();

    // copy result back
    cudaMemcpy(C, dC, mat_size*mat_size*sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print first 10 results
    printf("First 10 results:\n");
    for(int i=0; i<10; ++i){
        printf("C[%d] = %f\n", i, C[i]);
    }
    
    // Verify a few results manually
    printf("\nVerification (first element):\n");
    float expected = 0.0f;
    for(int k = 0; k < mat_size; ++k) {
        expected += A[0 * mat_size + k] * B[k * mat_size + 0];
    }
    printf("Expected C[0] = %f, Got C[0] = %f\n", expected, C[0]);
    
    // free
    free(A);
    free(B);
    free(C);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}
