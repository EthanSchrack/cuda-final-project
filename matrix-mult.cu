#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

__global__ void matrixMulKernel(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float value = 0.0f;
        for (int k = 0; k < N; ++k) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

void testCUDA(int matrixSize) {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    size_t size = matrixSize * matrixSize * sizeof(float);

    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    for (int i = 0; i < matrixSize * matrixSize; i++) {
        h_A[i] = rand() % 100;
        h_B[i] = rand() % 100;
    }

    auto start = std::chrono::high_resolution_clock::now();
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Memory Allocation: " << std::chrono::duration<float>(end - start).count() << " s\n";

    start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Host to Device Copy: " << std::chrono::duration<float>(end - start).count() << " s\n";

    dim3 block(16, 16);
    dim3 grid((matrixSize + block.x - 1) / block.x, (matrixSize + block.y - 1) / block.y);

    start = std::chrono::high_resolution_clock::now();
    matrixMulKernel<<<grid, block>>>(d_A, d_B, d_C, matrixSize);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Kernel Execution: " << std::chrono::duration<float>(end - start).count() << " s\n";

    start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Device to Host Copy: " << std::chrono::duration<float>(end - start).count() << " s\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}

int main() {
    int sizes[] = {512, 2048, 10000, 20000};
    for (int size : sizes) {
        std::cout << "Matrix Size: " << size << "x" << size << "\n";
        testCUDA(size);
        std::cout << "\n";
    }
    return 0;
}
