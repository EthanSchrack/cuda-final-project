#include <iostream>
#include <cuda_runtime.h>

__global__ void fill(float *dest, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        dest[i] = i + 1;
    }
}

int main() {
    int N = 200;
    size_t size = N * sizeof(float);

    float *h_dest = (float*)malloc(size);

    float *d_dest;
    cudaMalloc(&d_dest, size);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    fill<<<gridSize, blockSize>>>(d_dest, N);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA kernel failed: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    cudaMemcpy(h_dest, d_dest, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        std::cout << h_dest[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_dest);
    free(h_dest);

    return 0;
}
