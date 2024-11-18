import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import time

# Kernel function for matrix multiplication
matrix_mult_kernel = """
__global__ void matmul(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float value = 0;
        for (int k = 0; k < N; k++) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}
"""

def generate_matrix(size):
    """Generate a random square matrix."""
    return np.random.random((size, size)).astype(np.float32)

def matrix_multiplication_gpu(A, B, N):
    """Perform matrix multiplication on the GPU."""
    timers = {}
    start = time.time()

    # Allocate memory on the device
    A_gpu = gpuarray.to_gpu(A)
    B_gpu = gpuarray.to_gpu(B)
    C_gpu = gpuarray.empty((N, N), np.float32)
    timers['Memory Allocation'] = time.time() - start

    # Compile kernel
    start = time.time()
    mod = SourceModule(matrix_mult_kernel)
    matmul = mod.get_function("matmul")
    timers['Kernel Compilation'] = time.time() - start

    # Launch kernel
    start = time.time()
    block_size = 16
    grid_size = (N + block_size - 1) // block_size
    block_dim = (block_size, block_size, 1)
    grid_dim = (grid_size, grid_size)
    matmul(A_gpu, B_gpu, C_gpu, np.int32(N), block=block_dim, grid=grid_dim)
    cuda.Context.synchronize()
    timers['Kernel Execution'] = time.time() - start

    # Copy results back to host
    start = time.time()
    C = C_gpu.get()
    timers['Copy Back'] = time.time() - start

    return C, timers

if __name__ == "__main__":
    sizes = [512, 2048, 10000, 20000]
    for size in sizes:
        A = generate_matrix(size)
        B = generate_matrix(size)

        start = time.time()
        C, timers = matrix_multiplication_gpu(A, B, size)
        total_time = time.time() - start

        print(f"Matrix Size: {size}x{size}")
        for step, duration in timers.items():
            print(f"{step}: {duration:.6f} s")
        print(f"Total Execution Time: {total_time:.6f} s\n")
