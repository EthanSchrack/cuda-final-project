import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import time

mandelbrot_kernel = """
__global__ void mandelbrotKernel(int *output, int width, int height, float x_min, float x_max, float y_min, float y_max, int max_iter) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx < width && idy < height) {
        float dx = (x_max - x_min) / width;
        float dy = (y_max - y_min) / height;

        float x0 = x_min + idx * dx;
        float y0 = y_min + idy * dy;

        float x = 0.0f, y = 0.0f;
        int iter = 0;

        while (x * x + y * y < 4.0f && iter < max_iter) {
            float x_temp = x * x - y * y + x0;
            y = 2.0f * x * y + y0;
            x = x_temp;
            iter++;
        }

        output[idy * width + idx] = iter;
    }
}
"""

def mandelbrot_gpu(width, height, max_iter):
    timers = {}
    size = width * height

    # Allocate memory for the output
    start = time.time()
    output_gpu = gpuarray.zeros(size, dtype=np.int32)
    timers['Memory Allocation'] = time.time() - start

    # Compile the kernel
    start = time.time()
    mod = SourceModule(mandelbrot_kernel)
    kernel = mod.get_function("mandelbrotKernel")
    timers['Kernel Compilation'] = time.time() - start

    # Execute the kernel
    start = time.time()
    block_dim = (16, 16, 1)
    grid_dim = ((width + 15) // 16, (height + 15) // 16)
    kernel(output_gpu, np.int32(width), np.int32(height),
           np.float32(-2.0), np.float32(1.0),
           np.float32(-1.5), np.float32(1.5),
           np.int32(max_iter),
           block=block_dim, grid=grid_dim)
    cuda.Context.synchronize()
    timers['Kernel Execution'] = time.time() - start

    # Copy the results back to host
    start = time.time()
    output = output_gpu.get()
    timers['Copy Back'] = time.time() - start

    return output.reshape((height, width)), timers

if __name__ == "__main__":
    width, height, max_iter = 4096, 4096, 1000000
    result, timers = mandelbrot_gpu(width, height, max_iter)

    # Save the output for visualization
    np.savetxt("mandelbrot_pycuda.txt", result, fmt='%d')

    # Print timing
    for step, duration in timers.items():
        print(f"{step}: {duration:.6f} s")
