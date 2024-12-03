#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <fstream>

void saveMandelbrotToFile(int *data, int width, int height, const std::string &filename) {
    std::ofstream file(filename);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            file << data[y * width + x] << " ";
        }
        file << "\n";
    }
    file.close();
}

// Define a device function to calculate the Mandelbrot set
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

// Host function to run the Mandelbrot set computation
void computeMandelbrot(int width, int height, int max_iter) {
    int *h_output, *d_output;
    size_t size = width * height * sizeof(int);

    // Allocate host memory
    h_output = (int *)malloc(size);

    auto start = std::chrono::high_resolution_clock::now();

    // Allocate device memory
    cudaMalloc(&d_output, size);

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Memory Allocation: " << std::chrono::duration<float>(end - start).count() << " s\n";

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    start = std::chrono::high_resolution_clock::now();

    // Execute the kernel
    mandelbrotKernel<<<grid, block>>>(d_output, width, height, -2.0f, 1.0f, -1.5f, 1.5f, max_iter);
    cudaDeviceSynchronize();

    end = std::chrono::high_resolution_clock::now();
    std::cout << "Kernel Execution: " << std::chrono::duration<float>(end - start).count() << " s\n";

    start = std::chrono::high_resolution_clock::now();

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    end = std::chrono::high_resolution_clock::now();
    std::cout << "Device to Host Copy: " << std::chrono::duration<float>(end - start).count() << " s\n";

    saveMandelbrotToFile(h_output, width, height, "mandelbrot.txt");

    // Clean up memory
    cudaFree(d_output);
    free(h_output);
}

int main() {
    int width = 4096;
    int height = 4096;
    int max_iter = 1000000;

    computeMandelbrot(width, height, max_iter);

    return 0;
}
