import pycuda.autoinit
import pycuda.driver as drv
import pycuda.compiler as compiler
import numpy as np

module = compiler.SourceModule("""
__global__ void fill(float *dest) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    dest[i] = i + 1;
}""")
fill = module.get_function("fill")

N = 200
cpu_data = np.empty(N, dtype=np.float32)
fill(drv.Out(cpu_data), block=(N,1,1), grid=(1,1,1))  
print(cpu_data)
