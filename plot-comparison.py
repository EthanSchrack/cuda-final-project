import matplotlib.pyplot as plt

# Manually input times for Python and C++
# Each element in the list corresponds to a matrix size (e.g., 128, 256, 512)
matrix_sizes = [128, 512, 1024]

# Python CPU times (sum of allocation, kernel compilation, and copy back)
python_cpu = [0.006779, 0.001221, 0.002612]  # Example data in seconds

# Python Kernel times (execution only)
python_kernel = [0.000257, 0.000180, 0.001013]  # Example data in seconds

# C++ CPU times (sum of allocation, host-to-device copy, and device-to-host copy)
cpp_cpu = [0.0878, 0.0010804, 0.0034106]  # Example data in seconds

# C++ Kernel times (execution only)
cpp_kernel = [0.000121668, 0.000221761, 0.0010596]  # Example data in seconds

# Plot line graph
plt.figure(figsize=(10, 6))
plt.plot(matrix_sizes, python_cpu, label="Python CPU Time", marker="o")
plt.plot(matrix_sizes, python_kernel, label="Python Kernel Time", marker="o")
plt.plot(matrix_sizes, cpp_cpu, label="C++ CPU Time", marker="o")
plt.plot(matrix_sizes, cpp_kernel, label="C++ Kernel Time", marker="o")

# Labels, title, and legend
plt.xlabel("Matrix Size")
plt.ylabel("Execution Time (s)")
plt.title("Execution Time Comparison: Python vs C++")
plt.legend()
plt.grid(True)

# Save the plot locally
output_filename = "execution_time_comparison.png"
plt.savefig(output_filename)
print(f"Plot saved as {output_filename}")