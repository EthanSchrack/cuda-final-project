import re
import matplotlib.pyplot as plt
import numpy as np

python_output = """
Matrix Size: 512x512
Memory Allocation: 0.000729 s
Kernel Compilation: 0.004624 s
Kernel Execution: 0.000689 s
Copy Back: 0.000494 s
Total Execution Time: 0.006882 s

Matrix Size: 2048x2048
Memory Allocation: 0.004945 s
Kernel Compilation: 0.000291 s
Kernel Execution: 0.007703 s
Copy Back: 0.004877 s
Total Execution Time: 0.018876 s

Matrix Size: 10000x10000
Memory Allocation: 0.097418 s
Kernel Compilation: 0.000301 s
Kernel Execution: 0.875046 s
Copy Back: 0.108501 s
Total Execution Time: 1.086022 s

Matrix Size: 20000x20000
Memory Allocation: 0.386830 s
Kernel Compilation: 0.000377 s
Kernel Execution: 7.539496 s
Copy Back: 0.425728 s
Total Execution Time: 8.358756 s
"""

cpp_output = """
Matrix Size: 512x512
Memory Allocation: 0.0949833 s
Host to Device Copy: 0.000455696 s
Kernel Execution: 0.000269111 s
Device to Host Copy: 0.000618965 s

Matrix Size: 2048x2048
Memory Allocation: 0.000300451 s
Host to Device Copy: 0.00462181 s
Kernel Execution: 0.0077151 s
Device to Host Copy: 0.00807563 s

Matrix Size: 10000x10000
Memory Allocation: 0.000368068 s
Host to Device Copy: 0.106347 s
Kernel Execution: 0.861423 s
Device to Host Copy: 0.185446 s

Matrix Size: 20000x20000
Memory Allocation: 0.000615265 s
Host to Device Copy: 0.423981 s
Kernel Execution: 7.31416 s
Device to Host Copy: 0.742126 s

"""

def parse_output(output, is_python=False):
    results = []
    matrix_size_pattern = r"Matrix Size: (\d+)x\1"
    time_pattern = r"([\w\s]+): ([\d.]+) s"

    current_result = {}
    for line in output.strip().split("\n"):
        if match := re.match(matrix_size_pattern, line):
            if current_result:
                results.append(current_result)
            current_result = {"Matrix Size": int(match.group(1))}
        elif match := re.match(time_pattern, line):
            key, value = match.groups()
            current_result[key.strip()] = float(value)
    if current_result:
        results.append(current_result)
    return results

def extract_times(data, labels):
    times = []
    for entry in data:
        times.append(sum(entry.get(label, 0) for label in labels))
    return times

def plot_line_graph(python_data, cpp_data, filename="line_graph.png"):
    python_sizes = [entry["Matrix Size"] for entry in python_data]
    cpp_sizes = [entry["Matrix Size"] for entry in cpp_data]

    python_cpu_labels = ["Memory Allocation", "Kernel Compilation", "Copy Back"]
    cpp_cpu_labels = ["Memory Allocation", "Host to Device Copy", "Device to Host Copy"]
    python_cpu_times = extract_times(python_data, python_cpu_labels)
    cpp_cpu_times = extract_times(cpp_data, cpp_cpu_labels)

    python_kernel_times = extract_times(python_data, ["Kernel Execution"])
    cpp_kernel_times = extract_times(cpp_data, ["Kernel Execution"])

    plt.figure(figsize=(10, 6))
    plt.plot(python_sizes, python_cpu_times, label="Python CPU Time", marker="o", color="blue")
    plt.plot(python_sizes, python_kernel_times, label="Python Kernel Time", marker="o", color="blue", alpha=0.4)
    plt.plot(cpp_sizes, cpp_cpu_times, label="C++ CPU Time", marker="o", color="red")
    plt.plot(cpp_sizes, cpp_kernel_times, label="C++ Kernel Time", marker="o", color="red", alpha=0.4)

    plt.xlabel("Matrix Size")
    plt.ylabel("Execution Time (s)")
    plt.title("Execution Time Comparison: Python vs C++")
    plt.legend()
    plt.grid(True)

    plt.savefig(filename)
    print(f"Plot saved as {filename}")

python_data = parse_output(python_output, is_python=True)
cpp_data = parse_output(cpp_output)

plot_line_graph(python_data, cpp_data, filename="execution_time_comparison.png")
