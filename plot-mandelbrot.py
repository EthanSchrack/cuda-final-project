import re
import matplotlib.pyplot as plt

# Example Mandelbrot outputs for testing (adjust with your actual logs)
python_output = """
Grid Resolution: 1080x1080
Memory Allocation: 0.101610 s
Kernel Compilation: 0.000192 s
Kernel Execution: 0.000077 s
Copy Back: 0.001729 s

Grid Resolution: 4096x4096
Memory Allocation: 0.100645 s
Kernel Compilation: 0.000184 s
Kernel Execution: 0.000248 s
Copy Back: 0.018550 s

Grid Resolution: 10000x10000
Memory Allocation: 0.106556 s
Kernel Compilation: 0.000574 s
Kernel Execution: 0.001227 s
Copy Back: 0.108348 s
"""

cpp_output = """
Grid Resolution: 1080x1080
Memory Allocation: 0.137366 s
Kernel Execution: 0.000147239 s
Device to Host Copy: 0.00249588 s

Grid Resolution: 4096x4096
Memory Allocation: 0.134118 s
Kernel Execution: 0.000320413 s
Device to Host Copy: 0.0331211 s

Grid Resolution: 10000x10000
Memory Allocation: 0.131178 s
Kernel Execution: 0.0012311 s
Device to Host Copy: 0.177968 s
"""

def parse_output(output, is_python=False):
    results = []
    resolution_pattern = r"Grid Resolution: (\d+)x\1"
    time_pattern = r"([\w\s]+): ([\d.]+) s"

    current_result = {}
    for line in output.strip().split("\n"):
        if match := re.match(resolution_pattern, line):
            if current_result:
                results.append(current_result)
            current_result = {"Grid Resolution": int(match.group(1))}
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

def plot_line_graph(python_data, cpp_data, filename="mandelbrot_comparison.png"):
    python_resolutions = [entry["Grid Resolution"] for entry in python_data]
    cpp_resolutions = [entry["Grid Resolution"] for entry in cpp_data]

    python_cpu_labels = ["Memory Allocation", "Copy Back"]
    cpp_cpu_labels = ["Memory Allocation", "Host to Device Copy", "Device to Host Copy"]

    python_cpu_times = extract_times(python_data, python_cpu_labels)
    cpp_cpu_times = extract_times(cpp_data, cpp_cpu_labels)

    python_kernel_times = extract_times(python_data, ["Kernel Execution"])
    cpp_kernel_times = extract_times(cpp_data, ["Kernel Execution"])

    # Plot line graph
    plt.figure(figsize=(10, 6))
    plt.plot(python_resolutions, python_cpu_times, label="Python CPU Time", marker="o", color="blue")
    plt.plot(python_resolutions, python_kernel_times, label="Python Kernel Time", marker="o", color="blue", alpha=0.4)
    plt.plot(cpp_resolutions, cpp_cpu_times, label="C++ CPU Time", marker="o", color="red")
    plt.plot(cpp_resolutions, cpp_kernel_times, label="C++ Kernel Time", marker="o", color="red", alpha=0.4)

    plt.xlabel("Grid Resolution")
    plt.ylabel("Execution Time (s)")
    plt.title("Execution Time Comparison: Python vs C++ Mandelbrot Set")
    plt.legend()
    plt.grid(True)

    plt.savefig(filename)
    print(f"Plot saved as {filename}")

# Parse the sample outputs
python_data = parse_output(python_output, is_python=True)
cpp_data = parse_output(cpp_output)

# Save the line graph
plot_line_graph(python_data, cpp_data, filename="mandelbrot_time_comparison.png")
