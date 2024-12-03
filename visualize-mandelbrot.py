import numpy as np
import matplotlib.pyplot as plt

def visualize_mandelbrot(filename, output_image):
    data = np.loadtxt(filename)
    plt.imshow(data, cmap='inferno', extent=(-2, 1, -1.5, 1.5))
    plt.colorbar(label='Iterations')
    plt.title('Mandelbrot Set')
    plt.xlabel('Real axis')
    plt.ylabel('Imaginary axis')
    plt.savefig(output_image, dpi=300)
    print(f"Visualization saved as {output_image}")

def visualize_mandelbrot_continuous(filename, output_image, max_iterations):
    """Visualize the Mandelbrot set with improved coloring."""
    # Load data from the file
    data = np.loadtxt(filename)

    # Normalize iteration counts for better color mapping
    normalized_data = data / max_iterations

    # Apply a perceptually uniform colormap
    plt.imshow(normalized_data, cmap='viridis', extent=(-2, 1, -1.5, 1.5))
    plt.colorbar(label='Normalized Iterations')
    plt.title('Mandelbrot Set (Improved Coloring)')
    plt.xlabel('Real axis')
    plt.ylabel('Imaginary axis')

    # Save the figure to a file
    plt.savefig(output_image, dpi=300)
    print(f"Improved Mandelbrot visualization saved as {output_image}")


if __name__ == "__main__":
    max_iters = 1000
    visualize_mandelbrot_continuous("mandelbrot.txt", "mandelbrot-c++.png", max_iters)
    visualize_mandelbrot("mandelbrot_pycuda.txt", "mandelbrot-py.png")
    # visualize_mandelbrot_continuous("mandelbrot_pycuda.txt", "mandelbrot-py.png", max_iters)
