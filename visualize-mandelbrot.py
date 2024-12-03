import numpy as np
import matplotlib.pyplot as plt

def visualize_mandelbrot(filename, output_image):
    data = np.loadtxt(filename)
    plt.imshow(data, cmap='plasma', extent=(-2, 1, -1.5, 1.5))
    plt.title('Mandelbrot Set')
    plt.xlabel('Real axis')
    plt.ylabel('Imaginary axis')
    plt.savefig(output_image, dpi=300)
    print(f"Visualization saved as {output_image}")



if __name__ == "__main__":
    visualize_mandelbrot("mandelbrot.txt", "mandelbrot-c++.png")
    visualize_mandelbrot("mandelbrot_pycuda.txt", "mandelbrot-py.png")

