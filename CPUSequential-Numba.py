import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from numba import autojit

const_real = -.835
const_imag = -.2321

@autojit
def julia(real, imag, max_iterations):
    c = complex(const_real, const_imag)
    z = complex(real, imag)
    for i in range(max_iterations):
        z = z**2 + c
        if (z.real**2 + z.imag**2) >= 4:
            return i

    return max_iterations

@autojit
def fractal(x_min, x_max, y_min, y_max, image, iterations):
    h,w = image.shape

    for x in range(w):
        for y in range(h):
            real = x_min + x * ((x_max - x_min) / w)
            imag = y_min + y * ((y_max - y_min) / h)
            image[y, x] = julia(real, imag, iterations)


def main():
    image = np.zeros((1024, 1024), dtype = np.uint8)

    start = timer()
    fractal(-1.5, 1.5, -1.5, 1.5, image, 50) 
    dt = timer() - start

    print("Julia created in %f s, this includes JIT compilation overheads." % dt)

    image = np.zeros((1024, 1024))

    start = timer()
    fractal(-1.5, 1.5, -1.5, 1.5, image, 50) 
    dt = timer() - start

    print("Julia created in %f s, without JIT compilation overheads." % dt)

    plt.imsave(os.path.basename(sys.argv[0][:-3]), image)

if __name__ == "__main__":
    main()