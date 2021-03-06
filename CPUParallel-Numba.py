import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import multiprocessing as mp
from functools import partial
from numba import jit

const_real = -.835
const_imag = -.2321

@jit
def julia(x, y, max_iterations):
    c = complex(const_real, const_imag)
    z = complex(x, y)
    for i in range(max_iterations):
        z = z**2 + c
        if (z.real**2 + z.imag**2) >= 4:
            return i

    return max_iterations

@jit
def julia_row(y, h, w, x_min, x_max, y_min, y_max, iterations):
    row = np.zeros(w)
    for x in range(w):
        real = x_min + x * (x_max - x_min) / w
        imag = y_min + y * (y_max - y_min) / h
        row[x] = julia(real, imag, iterations)
    return y, row

@jit
def fractal(x_min, x_max, y_min, y_max, image, iterations):
    h, w = image.shape

    pool = mp.Pool()

    julia_row_partial = partial(julia_row,h=h,w=w,x_min=x_min,x_max=x_max,y_min=y_min,y_max=y_max,iterations=iterations)

    for y, row_pxl in pool.map(julia_row_partial, range(h)):
        image[y] = row_pxl

def main():
    image = np.zeros((1024, 1024))

    start = timer()
    fractal(-1.5, 1.5, -1.5, 1.5, image, 50) 
    dt = timer() - start

    print("Julia created in %f s, this includes JIT compilation overheads." % dt)

    image = np.zeros((1024, 1024))

    start = timer()
    fractal(-1.5, 1.5, -1.5, 1.5, image, 50) 
    dt = timer() - start

    print("Julia created in %f s, without JIT compilation overheads." % dt)

    plt.imsave("CPUParallel-Numba.png", image)

if __name__ == "__main__":
    main()
