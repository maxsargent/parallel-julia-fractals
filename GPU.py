import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from numba import cuda
from numba import *

os.environ["NUMBAPRO_NVVM"] = "/usr/local/cuda-10.0/nvvm/lib64/libnvvm.so"
os.environ["NUMBAPRO_LIBDEVICE"] = "/usr/local/cuda-10.0/nvvm/libdevice/"

const_real = -.835
const_imag = -.2321

@cuda.jit(device=True)
def julia(real, imag, max_iterations):
    c = complex(const_real, const_imag)
    z = complex(real, imag)
    for i in range(max_iterations):
        z = z**2 + c
        if (z.real**2 + z.imag**2) >= 4:
            return i

    return max_iterations

@cuda.jit
def julia_kernel(x_min, x_max, y_min, y_max, image, iterations):
    h,w = image.shape

    x_start, y_start = cuda.grid(2)
    x_grid = cuda.gridDim.x * cuda.blockDim.x;
    y_grid = cuda.gridDim.y * cuda.blockDim.y;

    for x in range(x_start, w, x_grid):
        for y in range(y_start, h, y_grid):
            real = x_min + x * (x_max - x_min) / w
            imag = y_min + y * (y_max - y_min) / h
            image[y, x] = julia(real, imag, iterations)

def main():
    image = np.zeros((1024, 1024), dtype = np.uint8)
    blockdim = (32, 8)
    griddim = (32,16)

    start = timer()
    device_image = cuda.to_device(image)
    julia_kernel[griddim, blockdim](-1.5, 1.5, -1.5, 1.5, device_image, 50) 
    device_image.to_host()
    dt = timer() - start

    print("Julia created in %f s, this includes JIT compilation overheads." % dt)

    image = np.zeros((1024, 1024), dtype = np.uint8)
    blockdim = (32, 8)
    griddim = (32,16)

    start = timer()
    device_image = cuda.to_device(image)
    julia_kernel[griddim, blockdim](-1.5, 1.5, -1.5, 1.5, device_image, 50) 
    device_image.to_host()
    dt = timer() - start

    print("Julia created in %f s, without JIT compilation overheads." % dt)

    plt.imsave(os.path.basename(sys.argv[0][:-3]), image)

if __name__ == "__main__":
	main()