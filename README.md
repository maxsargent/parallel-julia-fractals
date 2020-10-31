# Julia Fractals

This repository contains 5 scripts, each script generates a Julia fractal. We generate the fractal using a single CPU, paralell processes with multiple CPUs and then a massivley parallel processes by utilizing the GPU.

The CPU scripts have two versions - one using interpreted python and another using numba JIT compiled python.

Files contained in this repository:

 - CPUSequential.py - Single process (not a thread in pythons case thanks very much GIL).
 - CPUSequential-Numba.py - Single process but compiled with numba.
 - CPUParallel.py - Multi process (again not technically a thread in python).
 - CPUParallel-Numba.py - Multi process but compiled with numba.
 - GPU.py - GPU using numba.
