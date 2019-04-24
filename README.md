# Julia Fractals

This respository contains three scripts, each script imports the config object which defines ceratin parameters of the Julia fractal we want to create.
Each script creates the Julia fractal in a different way, single CPU, parallel CPU and paralell GPU.

Files contained in this repository:

 - CPUSequential.py - Single process (not a thread in pythons case thanks very much GIL).
 - CPUSequential-Numba.py - Single process but compiled with numba.
 - CPUParallel.py - Multi process (again not technically a thread in python).
 - CPUParallel-Numba.py - Multi process but compiled with numba.
 - GPU.py - GPU using numba.