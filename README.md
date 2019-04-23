# Julia Fractals

This respository contains three scripts, each script imports the config object which defines ceratin parameters of the Julia fractal we want to create.
Each script creates the Julia fractal in a different way, single CPU, parallel CPU and paralell GPU.

Files contained in this repository:

 - Config.py - Contains configuration parameters relevant to all three scripts.
 - CPUSequential.py - Single thread (process in pythons case thanks very much GIL).
 - CPUParallel.py - Multi threaded (again technically a process not a thread in python).
 - GPUParallel.py - GPU version using Numba for JIT compilation of python into machine code & using Numbas CUDA API.