'''
Check dependencies for VizDoom experiment
'''

import sys
print(f"Python version: {sys.version}")

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("TensorFlow not installed")

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError:
    print("NumPy not installed")

try:
    import gym
    print(f"Gym version: {gym.__version__}")
except ImportError:
    print("Gym not installed")

try:
    import cma
    print(f"CMA version: {cma.__version__}")
except ImportError:
    print("CMA not installed")

try:
    from mpi4py import MPI
    print("MPI4Py is installed")
except ImportError:
    print("MPI4Py not installed")

try:
    import scipy
    print(f"SciPy version: {scipy.__version__}")
    try:
        from scipy.misc import imresize
        print("scipy.misc.imresize is available")
    except ImportError:
        print("scipy.misc.imresize is not available")
except ImportError:
    print("SciPy not installed")

try:
    import ppaquette_gym_doom
    print("VizDoom environment (ppaquette_gym_doom) is installed")
except ImportError:
    print("VizDoom environment (ppaquette_gym_doom) not installed")

try:
    import doom_py
    print("doom-py is installed")
except ImportError:
    print("doom-py not installed")

print("\nRecommendation:")
print("Based on the experiment requirements, you need:")
print("- Python 3.6 or 3.7")
print("- TensorFlow 1.8.0")
print("- NumPy 1.13.3")
print("- Gym 0.9.4")
print("- CMA 2.2.0")
print("- MPI4Py 2.0.0")
print("- SciPy with imresize (1.2.0)")
print("- VizDoom environment (ppaquette_gym_doom)")
print("- doom-py")
print("\nSee ENVIRONMENT_SETUP.md for detailed instructions on setting up the environment.")
