'''
Test script for VizDoom Docker environment
'''

import os
import sys
import numpy as np
import tensorflow as tf

print("Python version:", sys.version)
print("NumPy version:", np.__version__)
print("TensorFlow version:", tf.__version__)

try:
    import gym
    print("Gym version:", gym.__version__)
except ImportError:
    print("Gym not installed")

try:
    import cma
    print("CMA version:", cma.__version__)
except ImportError:
    print("CMA not installed")

try:
    from mpi4py import MPI
    print("MPI4Py is installed")
except ImportError:
    print("MPI4Py not installed")

try:
    import scipy
    print("SciPy version:", scipy.__version__)
    try:
        from scipy.misc import imresize
        print("scipy.misc.imresize is available")
    except ImportError:
        print("scipy.misc.imresize is not available")
except ImportError:
    print("SciPy not installed")

try:
    import ppaquette_gym_doom
    print("VizDoom environment is installed")
    
    # Try to create the environment
    try:
        env = gym.make('ppaquette/DoomTakeCover-v0')
        print("Successfully created VizDoom environment")
        env.close()
    except Exception as e:
        print("Error creating VizDoom environment:", str(e))
        
except ImportError:
    print("VizDoom environment not installed")

# Check if we have the model files
if os.path.exists('/app/tf_models/vae.json'):
    print("VAE model file exists")
else:
    print("VAE model file missing")

if os.path.exists('/app/tf_models/rnn.json'):
    print("RNN model file exists")
else:
    print("RNN model file missing")

if os.path.exists('/app/tf_models/initial_z.json'):
    print("initial_z file exists")
else:
    print("initial_z file missing")

print("\nVizDoom Docker environment test completed!")
