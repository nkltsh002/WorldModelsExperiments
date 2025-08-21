# Setting Up the Environment for VizDoom Experiments

The code for the VizDoom experiments requires specific versions of libraries that may not be compatible with the latest Python versions. Here's a step-by-step guide to setting up the proper environment:

## 1. Create a Python 3.6 or 3.7 Environment

The code was designed for older versions of TensorFlow and other libraries that are not fully compatible with Python 3.8+. It's recommended to use Python 3.6 or 3.7.

```bash
# Using conda (recommended)
conda create -n vizdoom python=3.7
conda activate vizdoom

# Or using virtualenv
python -m venv vizdoom_env
# On Windows
vizdoom_env\Scripts\activate
# On Linux/Mac
source vizdoom_env/bin/activate
```

## 2. Install Specific Versions of Dependencies

```bash
# TensorFlow 1.8.0
pip install tensorflow==1.8.0

# NumPy 1.13.3
pip install numpy==1.13.3

# Gym 0.9.4
pip install gym==0.9.4

# CMA 2.2.0
pip install cma==2.2.0

# MPI4Py 2.0.0
pip install mpi4py==2.0.0

# SciPy with imresize
pip install scipy==1.2.0

# VizDoom environment
pip install git+https://github.com/ppaquette/gym-doom.git

# Install doom-py
pip install doom-py
```

## 3. Potential OS-specific Dependencies

On Linux, you might need to install some additional dependencies for VizDoom:

```bash
# Ubuntu/Debian
sudo apt-get install build-essential zlib1g-dev libsdl2-dev libjpeg-dev nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev libopenal-dev timidity libwildmidi-dev unzip

# CentOS/RHEL
sudo yum install gcc-c++ make zlib-devel SDL2-devel libjpeg-turbo-devel nasm bzip2-devel gtk2-devel cmake git fluidsynth-devel openal-soft-devel
```

On Windows, you might need to install Visual C++ Build Tools.

## 4. Verify Installation

After installing all dependencies, verify that everything is set up correctly:

```python
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

import numpy as np
print(f"NumPy version: {np.__version__}")

import gym
print(f"Gym version: {gym.__version__}")

import cma
print(f"CMA version: {cma.__version__}")

from mpi4py import MPI
print("MPI4Py is installed")

from scipy.misc import imresize
print("scipy.misc.imresize is available")

import ppaquette_gym_doom
print("VizDoom environment is installed")

import doom_py
print("doom-py is installed")
```

## 5. Adjusting the Code

If you're still encountering issues, you might need to make some adjustments to the code:

1. Update import statements for deprecated functions
2. Fix syntax warnings (e.g., `is` vs `==` for string comparisons)
3. Adjust TensorFlow API calls for compatibility

## 6. Running the Experiments

Once the environment is set up, follow the instructions in the REPLICATION_GUIDE.md file to run the experiments.
