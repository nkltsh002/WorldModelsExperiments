# VizDoom Experiment Replication Guide

This guide will help you replicate the VizDoom experiment from the World Models paper.

## Environment Setup

The code requires specific versions of libraries that may not be compatible with the latest Python versions. Before proceeding, make sure to set up the correct environment:

1. **Using Conda (recommended)**:
   ```bash
   conda env create -f environment.yml
   conda activate vizdoom
   ```

2. **Using Pip**:
   ```bash
   # Create a Python 3.7 virtual environment
   python -m venv vizdoom_env
   # On Windows
   vizdoom_env\Scripts\activate
   # On Linux/Mac
   source vizdoom_env/bin/activate

   # Install dependencies
   pip install -r requirements.txt
   ```

For detailed instructions on setting up the environment, see [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md).

## Simulation Scripts
For testing purposes, we've created simplified simulation scripts that demonstrate the workflow:
- `extract_test.py`: Simulates the data extraction process
- `vae_train_test.py`: Simulates VAE training
- `series_test.py`: Simulates series data processing
- `rnn_train_test.py`: Simulates RNN training
- `train_test.py`: Simulates CMA-ES training
- `model_test.py`: Simulates model testing
- `run_vizdoom_simulation.ps1`: Runs all the simulation scripts in sequence

You can run the entire simulation with:
```powershell
powershell -ExecutionPolicy Bypass -File .\run_vizdoom_simulation.ps1
```

## System Requirements
The actual process requires significant compute resources:
- 64-core CPU instance with ~200GB storage and 220GB RAM for data extraction and CMA-ES training
- GPU instance (P100) with ~200GB storage and 220GB RAM for VAE and RNN training

## Time Estimates
The complete process on these instances would take approximately:
- Data extraction: ~5 hours
- VAE and RNN training: 6-8 hours
- CMA-ES training: 4-5 hours (for 200 generations)

## Step-by-Step Instructions

### Step 1: Data Extraction (on 64-core CPU instance)
This step generates 12,800 episode recordings (.npz files) using random policy:

```bash
# Create a tmux session to run in the background
tmux new -s extraction

# Run the extraction script (launches 64 parallel processes)
bash extract.bash

# This will take several hours to complete
# You can detach from the tmux session with Ctrl+B, D
```

### Step 2: Transfer Data to GPU Instance
After extraction, transfer the generated .npz files to a GPU instance:

```bash
# Using scp (if both instances are in the same region, this should be fast)
scp -r doomrnn/record/* user@gpu-instance:~/WorldModelsExperiments/doomrnn/record/

# After copying, you can shut down the CPU instance
```

### Step 3: Train VAE and RNN (on GPU instance)
This step trains the VAE, processes the data, and trains the MDN-RNN:

```bash
# Create a tmux session for GPU training
tmux new -s gpu_training

# Run the GPU jobs script
bash gpu_jobs.bash

# This will sequentially:
# 1. Train the VAE (saves to tf_vae/vae.json)
# 2. Process the data using the trained VAE (saves to series/)
# 3. Train the MDN-RNN (saves to tf_rnn/rnn.json and tf_initial_z/initial_z.json)

# This will take 6-8 hours to complete
# You can detach from the tmux session with Ctrl+B, D
```

### Step 4: Copy Trained Models
After training, copy the model files to the tf_models directory:

```bash
# On the GPU instance
mkdir -p tf_models
cp tf_vae/vae.json tf_models/
cp tf_rnn/rnn.json tf_models/
cp tf_initial_z/initial_z.json tf_models/

# Commit the changes to your git repository
git add doomrnn/tf_models/*.json
git commit -m "Add trained VizDoom models"
git push

# After this, you can shut down the GPU instance
```

### Step 5: Train Controller using CMA-ES (back on 64-core CPU instance)
This step trains the controller inside the generated environment using CMA-ES:

```bash
# Start a new 64-core CPU instance and clone your repository

# Create a tmux session for CMA-ES training
tmux new -s cma_training

# Run the CMA-ES training
python train.py

# Monitor progress using the plot_training_progress.ipynb notebook
# Let it run for at least 200 generations (4-5 hours)
# You can stop the training with Ctrl+C when satisfied

# Commit the log files to your repository
git add log/*.json
git commit -m "Add VizDoom CMA-ES training logs"
git push

# After this, you can shut down the CPU instance
```

### Step 6: Test Trained Models
After all training is complete, test the models on your local machine:

```bash
# Pull the latest changes from your repository
git pull

# Test the model in the actual environment with visualization
python model.py doomreal render log/doomrnn.cma.16.64.best.json

# Other testing options:
# Test in the generated environment
python model.py doomrnn render log/doomrnn.cma.16.64.best.json

# Run 100 episodes without visualization
python model.py doomreal norender log/doomrnn.cma.16.64.best.json
```

## Additional Resources
For detailed visualization and analysis, use the following notebooks:
- `vae_test.ipynb`: Visualize input/reconstruction images using the trained VAE
- `plot_training_progress.ipynb`: Monitor CMA-ES training progress

## Note on Dependencies
The code requires specific dependency versions:
- TensorFlow 1.8.0
- NumPy 1.13.3
- OpenAI Gym 0.9.4
- CMA 2.2.0
- Python 3
- gym-doom (https://github.com/ppaquette/gym-doom)
- mpi4py 2
