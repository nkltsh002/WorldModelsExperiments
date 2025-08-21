# VizDoom Experiment Replication - Quick Guide

## Simulation Option
For testing purposes, we've created a simulation that demonstrates the workflow:
```powershell
powershell -ExecutionPolicy Bypass -File .\run_vizdoom_simulation.ps1
```

## Process Overview

1. **Data Collection (CPU Instance)**
   - Run `bash extract.bash` in a tmux session
   - Creates 12,800 episode recordings (.npz files) in the record directory
   - Takes ~5 hours on a 64-core machine

2. **Copy Data to GPU Instance**
   - Use `scp` to transfer the record directory to a GPU instance
   - Shutdown the CPU instance after transfer

3. **Train VAE and RNN (GPU Instance)**
   - Run `bash gpu_jobs.bash` in a tmux session
   - Sequentially trains VAE, processes data, and trains MDN-RNN
   - Takes 6-8 hours on a P100 GPU

4. **Copy and Save Models**
   - Copy model files to tf_models directory:
     - `tf_vae/vae.json` → `tf_models/vae.json`
     - `tf_rnn/rnn.json` → `tf_models/rnn.json`
     - `tf_initial_z/initial_z.json` → `tf_models/initial_z.json`
   - Commit these files to your repository
   - Shutdown the GPU instance

5. **Train Controller (CPU Instance)**
   - Start a new CPU instance and clone your repository
   - Run `python train.py` in a tmux session
   - Train for at least 200 generations (~4-5 hours)
   - Monitor with `plot_training_progress.ipynb`
   - Commit the log files to your repository
   - Shutdown the CPU instance

6. **Test Models (Local Machine)**
   - Pull the latest changes from your repository
   - Test with: `python model.py doomreal render log/doomrnn.cma.16.64.best.json`
   - Alternative testing options:
     - `python model.py doomrnn render log/doomrnn.cma.16.64.best.json` (test in generated environment)
     - `python model.py doomreal norender log/doomrnn.cma.16.64.best.json` (run 100 episodes without visualization)

## Resource Requirements
- Data Extraction & CMA-ES: 64-core CPU, 220GB RAM, 200GB storage
- VAE & RNN Training: P100 GPU, 220GB RAM, 200GB storage

## Visualization
- Use `vae_test.ipynb` to visualize VAE reconstructions
- Use `plot_training_progress.ipynb` to monitor CMA-ES training
