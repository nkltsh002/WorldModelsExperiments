# World Models with CartPole

This is a simplified implementation of the World Models architecture described in the paper ["World Models"](https://worldmodels.github.io/) by Ha and Schmidhuber (2018), adapted for the CartPole environment.

## Overview

The World Models architecture consists of three main components:

1. **Vision (VAE)**: A variational autoencoder that compresses high-dimensional observations into a low-dimensional latent space.
2. **Memory (MDN-RNN)**: A recurrent neural network with a mixture density network output that models the dynamics of the environment in the latent space.
3. **Controller**: A simple neural network that maps latent states to actions, trained using evolutionary strategies.

In the original paper, the authors applied World Models to complex environments like VizDoom and CarRacing. This implementation simplifies the approach for the CartPole environment, which has a much simpler observation space (a 4D vector).

## File Structure

- `env.py`: Environment wrapper for CartPole
- `vae.py`: VAE model for encoding and decoding states
- `rnn.py`: MDN-RNN model for predicting future latent states
- `controller.py`: Controller model for mapping latent states to actions
- `vae_train.py`: Script to train the VAE model
- `rnn_train.py`: Script to train the MDN-RNN model
- `train.py`: Script to train the controller using CMA-ES
- `run_model.py`: Script to run the full World Models pipeline

## Training Pipeline

The training process consists of three stages:

1. **Train the VAE**: The VAE learns to encode the CartPole state into a 2D latent space.
2. **Train the RNN**: The RNN learns to predict the next latent state given the current latent state and action.
3. **Train the Controller**: The controller is trained using CMA-ES to maximize the reward in the environment.

## Usage

1. **Train the VAE**:
   ```bash
   python vae_train.py
   ```

2. **Train the RNN**:
   ```bash
   python rnn_train.py
   ```

3. **Train the Controller**:
   ```bash
   python train.py
   ```

4. **Run the Full Model**:
   ```bash
   python run_model.py
   ```

## Differences from Original Implementation

This implementation differs from the original World Models paper in several ways:

1. **Simpler Environment**: CartPole has a 4D state space, compared to the high-dimensional pixel observations in VizDoom and CarRacing.
2. **MLP-based VAE**: Since the state is already low-dimensional, we use a simple MLP-based VAE instead of a convolutional one.
3. **No Temperature Parameter**: The original implementation used a temperature parameter to control the stochasticity of the MDN-RNN predictions. This implementation uses the default temperature of 1.0.
4. **Direct State Encoding**: We directly encode the state vector, rather than rendering and encoding pixel observations.

## Extensions and Future Work

This implementation can be extended in several ways:

1. **Render to Pixels**: Render the CartPole environment to pixels and use a convolutional VAE, similar to the original paper.
2. **Temperature Parameter**: Add a temperature parameter to control the stochasticity of the MDN-RNN predictions.
3. **Dream Training**: Train the controller purely in the dream environment (using only the RNN for state predictions).
4. **Apply to More Complex Environments**: Extend this implementation to more complex environments like LunarLander or MountainCar.

## Requirements

- TensorFlow 1.x
- NumPy
- Gym
- CMA-ES

## References

- Ha, D., & Schmidhuber, J. (2018). World models. arXiv preprint arXiv:1803.10122.
- [World Models Website](https://worldmodels.github.io/)
