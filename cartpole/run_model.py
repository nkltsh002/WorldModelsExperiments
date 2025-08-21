import os
import numpy as np
import tensorflow as tf
import gym
from env import CartPoleWrapper
from vae import StateVAE
from rnn import MDNRNN
from controller import Controller

# Parameters
model_dir = "models"
vae_path = os.path.join(model_dir, "vae")
rnn_path = os.path.join(model_dir, "rnn")
controller_path = os.path.join(model_dir, "controller")
state_size = 4
z_size = 2
action_size = 2
hidden_size = 16
max_steps = 1000
use_rnn_predictions = False  # Set to True to use the RNN to predict states (dreaming)

def run_world_model(render=True, use_rnn=False):
    """
    Run the World Models pipeline with a trained controller
    If use_rnn=True, use the RNN to predict states (dreaming)
    """
    print("Running World Models with a trained controller...")
    
    # Load the VAE and RNN models
    vae = StateVAE(state_size=state_size, z_size=z_size)
    vae.load_model(vae_path)
    
    rnn = MDNRNN(z_size=z_size, action_size=action_size)
    rnn.load_model(rnn_path)
    
    # Load the controller
    controller = Controller(z_size=z_size, hidden_size=hidden_size, action_size=action_size)
    controller.load_model(controller_path)
    
    # Create the environment
    env = CartPoleWrapper(render_mode=render)
    state = env.reset()
    
    # Encode the initial state
    z = vae.encode(state)[0]
    rnn_state = None
    
    # Run the episode
    total_reward = 0
    z_real_history = [z]
    z_dream_history = [z]
    action_history = []
    
    for step in range(max_steps):
        # Get action from controller
        action = controller.get_action(z, deterministic=True)
        action_onehot = np.zeros(action_size)
        action_onehot[action] = 1
        action_history.append(action)
        
        # Step the environment
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        
        if done:
            break
        
        # Encode the next state (real observation)
        next_z_real = vae.encode(next_state)[0]
        z_real_history.append(next_z_real)
        
        # Predict the next state using the RNN (dreaming)
        next_z_dream, rnn_state = rnn.predict_next_z(z, action_onehot, rnn_state)
        z_dream_history.append(next_z_dream)
        
        # Use either the real observation or the RNN prediction
        if use_rnn:
            z = next_z_dream
        else:
            z = next_z_real
        
        # Update state
        state = next_state
    
    env.close()
    
    print(f"Episode finished with reward: {total_reward}")
    print(f"Episode length: {step+1} steps")
    
    # Close sessions
    controller.close_session()
    vae.close_session()
    rnn.close_session()
    
    return {
        'total_reward': total_reward,
        'episode_length': step+1,
        'z_real_history': np.array(z_real_history),
        'z_dream_history': np.array(z_dream_history),
        'action_history': np.array(action_history)
    }

def main():
    # Create directories if they don't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Check if all models exist
    models_exist = (
        os.path.exists(vae_path + ".index") and 
        os.path.exists(rnn_path + ".index") and 
        os.path.exists(controller_path + ".index")
    )
    
    if not models_exist:
        print("Error: One or more models are missing. Please train the models first.")
        print("Run the following scripts in order:")
        print("1. python vae_train.py")
        print("2. python rnn_train.py")
        print("3. python train.py")
        return
    
    # Run with real observations
    print("Running with real observations:")
    result_real = run_world_model(render=True, use_rnn=False)
    
    # Run with RNN predictions (dreaming)
    print("\nRunning with RNN predictions (dreaming):")
    result_dream = run_world_model(render=True, use_rnn=True)
    
    # Compare results
    print("\nResults comparison:")
    print(f"Real observations - Reward: {result_real['total_reward']}, Length: {result_real['episode_length']}")
    print(f"RNN predictions  - Reward: {result_dream['total_reward']}, Length: {result_dream['episode_length']}")

if __name__ == "__main__":
    main()
