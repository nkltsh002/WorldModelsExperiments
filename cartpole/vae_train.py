import os
import numpy as np
import tensorflow as tf
import gym
from env import CartPoleWrapper, generate_data
from vae import StateVAE

# Parameters
data_dir = "cartpole_data"
model_dir = "models"
vae_path = os.path.join(model_dir, "vae")
num_episodes = 200
max_steps = 200
state_size = 4
z_size = 2
batch_size = 64
num_epochs = 50

def collect_data():
    """Collect data from CartPole environment"""
    print("Collecting data from CartPole environment...")
    
    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate data
    generate_data(num_episodes=num_episodes, max_steps=max_steps, data_dir=data_dir)
    
    # Load and preprocess the data
    states = []
    actions = []
    
    env = CartPoleWrapper()
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_states = [state]
        episode_actions = []
        
        for step in range(max_steps):
            action = env.action_space.sample()  # Random action
            next_state, reward, done, _ = env.step(action)
            
            episode_states.append(next_state)
            episode_actions.append(action)
            
            if done:
                break
            
            state = next_state
        
        states.extend(episode_states)
        actions.extend(episode_actions)
    
    env.close()
    
    return np.array(states), np.array(actions)

def train_vae(states):
    """Train the VAE model"""
    print("Training VAE model...")
    
    # Create directories if they don't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Create and train the VAE
    vae = StateVAE(state_size=state_size, z_size=z_size, batch_size=batch_size)
    vae.train(states, num_epochs=num_epochs, batch_size=batch_size)
    
    # Save the model
    vae.save_model(vae_path)
    
    # Test reconstruction
    sample_idx = np.random.randint(0, len(states))
    original_state = states[sample_idx]
    encoded_z = vae.encode(original_state)
    decoded_state = vae.decode(encoded_z)[0]
    
    print("\nVAE Reconstruction Test:")
    print("Original state:", original_state)
    print("Encoded z:", encoded_z[0])
    print("Decoded state:", decoded_state)
    print("Reconstruction error:", np.mean(np.square(original_state - decoded_state)))
    
    # Close the session
    vae.close_session()
    
    return vae_path

def main():
    # Collect data
    states, actions = collect_data()
    print(f"Collected {len(states)} states and {len(actions)} actions")
    
    # Train VAE
    vae_model_path = train_vae(states)
    print(f"VAE model saved to {vae_model_path}")

if __name__ == "__main__":
    main()
