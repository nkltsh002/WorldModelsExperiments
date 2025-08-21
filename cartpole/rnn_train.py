import os
import numpy as np
import tensorflow as tf
import gym
from env import CartPoleWrapper
from vae import StateVAE
from rnn import MDNRNN

# Parameters
model_dir = "models"
vae_path = os.path.join(model_dir, "vae")
rnn_path = os.path.join(model_dir, "rnn")
num_episodes = 100
max_steps = 200
state_size = 4
z_size = 2
action_size = 2
hidden_units = 64
n_mixtures = 5
batch_size = 16
num_epochs = 30

def collect_rollouts(vae):
    """Collect rollouts from CartPole environment, encoded using the VAE"""
    print("Collecting rollouts from CartPole environment...")
    
    z_series = []
    action_series = []
    
    env = CartPoleWrapper()
    
    for episode in range(num_episodes):
        state = env.reset()
        
        # Encode the initial state
        z = vae.encode(state)[0]
        
        episode_z = [z]
        episode_actions = []
        
        for step in range(max_steps):
            # Take random action
            action = env.action_space.sample()
            action_onehot = np.zeros(action_size)
            action_onehot[action] = 1
            
            # Step the environment
            next_state, reward, done, _ = env.step(action)
            
            # Encode the next state
            next_z = vae.encode(next_state)[0]
            
            # Store the transition
            episode_z.append(next_z)
            episode_actions.append(action_onehot)
            
            if done:
                break
            
            # Update state
            state = next_state
            z = next_z
        
        # Only keep episodes with more than 1 transition
        if len(episode_z) > 1:
            z_series.append(np.array(episode_z))
            action_series.append(np.array(episode_actions))
        
        if (episode + 1) % 10 == 0:
            print(f"Collected {episode + 1}/{num_episodes} episodes")
    
    env.close()
    
    return z_series, action_series

def train_rnn(z_series, action_series):
    """Train the MDN-RNN model"""
    print("Training MDN-RNN model...")
    
    # Create directories if they don't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Create and train the RNN
    mdn_rnn = MDNRNN(z_size=z_size, action_size=action_size, hidden_units=hidden_units, 
                    n_mixtures=n_mixtures, batch_size=batch_size)
    
    mdn_rnn.train(z_series, action_series, num_epochs=num_epochs, batch_size=batch_size)
    
    # Save the model
    mdn_rnn.save_model(rnn_path)
    
    # Test prediction
    test_episode = 0
    z = z_series[test_episode][0]
    action = action_series[test_episode][0]
    next_z_actual = z_series[test_episode][1]
    next_z_pred, _ = mdn_rnn.predict_next_z(z, action)
    
    print("\nRNN Prediction Test:")
    print("Current z:", z)
    print("Action:", np.argmax(action))
    print("Actual next z:", next_z_actual)
    print("Predicted next z:", next_z_pred)
    print("Prediction error:", np.mean(np.square(next_z_actual - next_z_pred)))
    
    # Close the session
    mdn_rnn.close_session()
    
    return rnn_path

def main():
    # Load the VAE model
    vae = StateVAE(state_size=state_size, z_size=z_size)
    vae.load_model(vae_path)
    
    # Collect rollouts
    z_series, action_series = collect_rollouts(vae)
    print(f"Collected {len(z_series)} valid episodes")
    
    # Close VAE session
    vae.close_session()
    
    # Train RNN
    rnn_model_path = train_rnn(z_series, action_series)
    print(f"RNN model saved to {rnn_model_path}")

if __name__ == "__main__":
    main()
