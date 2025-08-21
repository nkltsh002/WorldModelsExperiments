import numpy as np
import gym
from gym import wrappers

def make_env(env_name, seed=-1, render_mode=False, record_data=False, data_dir=""):
    """
    Creates a CartPole-v1 environment with optional recording
    """
    env = gym.make(env_name)
    if seed >= 0:
        env.seed(seed)
    
    if record_data:
        env = wrappers.Monitor(env, data_dir, force=True)
    
    return env

class CartPoleWrapper:
    """
    Wrapper for the CartPole environment to make it compatible with World Models
    """
    def __init__(self, env_name="CartPole-v1", seed=-1, render_mode=False, record_data=False, data_dir=""):
        self.env = make_env(env_name, seed=seed, render_mode=render_mode, record_data=record_data, data_dir=data_dir)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        # Additional properties
        self.input_size = self.observation_space.shape[0]  # State dimension (4 for CartPole)
        self.action_size = self.action_space.n  # Number of actions (2 for CartPole)
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self, mode='human'):
        return self.env.render(mode=mode)
    
    def close(self):
        self.env.close()

def generate_data(num_episodes=100, max_steps=200, data_dir="data"):
    """
    Generate training data from CartPole environment
    """
    env = CartPoleWrapper(record_data=True, data_dir=data_dir)
    for episode in range(num_episodes):
        state = env.reset()
        for step in range(max_steps):
            # Take random actions
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            
            if done:
                break
            
            state = next_state
    
    env.close()
    print("Data generation complete. {} episodes saved to {}".format(num_episodes, data_dir))

if __name__ == "__main__":
    # Test the environment
    env = CartPoleWrapper(render_mode=True)
    state = env.reset()
    print("Initial state:", state)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    
    done = False
    total_reward = 0
    
    while not done:
        action = env.action_space.sample()  # Take random action
        state, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
    
    print("Episode finished with reward:", total_reward)
    env.close()
