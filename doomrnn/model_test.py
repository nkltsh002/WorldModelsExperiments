'''
Tests the trained models - SIMPLIFIED VERSION FOR TESTING
'''

import os
import numpy as np
import json

# Simulate model testing
print("Simulating model testing...")

# Define parameters
env_type = 'doomreal'  # Could be 'doomreal' or 'doomrnn'
render_mode = 'render'  # Could be 'render' or 'norender'
model_file = 'log/doomrnn.cma.16.64.best.json'

print(f"Testing model in {env_type} environment with {render_mode} mode")
print(f"Using model file: {model_file}")

# Verify that the model file exists
if os.path.exists(model_file):
    print(f"Model file exists: {model_file}")
    
    # Read the model file
    try:
        with open(model_file, 'r') as f:
            model_data = json.load(f)
        print(f"Model data: {model_data}")
    except:
        print(f"Error reading model file: {model_file}")
else:
    print(f"Model file does not exist: {model_file}")

# Simulate running 5 episodes
num_episodes = 5
total_reward = 0

for ep in range(num_episodes):
    # Simulate a random reward
    reward = np.random.randint(-100, 500)
    total_reward += reward
    print(f"Episode {ep+1}/{num_episodes}: reward = {reward}")

avg_reward = total_reward / num_episodes
print(f"Average reward over {num_episodes} episodes: {avg_reward}")
print("Model testing simulation complete")
