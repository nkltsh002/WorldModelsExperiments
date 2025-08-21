'''
Trains controller using CMA-ES - SIMPLIFIED VERSION FOR TESTING
'''

import os
import numpy as np
import json

# Simulate CMA-ES training
print("Simulating CMA-ES training...")

# Create necessary directories
log_dir = 'log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Define parameters
env_name = 'doomrnn'
optimizer = 'cma'
num_rollouts = 16
popsize = 64

# Simulate creating the log files
file_base = f"{env_name}.{optimizer}.{num_rollouts}.{popsize}"

# Generate dummy training history data
num_generations = 5  # Just simulate 5 generations for testing
hist_data = []
hist_best_data = []
best_reward = -1000

for gen in range(num_generations):
    # Simulate improving rewards over generations
    reward = -500 + gen * 100
    if reward > best_reward:
        best_reward = reward
    
    hist_data.append(reward)
    hist_best_data.append(best_reward)
    
    print(f"Generation {gen+1}/{num_generations}: reward = {reward}, best = {best_reward}")

# Save the history files
hist_file = os.path.join(log_dir, f"{file_base}.hist.json")
with open(hist_file, 'w') as f:
    json.dump(hist_data, f)

hist_best_file = os.path.join(log_dir, f"{file_base}.hist_best.json")
with open(hist_best_file, 'w') as f:
    json.dump(hist_best_data, f)

# Save a dummy best model
best_file = os.path.join(log_dir, f"{file_base}.best.json")
with open(best_file, 'w') as f:
    f.write('{"message": "This is a dummy best model for testing purposes"}')

print(f"Training history saved to {hist_file}")
print(f"Best reward history saved to {hist_best_file}")
print(f"Best model saved to {best_file}")
print("CMA-ES training simulation complete")
