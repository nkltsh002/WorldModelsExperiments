'''
saves ~ 2 episodes generated from a random policy (reduced version for testing in Docker)
'''

import numpy as np
import random
import os
import sys
import time

# Print environment information
print("Running extract_small_docker.py")
print("Python version:", sys.version)
print("NumPy version:", np.__version__)

try:
    import gym
    print("Gym version:", gym.__version__)
except ImportError:
    print("Gym not installed")
    sys.exit(1)

try:
    import ppaquette_gym_doom
    print("VizDoom environment is installed")
except ImportError:
    print("VizDoom environment not installed")
    sys.exit(1)

# Create mock data for testing
MAX_FRAMES = 100  # reduced for quick testing
MAX_TRIALS = 2    # just 2 trials for testing
MIN_LENGTH = 10   # reduced for testing

print(f"Will generate {MAX_TRIALS} trials with up to {MAX_FRAMES} frames each")

# Create record directory
DIR_NAME = 'record'
if not os.path.exists(DIR_NAME):
    os.makedirs(DIR_NAME)
    print(f"Created directory: {DIR_NAME}")

# Try to create the environment
try:
    env = gym.make('ppaquette/DoomTakeCover-v0')
    print("Successfully created VizDoom environment")
except Exception as e:
    print("Error creating VizDoom environment:", str(e))
    # Create mock data instead
    print("Will generate mock data instead")
    for trial in range(MAX_TRIALS):
        random_id = random.randint(0, 2**31-1)
        filename = os.path.join(DIR_NAME, f"{random_id}.npz")
        
        # Create mock observations and actions
        obs = np.random.randint(0, 255, size=(MAX_FRAMES, 64, 64, 3), dtype=np.uint8)
        actions = np.random.random(MAX_FRAMES).astype(np.float16) * 2 - 1
        
        print(f"Saving mock data to {filename}")
        np.savez_compressed(filename, obs=obs, action=actions)
    
    print("Mock data generation complete")
    sys.exit(0)

# If we get here, we can use the real environment
total_frames = 0

for trial in range(MAX_TRIALS):
    try:
        random_generated_int = random.randint(0, 2**31-1)
        filename = os.path.join(DIR_NAME, f"{random_generated_int}.npz")
        recording_obs = []
        recording_action = []

        np.random.seed(random_generated_int)
        env.seed(random_generated_int)

        # Random policy parameters
        repeat = np.random.randint(1, 11)

        obs = env.reset()
        
        for frame in range(MAX_FRAMES):
            if frame % repeat == 0:
                action = np.random.rand() * 2.0 - 1.0
                repeat = np.random.randint(1, 11)
            
            # Convert to the action space expected by the environment
            doom_action = 0
            if action > 0:
                doom_action = 1
            
            # VizDoom expects a list of actions
            action_list = [0] * env.action_space.n
            action_list[0] = doom_action  # Move right
            
            # Get the current observation (screen)
            screen = env.render(mode='rgb_array')
            if screen is None:
                screen = np.zeros((64, 64, 3), dtype=np.uint8)
            else:
                # Resize to 64x64
                from scipy.misc import imresize
                screen = imresize(screen, (64, 64, 3))
            
            recording_obs.append(screen)
            recording_action.append(action)
            
            obs, reward, done, info = env.step(action_list)
            
            if done:
                break
                
        total_frames += frame
        print(f"Trial {trial+1}/{MAX_TRIALS}: dead at frame {frame}, total recorded frames: {total_frames}")
        
        recording_obs = np.array(recording_obs, dtype=np.uint8)
        recording_action = np.array(recording_action, dtype=np.float16)
        
        if len(recording_obs) > MIN_LENGTH:
            print(f"Saving data to {filename}")
            np.savez_compressed(filename, obs=recording_obs, action=recording_action)
        else:
            print(f"Trial too short ({len(recording_obs)} frames), not saving")
            
    except Exception as e:
        print(f"Error in trial {trial+1}: {str(e)}")
        env.close()
        env = gym.make('ppaquette/DoomTakeCover-v0')
        continue

env.close()
print("Extraction complete!")
