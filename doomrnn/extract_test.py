'''
saves ~ 200 episodes generated from a random policy - MODIFIED FOR TESTING
'''

import numpy as np
import random
import os
import config
import gym
# Comment out for testing
# from doomreal import _process_frame
# from env import make_env
# from model import make_model

MAX_FRAMES = 2100 # from doomtakecover
MAX_TRIALS = 2 # MODIFIED for testing, original: 200
MIN_LENGTH = 100

render_mode = False # for debugging.

DIR_NAME = 'record'
if not os.path.exists(DIR_NAME):
    os.makedirs(DIR_NAME)

# Comment out for testing
# model = make_model(config.games['doomreal'])
# total_frames = 0
# model.make_env(render_mode=render_mode, load_model=False) # random weights

print("Testing extract.py with limited functionality")
print(f"Would generate {MAX_TRIALS} trials")
print(f"Each trial would generate up to {MAX_FRAMES} frames")
print(f"Output would be saved to the {DIR_NAME} directory")

# Simulating episode generation
for trial in range(MAX_TRIALS):
    print(f"Generating trial {trial+1}/{MAX_TRIALS}")
    # Create a dummy npz file for testing
    filename = os.path.join(DIR_NAME, "record_test_%03d.npz" % (trial))
    dummy_data = {"dummy": np.zeros((10, 10))}
    np.savez_compressed(filename, **dummy_data)
    print(f"  Saved {filename}")

print("Extraction test complete")
