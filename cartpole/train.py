import os
import numpy as np
import tensorflow as tf
import gym
import json
import cma
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
num_workers = 4
num_episodes = 5
max_steps = 500
sigma_init = 0.5
popsize = 32
generations = 50

class ControllerParams:
    """Class to handle controller parameters for CMA-ES"""
    def __init__(self, controller):
        self.controller = controller
        self.param_count = self._count_parameters()
        self.session = self.controller.sess
        self.shapes = self._get_parameter_shapes()
        self.params = self._get_parameters()
    
    def _count_parameters(self):
        """Count the total number of parameters in the controller"""
        total_params = 0
        for var in tf.trainable_variables():
            if var.name.startswith("hidden") or var.name.startswith("logits"):
                total_params += np.prod(var.shape.as_list())
        return total_params
    
    def _get_parameter_shapes(self):
        """Get the shapes of all trainable parameters"""
        shapes = []
        for var in tf.trainable_variables():
            if var.name.startswith("hidden") or var.name.startswith("logits"):
                shapes.append(var.shape.as_list())
        return shapes
    
    def _get_parameters(self):
        """Get the current parameters as a flat array"""
        params = []
        for var in tf.trainable_variables():
            if var.name.startswith("hidden") or var.name.startswith("logits"):
                params.append(self.session.run(var).flatten())
        return np.concatenate(params)
    
    def set_parameters(self, flat_params):
        """Set the parameters from a flat array"""
        idx = 0
        for var in tf.trainable_variables():
            if var.name.startswith("hidden") or var.name.startswith("logits"):
                shape = var.shape.as_list()
                size = np.prod(shape)
                param = flat_params[idx:idx+size].reshape(shape)
                self.session.run(var.assign(param))
                idx += size

def evaluate_controller(controller, vae, rnn, num_episodes=5, max_steps=500, render=False):
    """
    Evaluate the controller's performance in the environment
    Returns the average reward across episodes
    """
    env = CartPoleWrapper(render_mode=render)
    rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        z = vae.encode(state)[0]
        rnn_state = None
        
        total_reward = 0
        
        for step in range(max_steps):
            # Get action from controller
            action = controller.get_action(z, deterministic=True)
            
            # One-hot encode the action
            action_onehot = np.zeros(action_size)
            action_onehot[action] = 1
            
            # Step the environment
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            if done:
                break
            
            # Encode the next state
            next_z = vae.encode(next_state)[0]
            
            # Use the RNN to predict the next state
            # This is the hallucination/dreaming part in the World Models paper
            # For now, we're using the actual observation, but you could replace
            # this with RNN predictions for a more complete implementation
            z = next_z
            
            # Update state
            state = next_state
        
        rewards.append(total_reward)
    
    env.close()
    
    # Return the average reward
    return np.mean(rewards)

def train_controller():
    """Train the controller using CMA-ES"""
    print("Training controller using CMA-ES...")
    
    # Create directories if they don't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Load the VAE and RNN models
    vae = StateVAE(state_size=state_size, z_size=z_size)
    vae.load_model(vae_path)
    
    rnn = MDNRNN(z_size=z_size, action_size=action_size)
    rnn.load_model(rnn_path)
    
    # Create the controller
    controller = Controller(z_size=z_size, hidden_size=hidden_size, action_size=action_size)
    params = ControllerParams(controller)
    
    # CMA-ES optimizer
    es = cma.CMAEvolutionStrategy(
        params.params,
        sigma_init,
        {'popsize': popsize}
    )
    
    best_reward = -np.inf
    best_params = None
    history = []
    
    # CMA-ES optimization loop
    for generation in range(generations):
        # Sample solutions
        solutions = es.ask()
        
        # Evaluate solutions
        fitness = []
        for i, solution in enumerate(solutions):
            params.set_parameters(solution)
            reward = evaluate_controller(controller, vae, rnn, num_episodes=num_episodes, max_steps=max_steps)
            fitness.append(-reward)  # CMA-ES minimizes, so negate the reward
            
            print(f"Generation {generation+1}/{generations}, Individual {i+1}/{popsize}, Reward: {-fitness[-1]:.2f}")
        
        # Update CMA-ES
        es.tell(solutions, fitness)
        
        # Check for new best
        min_idx = np.argmin(fitness)
        if -fitness[min_idx] > best_reward:
            best_reward = -fitness[min_idx]
            best_params = solutions[min_idx]
            
            # Save the best parameters
            params.set_parameters(best_params)
            controller.save_model(controller_path)
            
            print(f"New best reward: {best_reward:.2f}")
        
        # Log progress
        history.append({
            'generation': generation,
            'mean_reward': -np.mean(fitness),
            'max_reward': -np.min(fitness),
            'min_reward': -np.max(fitness),
            'best_overall': best_reward
        })
        
        print(f"Generation {generation+1}/{generations}, Mean Reward: {-np.mean(fitness):.2f}, Best Reward: {best_reward:.2f}")
    
    # Save training history
    with open(os.path.join(model_dir, "controller_training_history.json"), "w") as f:
        json.dump(history, f)
    
    # Load the best parameters and evaluate
    params.set_parameters(best_params)
    final_reward = evaluate_controller(controller, vae, rnn, num_episodes=10, max_steps=max_steps, render=True)
    print(f"Final evaluation reward: {final_reward:.2f}")
    
    # Close sessions
    controller.close_session()
    vae.close_session()
    rnn.close_session()
    
    return controller_path

def main():
    # Train the controller
    controller_model_path = train_controller()
    print(f"Controller model saved to {controller_model_path}")

if __name__ == "__main__":
    main()
