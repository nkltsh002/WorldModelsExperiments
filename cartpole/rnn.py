import numpy as np
import tensorflow as tf

class MDNRNN:
    """
    MDN-RNN for modeling the dynamics in the latent space
    This model takes sequences of latent vectors and actions,
    and predicts the next latent vector using a mixture of Gaussians
    """
    def __init__(self, z_size=2, action_size=2, hidden_units=64, n_mixtures=5, 
                 batch_size=32, learning_rate=0.001, grad_clip=1.0):
        self.z_size = z_size
        self.action_size = action_size
        self.hidden_units = hidden_units
        self.n_mixtures = n_mixtures
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        
        # MDN-RNN parameters
        self.n_outputs = self.n_mixtures * (2 * self.z_size + 1)  # mu, sigma, and pi for each mixture
        
        # Build the graph
        self._build_graph()
        self._init_session()
    
    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            # Placeholders
            self.x = tf.placeholder(tf.float32, shape=[None, None, self.z_size + self.action_size], name="x_input")  # [batch, seq_len, z+action]
            self.y = tf.placeholder(tf.float32, shape=[None, None, self.z_size], name="y_target")  # [batch, seq_len, z]
            self.seq_lengths = tf.placeholder(tf.int32, shape=[None], name="seq_lengths")
            
            # RNN Cell
            lstm_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_units)
            self.initial_state = lstm_cell.zero_state(tf.shape(self.x)[0], tf.float32)
            
            # RNN
            outputs, self.final_state = tf.nn.dynamic_rnn(
                lstm_cell, 
                self.x, 
                initial_state=self.initial_state,
                dtype=tf.float32,
                sequence_length=self.seq_lengths
            )
            
            # Reshape outputs for dense layer
            outputs_flat = tf.reshape(outputs, [-1, self.hidden_units])
            
            # MDN outputs
            mdn_outputs = tf.layers.dense(outputs_flat, self.n_outputs)
            
            # Reshape MDN outputs
            self.seq_len = tf.shape(self.x)[1]
            mdn_outputs = tf.reshape(mdn_outputs, [-1, self.seq_len, self.n_outputs])
            
            # Split MDN outputs into mu, sigma, and pi components
            mdn_out_pi, mdn_out_mu, mdn_out_sigma = self._split_mdn_outputs(mdn_outputs)
            
            # Apply softmax to pi values
            mdn_out_pi = tf.nn.softmax(mdn_out_pi, axis=-1)
            
            # Apply exponential to sigma to ensure it's positive
            mdn_out_sigma = tf.exp(mdn_out_sigma)
            
            # Store MDN outputs
            self.pi = mdn_out_pi
            self.mu = mdn_out_mu
            self.sigma = mdn_out_sigma
            
            # Compute loss
            y_flat = tf.reshape(self.y, [-1, self.z_size])
            self.loss = self._mdn_loss(mdn_out_pi, mdn_out_mu, mdn_out_sigma, y_flat)
            
            # Optimizer with gradient clipping
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
            self.train_op = optimizer.apply_gradients(zip(gradients, variables))
            
            # Initialize variables
            self.init = tf.global_variables_initializer()
            
            # Saver for model checkpoints
            self.saver = tf.train.Saver(tf.global_variables())
    
    def _split_mdn_outputs(self, mdn_outputs):
        """Split MDN outputs into pi, mu, and sigma components"""
        # Each mixture has: 1 pi value, z_size mu values, and z_size sigma values
        split_sizes = [self.n_mixtures, self.n_mixtures * self.z_size, self.n_mixtures * self.z_size]
        
        # Split along the last dimension
        mdn_components = tf.split(mdn_outputs, split_sizes, axis=-1)
        
        # Extract components
        mdn_out_pi = mdn_components[0]  # [batch, seq_len, n_mixtures]
        
        # Reshape mu and sigma for easier use
        mdn_out_mu = tf.reshape(mdn_components[1], [-1, self.seq_len, self.n_mixtures, self.z_size])
        mdn_out_sigma = tf.reshape(mdn_components[2], [-1, self.seq_len, self.n_mixtures, self.z_size])
        
        return mdn_out_pi, mdn_out_mu, mdn_out_sigma
    
    def _mdn_loss(self, pi, mu, sigma, y):
        """Compute the MDN loss function"""
        # Reshape y to match mu and sigma dimensions
        y = tf.reshape(y, [-1, 1, self.z_size])
        
        # Compute gaussian likelihoods for each mixture
        exponent = -0.5 * tf.square((y - mu) / sigma)
        normal = tf.exp(exponent) / (sigma * tf.sqrt(2.0 * np.pi))
        
        # Multiply along z dimension
        normal = tf.reduce_prod(normal, axis=-1)  # [batch*seq_len, n_mixtures]
        
        # Weight by pi values
        pi = tf.reshape(pi, [-1, self.n_mixtures])  # [batch*seq_len, n_mixtures]
        weighted = normal * pi  # [batch*seq_len, n_mixtures]
        
        # Sum along mixture dimension
        mixture_sum = tf.reduce_sum(weighted, axis=1)  # [batch*seq_len]
        
        # Apply log and mean
        log_likelihood = tf.log(tf.maximum(mixture_sum, 1e-10))
        return -tf.reduce_mean(log_likelihood)
    
    def _init_session(self):
        """Initialize the TensorFlow session"""
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)
    
    def close_session(self):
        """Close the TensorFlow session"""
        self.sess.close()
    
    def train(self, z_series, action_series, num_epochs=10, batch_size=None):
        """
        Train the MDN-RNN
        z_series: list of z sequences [n_episodes, episode_length, z_dim]
        action_series: list of action sequences [n_episodes, episode_length, action_dim]
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        n_episodes = len(z_series)
        
        for epoch in range(num_epochs):
            # Shuffle the episodes
            indices = np.random.permutation(n_episodes)
            
            total_loss = 0
            n_batches = 0
            
            # Process each batch
            for batch_start in range(0, n_episodes, batch_size):
                batch_end = min(batch_start + batch_size, n_episodes)
                batch_indices = indices[batch_start:batch_end]
                actual_batch_size = len(batch_indices)
                
                # Get batch sequences
                batch_z = [z_series[i] for i in batch_indices]
                batch_actions = [action_series[i] for i in batch_indices]
                
                # Prepare inputs and targets
                max_seq_len = max(len(seq) for seq in batch_z)
                seq_lengths = [len(seq) for seq in batch_z]
                
                # Create padded inputs (z and action) and targets (next z)
                padded_x = np.zeros((actual_batch_size, max_seq_len, self.z_size + self.action_size))
                padded_y = np.zeros((actual_batch_size, max_seq_len, self.z_size))
                
                for i, (z_seq, action_seq) in enumerate(zip(batch_z, batch_actions)):
                    seq_len = len(z_seq)
                    
                    # Input: concatenate z and action
                    for j in range(seq_len - 1):
                        padded_x[i, j, :self.z_size] = z_seq[j]
                        padded_x[i, j, self.z_size:] = action_seq[j]
                        
                        # Target: next z
                        padded_y[i, j] = z_seq[j + 1]
                        
                    # Adjust sequence length to account for the missing last target
                    seq_lengths[i] = seq_len - 1
                
                # Skip empty batches
                if max(seq_lengths) <= 0:
                    continue
                
                # Train on batch
                feed = {
                    self.x: padded_x,
                    self.y: padded_y,
                    self.seq_lengths: seq_lengths
                }
                
                batch_loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed)
                total_loss += batch_loss
                n_batches += 1
            
            # Log progress
            avg_loss = total_loss / max(n_batches, 1)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    def predict_next_z(self, z, action, state=None):
        """
        Predict the next z given the current z and action
        Returns: predicted z and updated state
        """
        # Prepare input
        x = np.zeros((1, 1, self.z_size + self.action_size))
        x[0, 0, :self.z_size] = z
        x[0, 0, self.z_size:] = action
        
        # Initial state
        if state is None:
            feed = {self.x: x, self.seq_lengths: [1]}
        else:
            feed = {self.x: x, self.seq_lengths: [1], self.initial_state: state}
        
        # Run prediction
        pi, mu, sigma, new_state = self.sess.run([self.pi, self.mu, self.sigma, self.final_state], feed_dict=feed)
        
        # Sample from the mixture
        next_z = self._sample_from_mixture(pi[0, 0], mu[0, 0], sigma[0, 0])
        
        return next_z, new_state
    
    def _sample_from_mixture(self, pi, mu, sigma):
        """Sample from a mixture of Gaussians"""
        # Choose a mixture component
        mixture_idx = np.random.choice(self.n_mixtures, p=pi)
        
        # Sample from the chosen Gaussian
        z_pred = np.random.normal(mu[mixture_idx], sigma[mixture_idx])
        
        return z_pred
    
    def save_model(self, model_path):
        """Save the model to the given path"""
        self.saver.save(self.sess, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load the model from the given path"""
        self.saver.restore(self.sess, model_path)
        print(f"Model loaded from {model_path}")

if __name__ == "__main__":
    # Test the MDN-RNN
    z_size = 2
    action_size = 2
    
    # Create a simple test sequence
    sequence_length = 10
    n_episodes = 5
    
    # Create random sequences for testing
    z_series = [np.random.randn(sequence_length, z_size) for _ in range(n_episodes)]
    action_series = [np.random.randint(0, 2, (sequence_length, action_size)) for _ in range(n_episodes)]
    
    # Create the MDN-RNN
    mdn_rnn = MDNRNN(z_size=z_size, action_size=action_size)
    
    # Train for a few epochs
    mdn_rnn.train(z_series, action_series, num_epochs=2)
    
    # Test prediction
    z = z_series[0][0]
    action = action_series[0][0]
    next_z, _ = mdn_rnn.predict_next_z(z, action)
    
    print("Current z:", z)
    print("Action:", action)
    print("Predicted next z:", next_z)
    
    # Clean up
    mdn_rnn.close_session()
