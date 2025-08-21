import numpy as np
import tensorflow as tf

class StateVAE:
    """
    VAE for encoding and decoding CartPole state vectors
    
    Since CartPole states are already low-dimensional (4D vectors),
    we use a simple MLP architecture instead of a convolutional one.
    """
    def __init__(self, state_size=4, z_size=2, batch_size=64, learning_rate=0.001, kl_tolerance=0.5):
        self.state_size = state_size
        self.z_size = z_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.kl_tolerance = kl_tolerance
        
        # Create the VAE model
        self._build_graph()
        self._init_session()
        
    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            # Placeholders
            self.state_input = tf.placeholder(tf.float32, shape=[None, self.state_size], name="state_input")
            
            # Encoder network
            h_enc = tf.layers.dense(self.state_input, 20, activation=tf.nn.relu, name="enc_dense1")
            h_enc = tf.layers.dense(h_enc, 10, activation=tf.nn.relu, name="enc_dense2")
            
            # VAE latent space parameters
            self.mu = tf.layers.dense(h_enc, self.z_size, name="enc_mu")
            self.logvar = tf.layers.dense(h_enc, self.z_size, name="enc_logvar")
            self.sigma = tf.exp(self.logvar / 2.0)
            
            # Sampling
            epsilon = tf.random_normal([tf.shape(self.state_input)[0], self.z_size])
            self.z = self.mu + self.sigma * epsilon
            
            # Decoder network
            h_dec = tf.layers.dense(self.z, 10, activation=tf.nn.relu, name="dec_dense1")
            h_dec = tf.layers.dense(h_dec, 20, activation=tf.nn.relu, name="dec_dense2")
            self.state_output = tf.layers.dense(h_dec, self.state_size, name="dec_output")
            
            # Loss function
            # Reconstruction loss (mean squared error)
            self.r_loss = tf.reduce_mean(
                tf.square(self.state_input - self.state_output), 
                name="r_loss"
            )
            
            # KL divergence loss
            self.kl_loss = -0.5 * tf.reduce_mean(
                (1 + self.logvar - tf.square(self.mu) - tf.exp(self.logvar)),
                name="kl_loss"
            )
            
            # Apply KL tolerance
            self.kl_tolerance_value = self.kl_tolerance * self.z_size
            self.kl_loss_adjusted = tf.maximum(self.kl_loss - self.kl_tolerance_value, 0.0)
            
            # Total loss
            self.loss = self.r_loss + self.kl_loss_adjusted
            
            # Optimizer
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)
            
            # Initialize variables
            self.init = tf.global_variables_initializer()
            
            # Saver for model checkpoints
            self.saver = tf.train.Saver(tf.global_variables())
    
    def _init_session(self):
        """Initialize the TensorFlow session"""
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)
    
    def close_session(self):
        """Close the TensorFlow session"""
        self.sess.close()
    
    def encode(self, state):
        """Encode state to latent representation z"""
        if len(state.shape) == 1:  # Single state
            state = np.expand_dims(state, axis=0)
        
        return self.sess.run(self.z, feed_dict={self.state_input: state})
    
    def decode(self, z):
        """Decode latent representation z to state"""
        if len(z.shape) == 1:  # Single z
            z = np.expand_dims(z, axis=0)
        
        return self.sess.run(self.state_output, feed_dict={self.z: z})
    
    def train(self, states, num_epochs=10, batch_size=None):
        """Train the VAE on a batch of states"""
        if batch_size is None:
            batch_size = self.batch_size
        
        num_samples = states.shape[0]
        
        for epoch in range(num_epochs):
            # Shuffle the data
            p = np.random.permutation(num_samples)
            states_shuffled = states[p]
            
            # Train in mini-batches
            for i in range(0, num_samples, batch_size):
                batch_end = min(i + batch_size, num_samples)
                batch = states_shuffled[i:batch_end]
                
                feed = {self.state_input: batch}
                self.sess.run(self.train_op, feed_dict=feed)
            
            # Log progress for each epoch
            r_loss, kl_loss, total_loss = self.sess.run(
                [self.r_loss, self.kl_loss, self.loss],
                feed_dict={self.state_input: states}
            )
            
            print(f"Epoch {epoch+1}/{num_epochs}, r_loss: {r_loss:.4f}, kl_loss: {kl_loss:.4f}, total_loss: {total_loss:.4f}")
    
    def save_model(self, model_path):
        """Save the model to the given path"""
        self.saver.save(self.sess, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load the model from the given path"""
        self.saver.restore(self.sess, model_path)
        print(f"Model loaded from {model_path}")

if __name__ == "__main__":
    # Test the VAE
    state_size = 4  # CartPole state size
    z_size = 2      # Latent dimension
    
    # Create some random states for testing
    test_states = np.random.randn(100, state_size)
    
    # Create the VAE
    vae = StateVAE(state_size=state_size, z_size=z_size)
    
    # Train for a few epochs
    vae.train(test_states, num_epochs=5)
    
    # Test encoding and decoding
    original_state = test_states[0]
    encoded_z = vae.encode(original_state)
    decoded_state = vae.decode(encoded_z)[0]
    
    print("Original state:", original_state)
    print("Encoded z:", encoded_z[0])
    print("Decoded state:", decoded_state)
    print("Reconstruction error:", np.mean(np.square(original_state - decoded_state)))
    
    # Clean up
    vae.close_session()
