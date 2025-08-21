import numpy as np
import tensorflow as tf

class Controller:
    """
    Simple controller model that maps latent state (z) to actions
    For CartPole, this is a simple binary action (0 or 1)
    """
    def __init__(self, z_size=2, hidden_size=16, action_size=2, learning_rate=0.001):
        self.z_size = z_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Build the graph
        self._build_graph()
        self._init_session()
    
    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            # Placeholders
            self.z = tf.placeholder(tf.float32, shape=[None, self.z_size], name="z_input")
            self.action_target = tf.placeholder(tf.int32, shape=[None], name="action_target")
            
            # Simple neural network
            h = tf.layers.dense(self.z, self.hidden_size, activation=tf.nn.tanh, name="hidden1")
            h = tf.layers.dense(h, self.hidden_size, activation=tf.nn.tanh, name="hidden2")
            self.logits = tf.layers.dense(h, self.action_size, name="logits")
            self.action_probs = tf.nn.softmax(self.logits, name="action_probs")
            
            # Action sampling (for use during inference)
            self.action = tf.multinomial(self.logits, 1)[0, 0]
            
            # Loss function
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits, 
                    labels=self.action_target
                )
            )
            
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
    
    def get_action(self, z, deterministic=False):
        """
        Get an action for a given latent state z
        If deterministic=True, return the most probable action
        If deterministic=False, sample from the action distribution
        """
        if len(z.shape) == 1:
            z = np.expand_dims(z, axis=0)
        
        if deterministic:
            action_probs = self.sess.run(self.action_probs, feed_dict={self.z: z})
            return np.argmax(action_probs[0])
        else:
            return self.sess.run(self.action, feed_dict={self.z: z})
    
    def train(self, z_batch, action_batch, num_epochs=1, batch_size=32):
        """Train the controller on a batch of data"""
        num_samples = len(z_batch)
        
        for epoch in range(num_epochs):
            # Shuffle the data
            p = np.random.permutation(num_samples)
            z_shuffled = z_batch[p]
            action_shuffled = action_batch[p]
            
            total_loss = 0
            n_batches = 0
            
            # Train in mini-batches
            for i in range(0, num_samples, batch_size):
                batch_end = min(i + batch_size, num_samples)
                z = z_shuffled[i:batch_end]
                actions = action_shuffled[i:batch_end]
                
                feed = {self.z: z, self.action_target: actions}
                batch_loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed)
                
                total_loss += batch_loss
                n_batches += 1
            
            # Log progress
            avg_loss = total_loss / max(n_batches, 1)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    def save_model(self, model_path):
        """Save the model to the given path"""
        self.saver.save(self.sess, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load the model from the given path"""
        self.saver.restore(self.sess, model_path)
        print(f"Model loaded from {model_path}")

if __name__ == "__main__":
    # Test the controller
    z_size = 2
    action_size = 2
    
    # Create some random latent states
    z_batch = np.random.randn(100, z_size)
    action_batch = np.random.randint(0, action_size, 100)
    
    # Create the controller
    controller = Controller(z_size=z_size, action_size=action_size)
    
    # Train for a few epochs
    controller.train(z_batch, action_batch, num_epochs=5)
    
    # Test prediction
    test_z = np.random.randn(1, z_size)
    action = controller.get_action(test_z[0])
    action_prob = controller.sess.run(controller.action_probs, feed_dict={controller.z: test_z})
    
    print("Z:", test_z[0])
    print("Action:", action)
    print("Action probabilities:", action_prob[0])
    
    # Clean up
    controller.close_session()
