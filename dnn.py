import numpy as np
import pickle


def compute_softmax(x):
    """Compute softmax transformation in a numerically stable way.

    Args:
        x: logits, of the shape [d, batch]

    Returns:
        Softmax probability of the input logits, of the shape [d, batch].
    """
    out = x - np.max(x, 0, keepdims=True)
    out_exp = np.exp(out)
    exp_sum = np.sum(out_exp, 0, keepdims=True)
    probs = out_exp / exp_sum
    return probs


def compute_ce_loss(out, y, loss_mask):
    """Compute cross-entropy loss, averaged over valid training samples.

    Args:
        out: dnn output, of shape [dout, batch].
        y: integer labels, [batch].
        loss_mask: loss mask of shape [batch], 1.0 for valid sample, 0.0 for padded sample.

    Returns:
        Cross entropy loss averaged over valid samples, and gradient wrt output.
    """
    # Apply softmax to get probabilities
    probs = compute_softmax(out)

    # Select the probabilities for the true labels
    batch_size = y.shape[0]
    target_probs = probs[y, np.arange(batch_size)]

    # Compute the cross-entropy loss, applying the loss mask
    masked_loss = -np.log(target_probs) * loss_mask
    loss = np.sum(masked_loss) / np.sum(loss_mask)

    # Calculate the gradient of the loss with respect to the output
    grad = probs.copy()
    grad[y, np.arange(batch_size)] -= 1
    grad *= loss_mask  # Only mask out the invalid samples without averaging over the batch
    
    return loss, grad


class FeedForwardNetwork:

    def __init__(self, din, dout, num_hidden_layers, hidden_layer_width):
        self.weights = []
        self.biases = []

        # Input layer to the first hidden layer
        self.weights.append(np.random.uniform(-0.05, 0.05, (hidden_layer_width, din)))
        self.biases.append(np.zeros(hidden_layer_width))

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            self.weights.append(np.random.uniform(-0.05, 0.05, (hidden_layer_width, hidden_layer_width)))
            self.biases.append(np.zeros(hidden_layer_width))

        # Last hidden layer to the output layer
        self.weights.append(np.random.uniform(-0.05, 0.05, (dout, hidden_layer_width)))
        self.biases.append(np.zeros(dout))


    def forward(self, x):
        """Forward the feedforward neural network.

        Args:
            x: shape [din, batch].

        Returns:
            Output of shape [dout, batch], and a list of hidden layer activations,
            each of the shape [hidden_layer_width, batch].
        """
        activations = []
        a = x

        # Forward pass through hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(self.weights[i], a) + self.biases[i][:, np.newaxis]
            a = np.maximum(0, z)  # ReLU activation
            activations.append(a)

        # Output layer (no activation)
        out = np.dot(self.weights[-1], a) + self.biases[-1][:, np.newaxis]
        
        return out, activations


    def backward(self, x, hidden, loss_grad, loss_mask):
        """Backpropagation of feedforward neural network.

        Args:
            x: input, of shape [din, batch].
            hidden: list of hidden activations, each of the shape [hidden_layer_width, batch].
            loss_grad: gradient with respect to out, of shape [dout, batch].
            loss_mask: loss mask of shape [batch], 1.0 for valid sample, 0.0 for padded sample.

        Returns:
            Returns gradient, averaged over valid samples, with respect to weights and biases.
        """
        grad_w = []
        grad_b = []

        # Output layer gradient
        delta = loss_grad * loss_mask  # Apply loss mask directly to the gradient
        grad_w.append(np.dot(delta, hidden[-1].T) / np.sum(loss_mask))
        grad_b.append(np.sum(delta, axis=1) / np.sum(loss_mask))

        # Hidden layers gradients (backward pass)
        for i in range(len(self.weights) - 2, -1, -1):
            # Backpropagate delta through the weights of the next layer and apply ReLU derivative
            delta = np.dot(self.weights[i + 1].T, delta) * (hidden[i] > 0)
            
            # Apply the loss mask to the delta
            delta *= loss_mask

            # Calculate gradients for the current layer
            grad_w.insert(0, np.dot(delta, (x if i == 0 else hidden[i - 1]).T) / np.sum(loss_mask))
            grad_b.insert(0, np.sum(delta, axis=1) / np.sum(loss_mask))
            
        return grad_w, grad_b


    def update_model(self, w_updates, b_updates):
        """Update the weights and biases of the model.

        Args:
            w_updates: a list of updates to each weight matrix.
            b_updates: a list of updates to weight bias vector.
        """
        self.weights = [w + u for w, u in zip(self.weights, w_updates)]
        self.biases = [b + u for b, u in zip(self.biases, b_updates)]

    def predict(self, x):
        """Compute predictions on a minibath.

        Args:
            x: input, of shape [din, batch].
        
        Returns:
            The discrete model predictions and the probabilities of predicting each class.
        """
        out, _ = self.forward(x)
        probs = compute_softmax(out)
        return np.argmax(out, 0), probs

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({'weights': self.weights, 'biases': self.biases}, f)

    def restore_model(self, filename):
        with open(filename, 'rb') as f:
            loaded_dict = pickle.load(f)
            self.weights = loaded_dict['weights']
            self.biases = loaded_dict['biases']
