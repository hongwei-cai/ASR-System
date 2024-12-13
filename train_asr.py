import numpy as np
import editdistance
import dnn
import ctc
from input_generator import InputGenerator


# Define model parameters
DIN = 83 * 15  # input dimension
DOUT = 29  # output dimension (number of tokens)
NUM_HIDDEN_LAYERS = 3
HIDDEN_LAYER_WIDTH = 2048
DROPOUT_RATE = 0.5

# Instantiate the model
network = dnn.FeedForwardNetwork(DIN, DOUT, NUM_HIDDEN_LAYERS, HIDDEN_LAYER_WIDTH)

# Training parameters
INITIAL_LEARNING_RATE = 0.0005
INITIAL_BETA = 0.99
DECAY_RATE = 0.95
DECAY_EPOCHS = 50
NUM_EPOCHS = 30
BATCH_SIZE = 64
CONTEXT_LENGTH = 7
SUBSAMPLING_RATE = 3
GRADIENT_CLIP_VALUE = 5.0  # Gradient clipping value

# Input generators
train_iter = InputGenerator('train_data.json', batch_size=BATCH_SIZE, shuffle=True, context_length=CONTEXT_LENGTH, subsampling_rate=SUBSAMPLING_RATE)
valid_iter = InputGenerator('dev_data.json', batch_size=1, shuffle=False, context_length=CONTEXT_LENGTH, subsampling_rate=SUBSAMPLING_RATE)

def evaluate(model, data_iter):
    data_iter.epoch = 0
    data_iter.step_in_epoch = 0
    total_num_samples = 0.0
    total_num_errors = 0.0

    while data_iter.epoch == 0:
        batch = data_iter.next()
        for uttid, x, y in batch:
            x = np.transpose(x)
            pred_labels, _ = model.predict(x)
            total_num_samples += len(y)
            total_num_errors += editdistance.eval(pred_labels, y)

    return total_num_errors / total_num_samples

# Define helper functions for learning rate and momentum
def get_learning_rate(initial_lr, epoch, decay_rate, decay_epochs):
    return initial_lr * (decay_rate ** (epoch // decay_epochs))

def get_momentum(initial_beta, epoch, decay_rate, decay_epochs):
    return initial_beta * (decay_rate ** (epoch // decay_epochs))

# Training loop
def train():
    epoch = 0
    train_iter.epoch = 0
    train_iter.step_in_epoch = 0
    learning_rate = INITIAL_LEARNING_RATE
    beta = INITIAL_BETA
    momentum_w = [np.zeros_like(w) for w in network.weights]
    momentum_b = [np.zeros_like(b) for b in network.biases]

    while epoch < NUM_EPOCHS:
        learning_rate = get_learning_rate(INITIAL_LEARNING_RATE, epoch, DECAY_RATE, DECAY_EPOCHS)
        beta = get_momentum(INITIAL_BETA, epoch, DECAY_RATE, DECAY_EPOCHS)

        while train_iter.epoch == epoch:
            batch = train_iter.next()
            for uttid, x, y in batch:
                x = np.transpose(x)
                batch_size = x.shape[1]
                loss_mask = np.ones([batch_size], dtype=np.float32)

                out, hidden = network.forward(x)
                out = out.reshape(batch_size, -1, DOUT)  # Reshape to [batch, t_in, num_classes]
                out = np.clip(out, 1e-8, 1.0)  # Clip values to avoid log(0)
                costs = -np.log(out)  # Compute log probabilities
                _, alignment = ctc.compute_forced_alignment(costs, np.array([x.shape[1]]), np.array(y)[None, :], np.array([len(y)]))

                alignment = alignment.flatten()  # Flatten alignment to match the shape of y
                loss, loss_grad = dnn.compute_ce_loss(out.reshape(DOUT, -1), alignment, loss_mask)
                grad_w, grad_b = network.backward(x, hidden, loss_grad, loss_mask)

                # Gradient clipping
                grad_w = [np.clip(gw, -GRADIENT_CLIP_VALUE, GRADIENT_CLIP_VALUE) for gw in grad_w]
                grad_b = [np.clip(gb, -GRADIENT_CLIP_VALUE, GRADIENT_CLIP_VALUE) for gb in grad_b]

                momentum_w = [beta * mw + gw for (mw, gw) in zip(momentum_w, grad_w)]
                momentum_b = [beta * mb + gb for (mb, gb) in zip(momentum_b, grad_b)]
                w_updates = [-learning_rate * mw for mw in momentum_w]
                b_updates = [-learning_rate * mb for mb in momentum_b]
                network.update_model(w_updates, b_updates)

                # Check for NaN values
                for i, w in enumerate(network.weights):
                    if np.isnan(w).any():
                        print(f"NaN detected in weights at layer {i}")
                for i, b in enumerate(network.biases):
                    if np.isnan(b).any():
                        print(f"NaN detected in biases at layer {i}")

        if epoch % 10 == 0:
            network.save_model('asr_model.pkl')
            valid_error_rate = evaluate(network, valid_iter)
            print(f'epoch {epoch}, validation error_rate={valid_error_rate:.2f}, learning_rate={learning_rate}, momentum={beta}')
        
        epoch += 1

if __name__ == "__main__":
    train()