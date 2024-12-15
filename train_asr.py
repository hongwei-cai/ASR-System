import numpy as np
from dnn import FeedForwardNetwork, compute_ce_loss
from ctc import compute_softmax, compute_forced_alignment
from input_generator import InputGenerator
from recog_asr import evaluate_checkpoint_with_loss
from utils import pad_sequences

# Define model parameters
DIN = 83 * 15
DOUT = 29
NUM_HIDDEN_LAYERS = 3
HIDDEN_LAYER_WIDTH = 2048

# Instantiate the model
network = FeedForwardNetwork(DIN, DOUT, NUM_HIDDEN_LAYERS, HIDDEN_LAYER_WIDTH)

try:
    with open("asr_model.pkl",'rb') as f:
        network.restore_model("asr_model.pkl")
        print("Model loaded. Continuing training.")
except FileNotFoundError:
    print("Model not found. Starting from scratch.")

# Training parameters
NUM_EPOCHS = 60
BATCH_SIZE = 16

INITIAL_LEARNING_RATE = 0.005
INITIAL_BETA = 0.999
DECAY_RATE = 0.99
DECAY_EPOCHS = 9
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 1e-6
PATIENCE = 10

# Input generators
train_iter = InputGenerator('train_data.json', batch_size=BATCH_SIZE, shuffle=True, context_length=7, subsampling_rate=3)
valid_iter = InputGenerator('dev_data.json', batch_size=1, shuffle=False, context_length=7, subsampling_rate=3)

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
    best_valid_error_rate = float('inf')
    patience_counter = 0

    while train_iter.epoch < NUM_EPOCHS:
        learning_rate = get_learning_rate(INITIAL_LEARNING_RATE, epoch, DECAY_RATE, DECAY_EPOCHS)
        beta = get_momentum(INITIAL_BETA, epoch, DECAY_RATE, DECAY_EPOCHS)

        batch = train_iter.next()
        uttids, inputs, targets = zip(*batch)
        
        # Pad inputs and targets
        padded_inputs, input_lengths = pad_sequences(inputs)
        padded_targets, target_lengths = pad_sequences(targets)

        for i in range(len(uttids)):
            x = np.transpose(padded_inputs[i])
            y = padded_targets[i]
            batch_size = x.shape[1]
            loss_mask = np.ones([batch_size], dtype=np.float32)

            # Check input data for NaNs
            if np.isnan(x).any():
                print("Input contains NaNs. Skipping this batch.")
                continue

            # Forward pass
            out, hidden = network.forward(x)
            
            # Check forward output for NaNs
            if np.isnan(out).any():
                print("Forward output contains NaNs. Skipping this batch.")
                continue

            out = out.reshape(batch_size, -1, DOUT)
            assert out.shape[2] == DOUT, f"Unexpected output shape: {out.shape}"
            print(f"Forward output contains NaNs: {np.isnan(out).any()}, Infs: {np.isinf(out).any()}")

            out = compute_softmax(out)
            out = np.clip(out, 1e-8, 1.0)

            costs = -np.log(out)
            _, alignment = compute_forced_alignment(costs, np.array([x.shape[1]]), np.array(y)[None, :], np.array([len(y)]))

            alignment = alignment.flatten()  # Flatten alignment to match the shape of y
            loss, loss_grad = compute_ce_loss(out.reshape(DOUT, -1), alignment, loss_mask)
            l2_loss = WEIGHT_DECAY * sum(np.sum(w ** 2) for w in network.weights)
            loss += l2_loss

            grad_w, grad_b = network.backward(x, hidden, loss_grad, loss_mask)

            # Gradient clipping
            grad_w = [np.clip(gw, -MAX_GRAD_NORM, MAX_GRAD_NORM) for gw in grad_w]
            grad_b = [np.clip(gb, -MAX_GRAD_NORM, MAX_GRAD_NORM) for gb in grad_b]

            momentum_w = [beta * mw + gw for (mw, gw) in zip(momentum_w, grad_w)]
            momentum_b = [beta * mb + gb for (mb, gb) in zip(momentum_b, grad_b)]
            w_updates = [-learning_rate * mw for mw in momentum_w]
            b_updates = [-learning_rate * mb for mb in momentum_b]

            network.update_model(w_updates, b_updates)

        if epoch % 10 == 0:
            network.save_model(f'asr_model_epoch_{epoch}.pkl')
            valid_error_rate, valid_loss = evaluate_checkpoint_with_loss(network, valid_iter)
            print(f'epoch {epoch}, Validation CER: {valid_error_rate:.2%}, Validation Loss: {valid_loss:.4f}, learning_rate={learning_rate}, momentum={beta}')
            
            # Early stopping
            if valid_error_rate < best_valid_error_rate:
                best_valid_error_rate = valid_error_rate
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print("Early stopping triggered")
                    break
        
        epoch += 1

if __name__ == "__main__":
    train()