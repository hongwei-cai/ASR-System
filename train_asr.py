import numpy as np
import editdistance
import dnn
import ctc
from input_generator import InputGenerator


# Define model parameters
din = 83 * 15  # input dimension
dout = 29  # output dimension (number of tokens)
num_hidden_layers = 2
hidden_layer_width = 512

# Instantiate the model
network = dnn.FeedForwardNetwork(din, dout, num_hidden_layers, hidden_layer_width)

# Training parameters
num_epochs = 60
batch_size = 32
context_length = 7
subsampling_rate = 3

initial_learning_rate = 0.00003
initial_beta = 0.99
decay_rate = 0.95
decay_epochs = 5
gradient_clip_value = 0.05
weight_decay = 1e-2  # L2 regularization

# Input generators
train_iter = input_generator.InputGenerator('train_data.json', batch_size=batch_size, shuffle=True, context_length=context_length, subsampling_rate=subsampling_rate)
valid_iter = input_generator.InputGenerator('dev_data.json', batch_size=1, shuffle=False, context_length=context_length, subsampling_rate=subsampling_rate)

def evaluate(model, data_iter):
    data_iter.epoch = 0
    data_iter.step_in_epoch = 0
    total_num_samples = 0.0
    total_num_errors = 0.0
    total_loss = 0.0

    while data_iter.epoch != 1:
        batch = data_iter.next()
        for uttid, x, y in batch:
            x = np.transpose(x)
            pred_labels, _ = model.predict(x)
            total_num_samples += len(y)
            total_num_errors += editdistance.eval(pred_labels, y)
            # Compute loss for validation
            out, hidden = model.forward(x)
            out = out.reshape(1, -1, dout)  # Reshape to [batch, t_in, num_classes]
            out = np.clip(out, 1e-8, 1.0)  # Clip values to avoid log(0)
            costs = -np.log(out)  # Compute log probabilities
            _, alignment = ctc.compute_forced_alignment(costs, np.array([x.shape[1]]), np.array(y)[None, :], np.array([len(y)]))
            alignment = alignment.flatten()  # Flatten alignment to match the shape of y
            loss, _ = dnn.compute_ce_loss(out.reshape(dout, -1), alignment, np.ones([1], dtype=np.float32))
            total_loss += loss

    return total_num_errors / total_num_samples, total_loss / total_num_samples

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
    learning_rate = initial_learning_rate
    beta = initial_beta
    momentum_w = [np.zeros_like(w) for w in network.weights]
    momentum_b = [np.zeros_like(b) for b in network.biases]
    best_valid_error_rate = float('inf')
    patience = 5
    patience_counter = 0

    while train_iter.epoch < num_epochs:
        learning_rate = get_learning_rate(initial_learning_rate, epoch, decay_rate, decay_epochs)
        beta = get_momentum(initial_beta, epoch, decay_rate, decay_epochs)

        batch = train_iter.next()
        for uttid, x, y in batch:
            x = np.transpose(x)
            batch_size = x.shape[1]
            loss_mask = np.ones([batch_size], dtype=np.float32)

            # Normalize input features to the range [-1.0, 1.0]
            x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
            x = np.clip(x, -1.0, 1.0)

            # Print sample data
            # print(f"Utterance ID: {uttid}")
            # print(f"Input features shape: {x.shape}")
            # print(f"Input features sample: {x[:, :5]}")  # Print first 5 frames
            # print(f"Labels: {y}")

            # Verify data shapes
            # assert x.shape[0] == 1245, f"Unexpected number of features: {x.shape[0]}"
            # assert len(y) > 0, "Labels are empty"
            # assert all(isinstance(label, int) for label in y), "Labels contain non-integer values"

            # Verify data normalization
            # assert np.all(np.isfinite(x)), "Input features contain NaN or Inf values"
            # assert np.all(np.abs(x) <= 1.0), "Input features are not normalized"

            # Forward pass
            out, hidden = network.forward(x)
            
            # Verify forward pass outputs
            # print(f"Forward pass output shape: {out.shape}")
            # print(f"Forward pass output sample: {out[:, :5]}")  # Print first 5 outputs
            # assert np.all(np.isfinite(out)), "Forward pass output contains NaN or Inf values"
            
            # Reshape the output to [batch_size, num_frames, dout]
            out = out.reshape(batch_size, -1, dout)
            # assert out.shape[2] == dout, f"Unexpected output shape: {out.shape}"

            out = np.clip(out, 1e-8, 1.0)  # Clip values to avoid log(0)
            costs = -np.log(out)  # Compute log probabilities
            _, alignment = ctc.compute_forced_alignment(costs, np.array([x.shape[1]]), np.array(y)[None, :], np.array([len(y)]))

            alignment = alignment.flatten()  # Flatten alignment to match the shape of y
            loss, loss_grad = dnn.compute_ce_loss(out.reshape(dout, -1), alignment, loss_mask)
            
            # Debug statements to monitor loss and gradients
            # print(f"Epoch {epoch}, Utterance {uttid}, Loss: {loss}")
            # if np.isnan(loss) or np.isinf(loss):
            #     print("Loss is NaN or Inf. Skipping update.")
            #     continue

            # Add L2 regularization to the loss
            l2_loss = weight_decay * sum(np.sum(w ** 2) for w in network.weights)
            loss += l2_loss

            grad_w, grad_b = network.backward(x, hidden, loss_grad, loss_mask)

            # Debug statements to monitor gradients
            # for i, (gw, gb) in enumerate(zip(grad_w, grad_b)):
            #     print(f"Layer {i} - Grad W: {np.mean(gw)}, Grad B: {np.mean(gb)}")
            #     print(f"Layer {i} - Grad W min: {np.min(gw)}, max: {np.max(gw)}")
            #     print(f"Layer {i} - Grad B min: {np.min(gb)}, max: {np.max(gb)}")
            # Gradient clipping
            grad_w = [np.clip(gw, -gradient_clip_value, gradient_clip_value) for gw in grad_w]
            grad_b = [np.clip(gb, -gradient_clip_value, gradient_clip_value) for gb in grad_b]

            momentum_w = [beta * mw + gw for (mw, gw) in zip(momentum_w, grad_w)]
            momentum_b = [beta * mb + gb for (mb, gb) in zip(momentum_b, grad_b)]
            w_updates = [-learning_rate * mw for mw in momentum_w]
            b_updates = [-learning_rate * mb for mb in momentum_b]

            # Debug statements to monitor parameter updates
            # for i, (wu, bu) in enumerate(zip(w_updates, b_updates)):
            #     print(f"Layer {i} - Update W: {np.mean(wu)}, Update B: {np.mean(bu)}")
            #     print(f"Layer {i} - Update W min: {np.min(wu)}, max: {np.max(wu)}")
            #     print(f"Layer {i} - Update B min: {np.min(bu)}, max: {np.max(bu)}")

            network.update_model(w_updates, b_updates)

        if epoch % 10 == 0:
            network.save_model(f'asr_model_epoch_{epoch}.pkl')
            valid_error_rate, valid_loss = evaluate(network, valid_iter)
            print(f'epoch {epoch}, validation error_rate={valid_error_rate:.2f}, validation_loss={valid_loss:.2f}, learning_rate={learning_rate}, momentum={beta}')
            
            # Early stopping
            if valid_error_rate < best_valid_error_rate:
                best_valid_error_rate = valid_error_rate
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
        
        epoch += 1

if __name__ == "__main__":
    train()