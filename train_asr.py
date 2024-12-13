import numpy as np
import dnn
import ctc
import input_generator
import editdistance


# Define model parameters
din = 83 * 15  # input dimension
dout = 29  # output dimension (number of tokens)
num_hidden_layers = 3
hidden_layer_width = 2048
dropout_rate = 0.5  # Add dropout rate

# Instantiate the model
network = dnn.FeedForwardNetwork(din, dout, num_hidden_layers, hidden_layer_width)

# Training parameters
initial_learning_rate = 0.01
initial_beta = 0.99
decay_rate = 0.95
decay_epochs = 50
num_epochs = 60
batch_size = 32
context_length = 7
subsampling_rate = 3
gradient_clip_value = 5.0  # Gradient clipping value

# Input generators
train_iter = input_generator.InputGenerator('train_data.json', batch_size=batch_size, shuffle=True, context_length=context_length, subsampling_rate=subsampling_rate)
valid_iter = input_generator.InputGenerator('dev_data.json', batch_size=1, shuffle=False, context_length=context_length, subsampling_rate=subsampling_rate)

def evaluate(model, data_iter):
    data_iter.epoch = 0
    data_iter.step_in_epoch = 0
    total_num_samples = 0.0
    total_num_errors = 0.0

    while data_iter.epoch != 1:
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
    learning_rate = initial_learning_rate
    beta = initial_beta
    momentum_w = [np.zeros_like(w) for w in network.weights]
    momentum_b = [np.zeros_like(b) for b in network.biases]

    while train_iter.epoch != num_epochs:
        learning_rate = get_learning_rate(initial_learning_rate, epoch, decay_rate, decay_epochs)
        beta = get_momentum(initial_beta, epoch, decay_rate, decay_epochs)

        batch = train_iter.next()
        for uttid, x, y in batch:
            x = np.transpose(x)
            batch_size = x.shape[1]
            loss_mask = np.ones([batch_size], dtype=np.float32)

            out, hidden = network.forward(x)
            out = out.reshape(batch_size, -1, dout)  # Reshape to [batch, t_in, num_classes]
            out = np.clip(out, 1e-8, 1.0)  # Clip values to avoid log(0)
            costs = -np.log(out)  # Compute log probabilities
            _, alignment = ctc.compute_forced_alignment(costs, np.array([x.shape[1]]), np.array(y)[None, :], np.array([len(y)]))

            alignment = alignment.flatten()  # Flatten alignment to match the shape of y
            loss, loss_grad = dnn.compute_ce_loss(out.reshape(dout, -1), alignment, loss_mask)
            grad_w, grad_b = network.backward(x, hidden, loss_grad, loss_mask)

            # Gradient clipping
            grad_w = [np.clip(gw, -gradient_clip_value, gradient_clip_value) for gw in grad_w]
            grad_b = [np.clip(gb, -gradient_clip_value, gradient_clip_value) for gb in grad_b]

            momentum_w = [beta * mw + gw for (mw, gw) in zip(momentum_w, grad_w)]
            momentum_b = [beta * mb + gb for (mb, gb) in zip(momentum_b, grad_b)]
            w_updates = [-learning_rate * mw for mw in momentum_w]
            b_updates = [-learning_rate * mb for mb in momentum_b]
            network.update_model(w_updates, b_updates)

        if epoch % 10 == 0:
            network.save_model('asr_model.pkl')
            valid_error_rate = evaluate(network, valid_iter)
            print(f'epoch {epoch}, validation error_rate={valid_error_rate:.2f}, learning_rate={learning_rate}, momentum={beta}')
        
        epoch += 1

if __name__ == "__main__":
    train()