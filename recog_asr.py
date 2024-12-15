import numpy as np
import editdistance
from dnn import FeedForwardNetwork, compute_ce_loss
from ctc import beam_search, compute_forced_alignment
from input_generator import InputGenerator
from utils import pad_sequences

# Define model parameters
DIN = 83 * 15
DOUT = 29
NUM_HIDDEN_LAYERS = 3
HIDDEN_LAYER_WIDTH = 2048

# Instantiate the model
network = FeedForwardNetwork(DIN, DOUT, NUM_HIDDEN_LAYERS, HIDDEN_LAYER_WIDTH)

# Input generator for validation
valid_iter = InputGenerator(
    'dev_data.json', batch_size=1, shuffle=False, context_length=7, subsampling_rate=3)


def evaluate_checkpoint_with_loss(model, data_iter):
    """
    Evaluate the model on the validation dataset and compute CER and average loss.
    Args:
        model: The trained FeedForwardNetwork model.
        data_iter: InputGenerator for validation data.
    Returns:
        character_error_rate: Character Error Rate (CER) as a float.
        average_loss: Average cross-entropy loss across all samples.
    """
    data_iter.epoch = 0
    data_iter.step_in_epoch = 0
    total_edit_distance = 0
    total_num_chars = 0
    total_loss = 0.0

    while data_iter.epoch != 1:
        batch = data_iter.next()
        inputs, targets = [], []

        for uttid, x, y in batch:
            inputs.append(x)
            targets.append(y)

        # Pad inputs and transpose for the model
        padded_inputs, input_lens = pad_sequences(inputs)
        batch_size, max_input_len, features = padded_inputs.shape
        padded_inputs = np.transpose(padded_inputs, (0, 2, 1)).reshape(features, -1)

        # Forward pass
        out, _ = model.forward(padded_inputs)
        reshaped_out = out.T.reshape(batch_size, max_input_len, DOUT)
        log_probs = np.log(np.clip(reshaped_out, 1e-8, 1.0))

        # Compute CER
        decoded_hypotheses = []
        for i in range(batch_size):
            decoded_hypotheses.append(beam_search(log_probs[i:i+1], input_lens[i:i+1]))

        # Compute CER and Loss using forced alignment
        for i in range(batch_size):
            y = targets[i]
            costs = -log_probs[i:i+1]  # Retain batch dimension for costs
            _, alignment = compute_forced_alignment(
                costs, input_lens[i:i+1], np.array(y)[None, :], np.array([len(y)])
            )
            alignment = alignment.flatten()
            
            # Create a loss mask matching the flattened output
            loss_mask = np.ones_like(alignment, dtype=np.float32)

            # Flatten the log probabilities for loss computation
            flattened_probs = log_probs[i].reshape(-1, DOUT)

            # Compute loss
            loss, _ = compute_ce_loss(flattened_probs.T, alignment, loss_mask)
            total_loss += loss

            # CER computation
            hypothesis_str = data_iter.tokenizer.IdsToString(decoded_hypotheses[i][0])
            ground_truth_str = data_iter.tokenizer.IdsToString(y)
            edit_distance = editdistance.eval(hypothesis_str, ground_truth_str)
            total_edit_distance += edit_distance
            total_num_chars += len(ground_truth_str)

    # Compute final CER and average loss
    character_error_rate = total_edit_distance / total_num_chars if total_num_chars > 0 else 0.0
    average_loss = total_loss / total_num_chars if total_num_chars > 0 else 0.0
    return character_error_rate, average_loss

if __name__ == "__main__":
    # Load the model checkpoint
    network.restore_model('asr_model.pkl')
    
    # Evaluate the model on the dev set
    cer, avg_loss = evaluate_checkpoint_with_loss(network, valid_iter)
    print(f'Character Error Rate (CER) on dev set: {cer:.2%}, Average loss: {avg_loss:.4f}')