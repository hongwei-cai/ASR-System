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

# Instantiate the model
network = dnn.FeedForwardNetwork(DIN, DOUT, NUM_HIDDEN_LAYERS, HIDDEN_LAYER_WIDTH)

# Input generator for validation
CONTEXT_LENGTH = 7
SUBSAMPLING_RATE = 3
valid_iter = InputGenerator('dev_data.json', batch_size=1, shuffle=False, context_length=CONTEXT_LENGTH, subsampling_rate=SUBSAMPLING_RATE)

def evaluate_checkpoint(model, data_iter):
    data_iter.epoch = 0
    data_iter.step_in_epoch = 0
    total_edit_distance = 0
    total_num_chars = 0

    while data_iter.epoch != 1:
        batch = data_iter.next()
        for uttid, x, y in batch:
            x = np.transpose(x)
            log_probs, _ = model.predict(x)
            # print(f"log_probs shape: {log_probs.shape}")
            
            # Ensure log_probs is 3D: [batch, time, num_classes]
            if log_probs.ndim == 1:
                log_probs = log_probs[np.newaxis, :, np.newaxis]
            elif log_probs.ndim == 2:
                log_probs = log_probs[np.newaxis, :, :]
            
            input_lens = np.array([log_probs.shape[1]])
            output_lens = np.array([len(y)])
            
            # Decode the utterance using beam search
            decoded_hypothesis = ctc.beam_search(log_probs, input_lens)
            hypothesis_str = data_iter.tokenizer.IdsToString(decoded_hypothesis[0])
            ground_truth_str = data_iter.tokenizer.IdsToString(y)
            
            # Compute character edit distance
            edit_distance = editdistance.eval(hypothesis_str, ground_truth_str)
            total_edit_distance += edit_distance
            total_num_chars += len(ground_truth_str)

    # Compute final character error rate
    character_error_rate = total_edit_distance / total_num_chars
    return character_error_rate

if __name__ == "__main__":
    # Load the model checkpoint
    network.restore_model('asr_model.pkl')
    
    # Evaluate the model on the dev set
    cer = evaluate_checkpoint(network, valid_iter)
    print(f'Character Error Rate (CER) on dev set: {cer:.2f}')