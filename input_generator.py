import numpy as np
import kaldiio
import json


class CharacterTokenizer(object):
    def __init__(self):
        chars = "*ABCDEFGHIJKLMNOPQRSTUVWXYZ' "
        self.char2id = {char: i for i, char in enumerate(chars)}
        self.id2char = {i: char for i, char in enumerate(chars)}

    def StringToIds(self, string):
        return [self.char2id[char] for char in string]

    def IdsToString(self, ids):
        return ''.join([self.id2char[i] for i in ids])


def splice_and_subsample(feature_matrix, context_length, subsampling_rate):
    T, d = feature_matrix.shape

    # Initialize spliced matrix with boundary padding
    spliced = np.zeros((T, (2 * context_length + 1) * d))

    # For each frame, concatenate context frames (handle boundaries by repeating the edge frames)
    for t in range(T):
        left_context = max(0, t - context_length)  # Handle left boundary
        right_context = min(T, t + context_length + 1)  # Handle right boundary

        spliced_frames = feature_matrix[left_context:right_context]

        # Pad missing frames from left/right context (using edge values)
        if t - context_length < 0:  # Left boundary
            pad_left = context_length - t
            spliced_frames = np.pad(spliced_frames, ((pad_left, 0), (0, 0)), mode='edge')
        if t + context_length >= T:  # Right boundary
            pad_right = (t + context_length + 1) - T
            spliced_frames = np.pad(spliced_frames, ((0, pad_right), (0, 0)), mode='edge')

        # Flatten spliced frames and assign to spliced matrix
        spliced[t] = spliced_frames.flatten()

    # Subsampling: Take every `subsampling_rate`-th frame
    subsampled_features = spliced[::subsampling_rate, :]

    return subsampled_features


class InputGenerator(object):
    def __init__(self, json_path, batch_size, shuffle, context_length, subsampling_rate):
        # Load the dataset from the json file (utts key)
        with open(json_path, 'r') as f:
            self.data = json.load(f)['utts']
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.context_length = context_length
        self.subsampling_rate = subsampling_rate
        self.tokenizer = CharacterTokenizer()
        
        # Prepare for batching
        self.utterance_ids = list(self.data.keys())
        self.num_utterances = len(self.utterance_ids)
        self.utterance_indices = list(range(self.num_utterances))
        self.epoch = 0
        self.steps_per_epoch = (self.num_utterances + self.batch_size - 1) // self.batch_size
        self.total_num_steps = 0
        self.current_step = 0
        
        if self.shuffle:
            np.random.shuffle(self.utterance_indices)
    

    def next(self):
        if self.current_step >= self.steps_per_epoch:
            self.epoch += 1
            self.current_step = 0
            if self.shuffle:
                np.random.shuffle(self.utterance_indices)

        expected_total_steps = (self.num_utterances + self.batch_size - 1) // self.batch_size

        if self.total_num_steps >= expected_total_steps:
            # If the current step is beyond the expected steps, return the last batch
            return self.process_last_batch()

        # Get the batch indices
        start_idx = self.current_step * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_utterances)
        batch_indices = self.utterance_indices[start_idx:end_idx]
        
        # Prepare the batch data
        batch = []
        for idx in batch_indices:
            utt_id = self.utterance_ids[idx]
            utt_info = self.data[utt_id]
            
            # Load acoustic features (full feature string)
            feat_matrix = kaldiio.load_mat(utt_info['feat'])
            
            # Splice and subsample the features
            spliced_subsampled_features = splice_and_subsample(feat_matrix, self.context_length, self.subsampling_rate)
            
            # Tokenize transcript
            transcript = utt_info['text']
            tokenized_transcript = self.tokenizer.StringToIds(transcript)
            
            batch.append((utt_id, spliced_subsampled_features, tokenized_transcript))
        
        self.current_step += 1
        self.total_num_steps += 1
        
        return batch


    def process_last_batch(self):
        last_batch_idx = (self.num_utterances + self.batch_size - 1) // self.batch_size - 1
        start_idx = last_batch_idx * self.batch_size
        end_idx = self.num_utterances
        batch_indices = self.utterance_indices[start_idx:end_idx]

        batch = []
        for idx in batch_indices:
            utt_id = self.utterance_ids[idx]
            utt_info = self.data[utt_id]
            
            feat_matrix = kaldiio.load_mat(utt_info['feat'])
            spliced_subsampled_features = splice_and_subsample(feat_matrix, self.context_length, self.subsampling_rate)
            transcript = utt_info['text']
            tokenized_transcript = self.tokenizer.StringToIds(transcript)
            
            batch.append((utt_id, spliced_subsampled_features, tokenized_transcript))

        return batch
    

if __name__ == '__main__':
    T, d = 4, 3
    C = 2
    X = np.arange(12)
    X = np.reshape(X, (T, d))
    # print(X)

    splice_and_subsample(X, C, 1)