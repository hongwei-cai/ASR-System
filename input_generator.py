import numpy as np
import kaldiio
import json


class CharacterTokenizer(object):
    def __init__(self):
        self.char2id = {}
        self.id2char = {}
        for i, char in enumerate("*ABCDEFGHIJKLMNOPQRSTUVWXYZ' "):
            self.char2id[char] = i
            self.id2char[i] = char

    def StringToIds(self, string):
        return [self.char2id[char] for char in string]

    def IdsToString(self, ids):
        return ''.join([self.id2char[i] for i in ids])


def splice_and_subsample(feature_matrix, context_length, subsampling_rate) -> np.ndarray:

    T, d = feature_matrix.shape

    # Splicing
    spliced_features = np.pad(feature_matrix, ((context_length, context_length), (0, 0)), mode='edge')
    spliced = []
    for i in range(context_length, T + context_length):
        frame = spliced_features[i - context_length : i + context_length + 1].reshape(-1)  # Concatenate context frames
        spliced.append(frame)
    spliced = np.array(spliced)
    # spliced_features = np.concatenate([np.tile(feature_matrix[0], (context_length, 1)), feature_matrix, np.tile(feature_matrix[-1], (context_length, 1))], axis=0)
    
    # Subsampling
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