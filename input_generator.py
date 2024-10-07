import numpy as np
import kaldiio
import json


class CharacterTokenizer(object):
    def __init__(self):
        chars = "0ABCDEFGHIJKLMNOPQRSTUVWXYZ' "
        self.char2id = {char: i for i, char in enumerate(chars)}
        self.id2char = {i: char for i, char in enumerate(chars)}

    def StringToIds(self, string):
        return [self.char2id[char] for char in string]

    def IdsToString(self, ids):
        return ''.join([self.id2char[i] for i in ids])


def splice_and_subsample(feature_matrix, context_length, subsampling_rate):
    T, d = feature_matrix.shape
    padded = np.pad(feature_matrix, ((context_length, context_length), (0, 0)), mode='edge')
    spliced = np.concatenate([padded[i:T + i] for i in range(2 * context_length + 1)], axis=1)
    return spliced[::subsampling_rate]


class InputGenerator(object):
    def __init__(self, json_path, batch_size, shuffle, context_length, subsampling_rate):
        # Load the dataset from the json file (utts key)
        with open(json_path, 'r', encoding='utf-8') as f:
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

        # Get the batch indices
        last_batch = self.total_num_steps >= self.steps_per_epoch

        start_idx = (self.steps_per_epoch - 1) * self.batch_size if last_batch else self.current_step * self.batch_size
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
        
        if not last_batch:
            self.current_step += 1
            self.total_num_steps += 1
        
        return batch
