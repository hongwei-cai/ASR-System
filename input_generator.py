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
    # spliced_features = np.concatenate([np.tile(feature_matrix[0], (context_length, 1)), feature_matrix, np.tile(feature_matrix[-1], (context_length, 1))], axis=0)
    
    # Subsampling
    subsampled_features = spliced_features[::subsampling_rate, :]
    return subsampled_features

