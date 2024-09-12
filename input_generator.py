import numpy as np
import kaldiio


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
    
