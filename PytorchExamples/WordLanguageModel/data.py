import os
from io import open
import torch


class Dictionary(object):
    def __init__(self):
        self.word_2_idx = {}
        self.idx_2_word = []

    def add_word(self, word):
        if word not in self.word_2_idx:
            self.idx_2_word.append(word)
            self.word_2_idx[word] = len(self.idx_2_word) - 1
        return self.word_2_idx[word]

    def __len__(self):
        return len(self.idx_2_word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))

    def tokenize(self, path):
        """Tokenizes text file"""
        assert os.path.exists(path)
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                words = line.split() + ['<eos>']
                [self.dictionary.add_word(word) for word in words]

        with open(path, 'r', encoding='utf8') as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                [ids.append(self.dictionary.word_2_idx[word]) for word in words]
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)
        return idss


