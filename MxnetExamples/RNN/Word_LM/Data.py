import os, gzip
import sys
import mxnet as mx
import numpy as np


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.word_count = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.word_count.append(0)
        index = self.word2idx[word]
        self.word_count[index] += 1
        return index

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(path + 'train.txt')
        self.valid = self.tokenize(path + 'valid.txt')
        self.test = self.tokenize(path + 'test.txt')

    def tokenize(self, path):
        """Tokenizes a text file"""
        assert os.path.exists(path)
        # Add word to dic
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenizes file content
        with open(path, 'r') as f:
            ids = np.zeros((tokens,), dtype=np.int32)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return mx.nd.array(ids, dtype=np.int32)


def batchify(data, batch_size):
    """ Reshape data into (num_example, batch_size)"""
    n_batch = data.shape[0] // batch_size
    data = data[:n_batch * batch_size]
    data = data.reshape((batch_size, n_batch)).T
    return data


class CorpusIter(mx.io.DataIter):
    """An iterator that returns the a batch of sequence each time"""
    def __init__(self, source, batch_size, bptt):
        super(CorpusIter, self).__init__()
        self.batch_size = batch_size
        self.provide_data = [('data', (bptt, batch_size), np.int32)]
        self.provide_label = [('label', (bptt, batch_size))]
        self._index = 0
        self._bptt = bptt
        self._source = batchify(source, batch_size)

    def iter_next(self):
        i = self._index
        if i + self._bptt > self._source.shape[0] - 1:
            return False
        self._next_data = self._source[i:i+self._bptt]
        self._next_label = self._source[i+1:i+1+self._bptt].astype(np.float32)
        self._index += self._bptt
        return True

    def next(self):
        if self.iter_next():
            return mx.io.DataBatch(data=self.getdata(), label=self.getlabel())
        else:
            raise StopIteration

    def reset(self):
        self._index = 0
        self._next_data = None
        self._next_label = None

    def getdata(self):
        return [self._next_data]

    def getlabel(self):
        return [self._next_label]