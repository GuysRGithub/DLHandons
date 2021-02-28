from collections import Counter
import logging
import math
import random
import mxnet as mx
import numpy as np


def _load_data(name):
    """
    Load data from file
    :param name: file name
    :return: data, negative, vocab , frequency
    """
    buf = open(name).read()
    tokens = buf.split(' ')
    vocab = {}
    freq = [0]
    data = []
    for token in tokens:
        if len(token) == 0:
            continue
        if token not in vocab:
            vocab[token] = len(vocab) + 1
            freq.append(0)
        wid = vocab[token]
        data.append(wid)
        freq[wid] += 1
    negative = []
    for i, v in enumerate(freq):
        if i == 0 or v < 5:
            continue
        v = int(math.pow(v * 1.0, 0.75))
        negative += [i for _ in range(v)]
    return data, negative, vocab, freq


class SubwordData(object):

    def __init__(self, data, units, weights, negative_units, negative_weight,
                 vocab, units_vocab, freq, max_len):
        """

        :param data: data (list index word)
        :param units:
        :param weights:
        :param negative_units:
        :param negative_weight:
        :param vocab:
        :param units_vocab:
        :param freq:
        :param max_len:
        """
        self.data = data
        self.units = units
        self.weights = weights
        self.negative_units = negative_units
        self.negative_weights = negative_weight
        self.vocab = vocab
        self.units_vocab = units_vocab
        self.freq = freq
        self.max_len = max_len


def _get_subword_units(token, gram):
    """
    Return subword-units presentation, given a word/token.
    :param token:
    :param gram: grammar (range)
    :return: list of token with length gram
    """
    if token == '</s>':
        return [token]
    t = '#' + token + "#"
    return [t[i:i + gram] for i in range(0, len(t) - gram + 1)]


def _get_subword_representation(word_id, vocab_inverse, units_vocab, max_len, gram, padding_char):
    """

    :param word_id:
    :param vocab_inverse:
    :param units_vocab:
    :param max_len:
    :param gram:
    :param padding_char:
    :return: list of unit with length gram and list of weight
    """
    token = vocab_inverse[word_id]
    units = [units_vocab[unit] for unit in _get_subword_units(token, gram)]
    weights = [1] * len(units) + [0] * (max_len - len(units))
    units = units + [units_vocab[padding_char]] * (max_len - len(units))
    return units, weights


def _prepare_subword_units(tks, gram, padding_char):
    """
    :param tks: tokens
    :param gram: gram (1, 2, 3, ..)
    :param padding_char: char to append at the end
    :return: dict units_vocab, int max len
    """
    units_vocab = {padding_char: 1}
    max_len = 0
    unit_set = set()
    logging.info('grams: %d', gram)
    logging.info('counting max len...')
    for tk in tks:
        res = _get_subword_units(tk, gram)
        unit_set.update(i for i in res)
        if max_len < len(res):
            max_len = len(res)
    logging.info('preparing units vocab...')
    for unit in unit_set:
        if len(unit) == 0:
            continue
        if unit not in units_vocab:
            units_vocab[unit] = len(units_vocab)
    return units_vocab, max_len


def _load_data_as_subword_units(name, min_count, gram, max_subwords, padding_char):
    """

    :param name: file name
    :param min_count: min freq
    :param gram:
    :param max_subwords:
    :param padding_char:
    :return: @class SubwordData
    """
    tks = []
    f_read = open(name, 'rb')
    logging.info('reading corpus from file')
    for line in f_read:
        line = line.strip().decode('utf-8')
        tks.extend(line.split(' '))

    logging.info("Total tokens: %d", len(tks))

    tks = [tk for tk in tks if len(tk) <= max_subwords]

    c = Counter(tks)

    logging.info('Total vocab: %d', len(c))

    vocab = {}
    vocab_inverse = {}
    freq = [0]
    data = []

    for tk in tks:
        if len(tk) == 0:
            continue
        if tk not in vocab:
            vocab[tk] = len(vocab)
            freq.append(0)
        word_id = vocab[tk]
        vocab_inverse[word_id] = tk
        data.append(word_id)
        freq[word_id] += 1

    negative = []
    for i, v in enumerate(freq):
        if i == 0 or v < min_count:
            continue
        v = int(math.pow(v * 1.0, 0.75))
        negative += [i for _ in range(v)]

    logging.info('Counting subword units...')
    units_vocab, max_len = _prepare_subword_units(tks, gram, padding_char)
    logging.info('vocabulary size: %d', len(vocab))
    logging.info('subword unit size: %d', len(units_vocab))

    logging.info('generating input data...')
    units = []
    weights = []
    for word_id in data:
        word_units, weight = _get_subword_representation(word_id, vocab_inverse, units_vocab,
                                                         max_len, gram, padding_char)
        units.append(word_units)
        weights.append(weight)

    negative_units = []
    negative_weights = []
    for word_id in negative:
        word_units, weight = _get_subword_representation(word_id, vocab_inverse, units_vocab,
                                                         max_len, gram, padding_char)
        negative_units.append(word_units)
        negative_weights.append(weight)

    return SubwordData(data=data, units=units, weights=weights,
                       negative_units=negative_units, negative_weight=negative_weights,
                       vocab=vocab, units_vocab=units_vocab, freq=freq, max_len=max_len)


class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


# class DataIterWords(mx.io.DataIter):
#     def __init__(self, name, batch_size, num_label):
#         super(DataIterWords, self).__init__()
#         self.batch_size = batch_size
#         self.data, self.negative, self.vocab, self.freq = _load_data(name)
#         self.vocab_size = 1 + len(self.vocab)
#         print("Vocabulary Size: {}".format(self.vocab_size))
#         self.num_label = num_label
#         self.provide_data = [('data', (batch_size, num_label - 1))]
#         self.provide_label = [('label', (self.batch_size, num_label)),
#                               ('label_weight', (self.batch_size, num_label))]
#
#     def sample_ne(self):
#         return self.negative[random.randint(0, len(self.negative) - 1)]
#
#     def __iter__(self):
#         batch_data = []
#         batch_label = []
#         batch_label_weight = []
#         start = random.randint(0, self.num_label - 1)
#         for i in range(start, len(self.data) - self.num_label - start, self.num_label):
#             context = self.data[i: i + self.num_label // 2] \
#                       + self.data[i + 1 + self.num_label // 2: i + self.num_label]
#             target_word = self.data[i + self.num_label // 2]
#             if self.freq[target_word] < 5:
#                 continue
#             target = [target_word] + [self.sample_ne() for _ in range(self.num_label - 1)]
#             target_weight = [1.0] + [0.0 for _ in range(self.num_label - 1)]
#             batch_data.append(context)
#             batch_label.append(target)
#             batch_label_weight.append(target_weight)
#             if len(batch_data) == self.batch_size:
#                 data_all = [mx.nd.array(batch_data)]
#                 label_all = [mx.nd.array(batch_label), mx.nd.array(batch_label_weight)]
#                 data_names = ['data']
#                 label_names = ['label', 'label_weight']
#                 batch_data = []
#                 batch_label = []
#                 batch_label_weight = []
#                 yield SimpleBatch(data_names, data_all, label_names, label_all)
#
#     def reset(self):
#         pass
#
#
# class DataIterLstm(mx.io.DataIter):
#     def __init__(self, name, batch_size, seq_len, num_label, init_states):
#         super(DataIterLstm, self).__init__()
#         self.batch_size = batch_size
#         self.data, self.negative, self.vocab, self.freq = _load_data(name)
#         self.vocab_size = 1 + len(self.vocab)
#         print("Vocabulary Size: {}".format(self.vocab_size))
#         self.seq_len = seq_len
#         self.num_label = num_label
#         self.init_states = init_states
#         self.init_state_names = [x[0] for x in self.init_states]
#         self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]
#         self.provide_data = [('data', (batch_size, seq_len))] + init_states
#         self.provide_label = [('label', (self.batch_size, seq_len, num_label)),
#                               ('label_weight', (self.batch_size, seq_len, num_label))]
#
#     def sample_ne(self):
#         return self.negative[random.randint(0, len(self.negative) - 1)]
#
#     def __iter__(self):
#         batch_data = []
#         batch_label = []
#         batch_label_weight = []
#         for i in range(0, len(self.data) - self.seq_len - 1, self.seq_len):
#             data = self.data[i: i+self.seq_len]
#             label = [[self.data[i+k+1]] \
#                      + [self.sample_ne() for _ in range(self.num_label-1)]\
#                      for k in range(self.seq_len)]
#             label_weight = [[1.0] \
#                             + [0.0 for _ in range(self.num_label-1)]\
#                             for k in range(self.seq_len)]
#
#             batch_data.append(data)
#             batch_label.append(label)
#             batch_label_weight.append(label_weight)
#             if len(batch_data) == self.batch_size:
#                 data_all = [mx.nd.array(batch_data)] + self.init_state_arrays
#                 label_all = [mx.nd.array(batch_label), mx.nd.array(batch_label_weight)]
#                 data_names = ['data'] + self.init_state_names
#                 label_names = ['label', 'label_weight']
#                 batch_data = []
#                 batch_label = []
#                 batch_label_weight = []
#                 yield SimpleBatch(data_names, data_all, label_names, label_all)
#
#     def reset(self):
#         pass


class DataIterSubWords(mx.io.DataIter):
    def __init__(self, f_name, batch_size, num_label, min_count, gram, max_subwords,
                 padding_char):
        super(DataIterSubWords, self).__init__()
        self.batch_size = batch_size
        self.min_count = min_count
        self.subword = _load_data_as_subword_units(f_name, min_count, gram, max_subwords,
                                                   padding_char)
        self.vocab_size = len(self.subword.units_vocab)
        self.num_label = num_label
        self.provide_data = [('data', (batch_size, num_label - 1, self.subword.max_len)),
                             ('mask', (batch_size, num_label - 1, self.subword.max_len, 1))]
        self.provide_label = [('label', (self.batch_size, num_label, self.subword.max_len)),
                              ('label_weight', (self.batch_size, num_label)),
                              ('label_mask', (self.batch_size, num_label, self.subword.max_len, 1))]

    def sample_ne(self):
        return self.subword.negative_units[random.randint(0, len(self.subword.negative_units) - 1)]

    def sample_ne_indices(self):
        return [random.randint(0, len(self.subword.negative_units) - 1)
                for _ in range(self.num_label - 1)]

    def __iter__(self):
        logging.info("DatIter start")
        batch_data = []
        batch_label = []
        batch_data_mask = []
        batch_label_mask = []
        batch_label_weight = []
        start = random.randint(0, self.num_label - 1)
        for i in range(start, len(self.subword.units) - self.num_label - start, self.num_label):
            context_units = self.subword.units[i: i + self.num_label // 2] + \
                            self.subword.units[i + 1 + self.num_label // 2: i + self.num_label]
            context_mask = self.subword.weights[i: i + self.num_label // 2] + \
                           self.subword.weights[i + 1 + self.num_label // 2: i + self.num_label]
            target_units = self.subword.units[i + self.num_label // 2]
            target_word = self.subword.data[i + self.num_label // 2]
            if self.subword.freq[target_word] < self.min_count:
                continue
            indices = self.sample_ne_indices()
            target = [target_units] + [self.subword.negative_units[i] for i in indices]
            target_weight = [1.0] + [0.0 for _ in range(self.num_label - 1)]
            target_mask = [self.subword.weights[i + self.num_label // 2]] + \
                          [self.subword.negative_weights[i] for i in indices]

            batch_data.append(context_units)
            batch_data_mask.append(context_mask)
            batch_label.append(target)
            batch_label_weight.append(target_weight)
            batch_label_mask.append(target_mask)

            if len(batch_data) == self.batch_size:
                batch_data_mask = np.reshape(batch_data_mask,
                                              (self.batch_size, self.num_label - 1,
                                               self.subword.max_len, 1))
                batch_label_mask = np.reshape(batch_label_mask,
                                              (self.batch_size, self.num_label,
                                               self.subword.max_len, 1))
                data_all = [mx.nd.array(batch_data), mx.nd.array(batch_data_mask)]
                label_all = [mx.nd.array(batch_label),
                             mx.nd.array(batch_label_weight),
                             mx.nd.array(batch_label_mask)]
                data_names = ['data', 'mask']
                label_names = ['label', 'label_weight', 'label_mask']
                batch_data = []
                batch_data_mask = []
                batch_label = []
                batch_label_weight = []
                batch_label_mask = []
                yield SimpleBatch(data_names, data=data_all, label_names=label_names,
                                  label=label_all)

    def reset(self):
        pass
