import itertools
import os
import re
from collections import Counter

import numpy as np


# import word2vec


def clean_str(string):
    string = re.sub(r"[^a-zA-Z0-9(),!?\'`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", r" \( ", string)
    string = re.sub(r"\)", r" \) ", string)
    string = re.sub(r"\?", r" \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()


def load_data_and_labels():
    pos_path = "./data/rt-polaritydata/rt-polarity.pos"
    neg_path = "./data/rt-polaritydata/rt-polarity.neg"
    if not os.path.exists(pos_path):
        os.system("git clone https://github.com/dennybritz/cnn-text-classification-tf.git")
        os.system('mv cnn-text-classification-tf/data .')
        os.system('rm -rf cnn-text-classification-tf')
    positive_examples = list(open(pos_path).readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(neg_path).readlines())
    negative_examples = [s.strip() for s in negative_examples]

    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]

    positive_labels = [1 for _ in positive_examples]
    negative_labels = [0 for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def pad_sentences(sentences, padding_word="</s>"):
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i, sentence in enumerate(sentences):
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """Maps sentences and labels to vectors based on a vocabulary."""
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def build_input_data_with_word2vec(sentences, labels, word2vec_list):
    """
    Map sentences and labels to vectors based on a pretrained word2vec
    :param sentences:
    :param labels:
    :param word2vec_list:
    :return:
    """

    x_vec = []
    for sent in sentences:
        vec = []
        for word in sent:
            if word in word2vec_list:
                vec.append(word2vec_list[word])
            else:
                vec.append(word2vec_list['</s>'])
        x_vec.append(vec)
    x_vec = np.array(x_vec)
    y_vec = np.array(labels)
    return [x_vec, y_vec]


def load_data_with_word2vec(word2vec_list):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    :param word2vec_list:
    :return:
    """

    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    return build_input_data_with_word2vec(sentences_padded, labels, word2vec_list)


def load_data():
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]


def batch_iter(data, batch_size, num_epochs):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def load_pretrained_word2vec(in_file):
    if isinstance(in_file, str):
        in_file = open(in_file)

    word2vec_list = {}

    for idx, line in enumerate(in_file):
        if idx == 0:
            vocab_size, dim = line.strip().split()
        else:
            tokens = line.strip().split()
            word2vec_list[tokens[0]] = map(float, tokens[1:])

    return word2vec_list