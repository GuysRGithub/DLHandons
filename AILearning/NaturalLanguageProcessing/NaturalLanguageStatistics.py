from d2l import AllDeepLearning as d2l
from mxnet import nd
import random
import re
import AI.AILearning.RecurrentNeuronNetwork.TextPregropress as AI


def read_time_machine():
    with open('E:/Python_Data/Others/timemachine.txt', 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line.strip().lower()) for line in lines]


tokens = d2l.tokenize(read_time_machine())
vocab = d2l.Vocab(tokens)

freqs = [freq for token, freq in vocab.token_freqs]
bigram_tokens = [[pair for pair in zip(line[:-1], line[1:])] for line in tokens]
bigram_vocab = d2l.Vocab(bigram_tokens)

trigram = [[triple for triple in zip(line[:-2], line[1:-1], line[2:])] for line in tokens]
trigram_vocab = d2l.Vocab(trigram)
bigram_freqs = [freq for _, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for _, freq in trigram_vocab.token_freqs]


# d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token', ylabel='frequency',
#          xscale='log', yscale='log', legend=['unigram', 'bigram', 'trigram'])
# d2l.plt.show()


def seq_data_iter_random(corpus, batch_size, num_steps):
    corpus = corpus[random.randint(0, num_steps):]
    num_examples = ((len(corpus) - 1) // num_steps)
    example_indices = list(range(0, num_examples * num_steps, num_steps))
    random.shuffle(example_indices)
    data = lambda pos: corpus[pos: pos + num_steps]
    num_batches = num_examples // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        batch_indices = example_indices[i: i + batch_size]
        X = [data(j) for j in batch_indices]
        Y = [data(j + 1) for j in batch_indices]
        yield nd.array(X), nd.array(Y)


def seq_data_iter_consecutive(corpus, batch_size, num_steps):
    offset = random.randint(0, num_steps)
    num_indices = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = nd.array(corpus[offset: offset+num_indices])
    Ys = nd.array(corpus[offset+1: offset+1+num_indices])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_batches * num_steps, num_steps):
        X = Xs[:, i: i+num_steps]
        Y = Ys[:, i: i+num_steps]
        print("X: ", X, "\nY: ", Y)
        yield X, Y


class SeqDataLoader(object):
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            data_iter_fn = d2l.seq_data_iter_random
        else:
            data_iter_fn = d2l.seq_data_iter_consecutive
        self.corpus, self.vocab = AI.load_corpus_time_machine(max_tokens)
        self.get_iter = lambda: data_iter_fn(self.corpus, batch_size=batch_size, num_steps=num_steps)

    def __iter__(self):
        return self.get_iter()


def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


data, data_vocab = load_data_time_machine(2, 6, True)

# print(data_vocab.token_freqs[:10])
