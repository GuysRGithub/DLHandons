from d2l import AllDeepLearning as d2l
import math
from mxnet import gluon, nd, init
import random
import tarfile
import os

"""As its name implies, a word vector is a vector used to represent a word. It can also be thought of as the feature 
vector of a word. The technique of mapping words to vectors of real numbers is also known as word embedding.
"""
"""
cosine similarity: (x⊤y) / (∥x∥∥y∥)∈[−1,1].

Softmax operation: P(wo∣wc)=exp(u⊤ovc) / ∑i∈V(exp(u⊤ivc)),
Maximum likelihood estimation: −∑t=1->T ∑−m≤j≤m,j≠0 (logP(w(t+j)∣w(t))).

//////////////                SKip-Gram model          /////////////////// vi: central word, ui: context word * logP(
wo∣wc)=u⊤ovc−log(∑i∈V exp(u⊤ivc)). (index o, i, c) * Differentiation (vc): ∂logP(wo∣wc) / ∂vc =uo − (∑j∈V exp(
u⊤jvc)uj / ∑i∈V exp(u⊤ivc)) =uo − (∑j∈V(exp(u⊤jvc) / ∑i∈V exp(u⊤ivc)))uj =uo − ∑j∈V P(wj ∣ wc)uj. //////////////      
          Continuous bag of word model          /////////////////// vi: context word, ui: central word P(wc∣wo1,…,
          wo2m) = exp(1/2m * u⊤c(vo1+…+vo2m)) / ∑i∈V exp(1/2m * u⊤i(vo1+…+vo2m)). ( index c, o) Loss function: 
          −∑t=1->T logP(w(t)∣w(t−m),…,w(t−1),w(t+1),…,w(t+m)). logP(wc∣Wo)=u⊤cv¯o−log(∑i∈V exp(u⊤iv¯o)). * 
          Differentiation: ∂logP(wc∣Wo)∂voi =1/2m (uc−∑j∈V (exp(u⊤jv¯o)uj / ∑i∈V exp(u⊤iv¯o))) =1/2m (uc−∑j∈V P(
          wj∣Wo)uj).
          
           
/////////////          Summary word2vec           ////////////////////
 
          - A word vector is a vector used to represent a word. The technique of mapping words to vectors of real 
          numbers is also known as word embedding. - Word2vec includes both the continuous bag of words (CBOW) and 
          skip-gram models. The skip-gram model assumes that context words are generated based on the central target 
          word. The CBOW model assumes that the central target word is generated based on the context words. 

//////////////////////                Approximate Training Method       /////////////////// 

******  Negative sampling
Loss: −logP(w(t+j)∣w(t))
                        =−logP(D=1∣w(t),w(t+j))−∑k=1,wk∼P(w)->K logP(D=0∣w(t),wk)
                        =−log σ(u⊤i(t+j)vit)−∑k=1, wk∼P(w)->K log(1−σ(u.⊤hk * vit))
                        =−log σ(u⊤i(t+j)vit)−∑k=1, wk∼P(w)->K log σ(−u⊤hk vit).
                        ( index i_t+j, i_t, h_k,.. ) - _: index of index
                        σ: sigmoid function

***** Hierarchical Softmax
Loss: P(wo∣wc)=∏j=1->L(wo)−1 σ([[n(wo,j+1)=leftChild(n(wo,j))]] ⋅ u⊤n(wo,j)vc),

/////////////          Summary Approximate Training Method           //////////////////// 
            - Negative sampling 
            constructs the loss function by considering independent events that contain both positive and negative 
            examples. The 
            gradient computational overhead for each step in the training process is linearly related to the number of 
            noise 
            words we sample. 
            - Hierarchical softmax uses a binary tree and constructs the loss function based on the path 
            from the 
            root node to the leaf node. The gradient computational overhead for each step in the training process is 
            related to the logarithm of the dictionary size.
             
/////////////          Summary Approximate Training Method           //////////////////// 

            Subsampling attempts to minimize the impact of high-frequency words on the training of a word embedding model.

"""

url = "http://d2l-data.s3-accelerate.amazonaws.com/ptb.zip"
shah = '319d85e578af0cdc590547f26231e4e31cdf1e42'
data_dir = "E:/Python_Data/Word2vec/"


def read_pth():
    """
    Read file train, test,..
    :return: list of words
    """
    path = data_dir + "ptb/"
    with open(path + 'ptb.train.txt') as f:
        raw_text = f.read()
        return [line.split() for line in raw_text.split('\n')]


sentences = read_pth()
vocab = d2l.Vocab(sentences, min_freq=10)
# print('vocab size: %d' % len(vocab))
"""
Drop out prob: P(wi)=max(1−sqrt(t/f(wi)),0),
"""


def sub_sampling(sentences, vocab):
    """
    Reduce the effect of high-frequency words
    :param sentences:
    :param vocab:
    :return: words of sentences when reduce frequency of common word like a, the,...
    """
    # Map low frequency words into <unk>

    sentences = [[vocab.idx_to_token[vocab[tk]] for tk in line] for line in sentences]  # return words from sentences
    counter = d2l.count_corpus(sentences)
    num_tokens = sum(counter.values())

    def keep(token):
        # reduce token when have much frequency with random uniform :'))
        return random.uniform(0, 1) < math.sqrt(1e-4 / counter[token] * num_tokens)

    return [[tk for tk in line if (keep(tk))] for line in sentences]


subsampled = sub_sampling(sentences, vocab)


def compare_count(token):
    return '# of "%s": before=%d, after=%d' % \
           (token,
            sum(line.count(token) for line in sentences),
            sum(line.count(token) for line in subsampled))


def get_center_and_contexts(corpus, max_window_size):
    # It use index instead of word to compute
    centers, contexts = [], []
    for line in corpus:
        # Each sentence needs at least 2 words to form a
        # "central target word - context word" pair
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts


class RandomGenerator(object):
    def __init__(self, sampling_weights):
        self.population = list(range(len(sampling_weights)))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            self.candidates = random.choices(self.population, self.sampling_weights,
                                             k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i-1]


def get_negatives(all_contexts, corpus, K):
    counter = d2l.count_corpus(corpus)
    sampling_weights = [counter[i]**0.75 for i in range(len(counter))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # Noise words cannot be context words
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives


def batchify(data):
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (nd.array(centers).reshape(-1, 1), nd.array(contexts_negatives),
            nd.array(masks), nd.array(labels))


# x_1 = (1, [2, 2], [3, 3, 3, 3])
# x_2 = (1, [2, 2, 2], [3, 3])
# batch = batchify((x_1, x_2))
names = ['centers', 'contexts_negatives', 'masks', 'labels']
# for name, data in zip(names, batch):
#     print(name, "=", data)

# tiny_data_set = [list(range(7)), list(range(7, 10))]
# print("dataset", tiny_data_set)
# for center, context in zip(*get_center_and_contexts(tiny_data_set, 2)):
#     print('center', center, 'has contexts', context)
# corpus = [vocab[line] for line in subsampled]
# #
# all_centers, all_contexts = get_center_and_contexts(corpus, 5)
# all_negatives = get_negatives(all_contexts, corpus, 5)

# d2l.set_figsize((3.5, 2.5))
# d2l.plt.hist([[len(line) for line in sentences],
#               [len(line) for line in subsampled]])
# d2l.plt.xlabel('# token per sentences')
# d2l.plt.ylabel("count")
# d2l.plt.legend(['origin', 'subsampled'])
# d2l.plt.show()


def load_data_ptb(batch_size, max_window_size, num_noise_words):
    sentences = read_pth()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled = sub_sampling(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_center_and_contexts(corpus, max_window_size)
    all_negatives = get_negatives(all_contexts, corpus, num_noise_words)
    dataset = gluon.data.ArrayDataset(all_centers, all_contexts, all_negatives)
    data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True, batchify_fn=batchify)
    return data_iter, vocab


# data_iter, vocab = load_data_ptb(512, 5, 5)
# for batch in data_iter:
#     for name, data in zip(names, batch):
#         print(name, '=', data.shape)
#     break




