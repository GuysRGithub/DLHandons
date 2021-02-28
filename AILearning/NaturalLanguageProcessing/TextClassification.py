from d2l import AllDeepLearning as d2l
from mxnet import gluon, nd
import os

data_dir = "E:/Python_Data/aclImdb/"


def read_imdb(data_dir, is_train):
    data, labels = [], []
    for label in ['pos/', 'neg/']:
        folder_name = data_dir + ('train/' if is_train else 'test/') + label
        for file in os.listdir(folder_name):
            with open(folder_name + file, 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels


def load_data_imdb(batch_size, num_steps=500):
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = nd.array([d2l.trim_pad(vocab[line], num_steps, vocab.unk) for line in train_tokens])
    test_features = nd.array([d2l.trim_pad(vocab[line], num_steps, vocab.unk) for line in test_tokens])
    train_iter = d2l.load_array((train_features, train_data[1]), batch_size)
    test_iter = d2l.load_array((test_features, test_data[1]), batch_size, False)
    return train_iter, test_iter, vocab


train_data = read_imdb(data_dir, True)
# print('# trainings:', len(train_data[0]))
# for x, y in zip(train_data[0][:3], train_data[1][:3]):
#     print('label:', y, 'review:', x[:60])

train_tokens = d2l.tokenize(train_data[0], token='word')
vocab = d2l.Vocab(train_tokens, min_freq=5)

# d2l.set_figsize((3.5, 2.5))
# d2l.plt.hist([len(line) for line in train_tokens], bins=range(0, 1000, 50))
# d2l.plt.show()
num_steps = 500
train_features = nd.array([d2l.trim_pad(vocab[line], num_steps, vocab.unk) for line in train_tokens])
train_iter = d2l.load_array((train_features, train_data[1]), 64)
# for X, y in train_iter:
#     print('X', X.shape, 'y', y.shape)
#     break
# print('batch:', len(train_iter))


"""
Text classification can classify a text sequence into a category.

To classify a text sentiment, we load an IMDb dataset and tokenize its words. Then we pad the text sequence for short 
reviews and create a data iterator. 

"""