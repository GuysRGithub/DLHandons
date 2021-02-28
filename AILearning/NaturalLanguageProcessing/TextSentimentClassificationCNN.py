from d2l import AllDeepLearning as d2l
from mxnet import gluon, nd, init
from mxnet.gluon import nn
from mxnet.contrib import text
from AI.AILearning.NaturalLanguageProcessing import TextClassification as loader

batch_size = 64
train_iter, test_iter, vocab = loader.load_data_imdb(batch_size)


def corr1d(X, K):
    w = K.shape[0]
    Y = nd.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i: i + w] * K).sum()
    return Y


def corr1d_multi_in(X, K):
    # First, we traverse along the 0th dimension (channel dimension) of X and
    # K. Then, we add them together by using * to turn the result list into a
    # positional argument of the add_n function
    return sum(corr1d(x, k) for x, k in zip(X, K))


"""  ###########################                TextCNN           ###################################
    Define multiple one-dimensional convolution kernels and use them to perform convolution calculations on the 
    inputs. Convolution kernels with different widths may capture the correlation of different numbers of adjacent words. 

    Perform max-over-time pooling on all output channels, and then concatenate the pooling output values of these channels 
    in a vector.

    The concatenated vector is transformed into the output for each category through the fully connected layer. A dropout 
    layer can be used in this step to deal with overfitting. 
"""


class TextCNN(nn.Block):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels, **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # The embedding layer does not participate in training
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.drop_out = nn.Dropout(0.5)
        self.decoder = nn.Dense(2)
        # The max-over-time pooling layer has no weight, so it can share an
        # instance
        self.pool = nn.GlobalAvgPool1D()
        # Create multiple one-dimensional convolutional layers
        self.conv = nn.Sequential()
        for c, k in zip(num_channels, kernel_sizes):
            self.conv.add(nn.Conv1D(c, k, activation='relu'))

    def forward(self, inputs):
        # Concatenate the output of two embedding layers with shape of
        # (batch size, number of words, word vector dimension) by word vector
        embeddings = nd.concat(*(self.embedding(inputs), self.constant_embedding(inputs)))
        # According to the input format required by Conv1D, the word vector
        # dimension, that is, the channel dimension of the one-dimensional
        # convolutional layer, is transformed into the previous dimension
        embeddings = embeddings.transpose((0, 2, 1))
        # For each one-dimensional convolutional layer, after max-over-time
        # pooling, an ndarray with the shape of (batch size, channel size, 1)
        # can be obtained. Use the flatten function to remove the last
        # dimension and then concatenate on the channel dimension
        encoding = nd.concat(*[nd.squeeze(
            self.pool(conv(embeddings)), axis=-1) for conv in self.conv], dim=1)
        outputs = self.decoder(encoding)
        return outputs


embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
ctx = d2l.try_all_gpus()
net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)
net.initialize(init.Xavier(), ctx=ctx)


glove_embedding = text.embedding.create('glove', pretrained_file_name='glove.6B.100d.txt')
embeds = glove_embedding.get_vecs_by_tokens(vocab.idx_to_token)
net.embedding.weight.set_data(embeds)
net.constant_embedding.weight.set_data(embeds)
net.constant_embedding.collect_params().setattr('grad_req', 'null')

lr, num_epochs = 0.001, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch12(net, train_iter, test_iter, loss, trainer, num_epochs, ctx)
d2l.plt.show()







