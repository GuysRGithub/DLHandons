from d2l import AllDeepLearning as d2l
from mxnet import nd, init, autograd, gluon
from mxnet.gluon import nn, rnn
from mxnet.contrib import text
from AI.AILearning.NaturalLanguageProcessing import TextClassification as loader

"""
    In this model, each word first obtains a feature vector from the embedding layer. Then, we further encode the 
        feature sequence using a bidirectional recurrent neural network to obtain sequence information. Finally, we transform 
        the encoded sequence information to output through the fully connected layer. Specifically, we can concatenate hidden 
        states of bidirectional long-short term memory in the initial timestep and final timestep and pass it to the output 
        layer classification as encoded feature sequence information. In the BiRNN class implemented below, the Embedding 
        instance is the embedding layer, the LSTM instance is the hidden layer for sequence encoding, and the Dense instance 
        is the output layer for generated classification results.
        
         
"""
batch_size = 64
train_iter, test_iter, vocab = loader.load_data_imdb(batch_size)


class BiRNN(nn.Block):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Set Bidirectional to True to get a bidirectional recurrent neural
        # network
        self.encoder = rnn.LSTM(num_hiddens, num_layers,
                                bidirectional=True, input_size=embed_size)
        self.decoder = nn.Dense(2)

    def forward(self, inputs):
        # The shape of inputs is (batch size, number of words). Because LSTM
        # needs to use sequence as the first dimension, the input is
        # transformed and the word feature is then extracted. The output shape
        # is (number of words, batch size, word vector dimension).
        embeddings = self.embedding(inputs.T)
        # Since the input (embeddings) is the only argument passed into
        # rnn.LSTM, it only returns the hidden states of the last hidden layer
        # at different timestep (outputs). The shape of outputs is
        # (number of words, batch size, 2 * number of hidden units).
        outputs = self.encoder(embeddings)
        # Concatenate the hidden states of the initial timestep and final
        # timestep to use as the input of the fully connected layer. Its
        # shape is (batch size, 4 * number of hidden units)
        encoding = nd.concat(*(outputs[0], outputs[-1]), dim=1)
        outs = self.decoder(encoding)
        return outs


def predict_sentiment(net, vocab, sentence):
    sentence = nd.array(vocab[sentence.split()], ctx=d2l.try_gpu())
    label = nd.argmax(net(sentence.reshape(1, -1)), axis=1)
    return 'positive' if label == 1 else 'negative'


embed_size, num_hiddens, num_layers, ctx = 100, 100, 2, d2l.try_gpu()
net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
net.initialize(init.Xavier(), ctx=ctx)

glove_embedding = text.embedding.create('glove', pretrained_file_name='glove.6B.100d.txt')
embeds = glove_embedding.get_vecs_by_tokens(vocab.idx_to_token)
print(embeds.shape)

net.embedding.weight.set_data(embeds)
net.embedding.collect_params().setattr('grad_req', 'null')
lr, num_epochs = 0.01, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch12(net, train_iter, test_iter, loss, trainer, num_epochs, ctx)

"""        #####################             SUMMARY             ################################
    Text classification transforms a sequence of text of indefinite length into a category of text. This is a downstream 
    application of word embedding.

    We can apply pre-trained word vectors and recurrent neural networks to classify the emotions in a text.
"""
