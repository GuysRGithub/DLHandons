from d2l import AllDeepLearning as d2l
from mxnet import nd
from mxnet.gluon import nn, rnn
import AI.AILearning.NaturalLanguageProcessing.NaturalLanguageStatistics as Ai
start = d2l.Timer()
batch_size, num_step = 32, 25
train_iter, vocab = Ai.load_data_time_machine(batch_size, num_step)
i = 0
num_hiddens = 256
rrn_layer = rnn.RNN(num_hiddens)
rrn_layer.initialize()
batch_size = 1
state = rrn_layer.begin_state(batch_size=batch_size)
print(len(state), state[0].shape)
num_step = 1
X = nd.random.uniform(shape=(num_step, batch_size, len(vocab)))
Y, state_new = rrn_layer(X, state)
print(Y.shape, len(state_new), state_new[0].shape)


class RNNModel(nn.Block):
    def __init__(self, rrn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rrn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size)

    def forward(self, inputs, state):
        X = nd.one_hot(inputs.T, self.vocab_size)
        Y, state = self.rnn(X, state)
        output = self.dense(Y.reshape(-1, Y.shape[-1]))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


ctx = d2l.try_gpu()
model = RNNModel(rrn_layer, len(vocab))
model.initialize(force_reinit=True, ctx=ctx)
num_epochs, lr = 500, 1
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, ctx)
d2l.plt.show()
print(d2l.Timer().stop() - start)
